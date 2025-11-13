import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from flax import nnx
import jax

from architectures.node import NeuralODE
from functionals.linear_funcitonal_class import LinearPotential as LinearFunctional
from functionals.internal_functional_class import (
    InternalPotential as InternalFunctional,
)
from functionals.interaction_functional_class import (
    InteractionPotential as InteractionFunctional,
)
from parametric_model.parametric_model import ParametricModel


class Potential:
    """
    A class to manage the three potentials: linear, internal, interaction
    1. Linear potential: F(ρ) = ∫ U(x)ρ(x)dx
    2. Internal potential: F(ρ) = ∫ f(ρ(x))dx
    3. Interaction potential: F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy
    where U is a user-defined potential function, f is a user-defined function, and W is a user-defined interaction function.
    The current implementation only support the entropy and Fisher information for the internal potential.
    These quantities are computed by solving ODEs, and their computation is done in the node.py file.
    """

    def __init__(
        self,
        linear: Optional[LinearFunctional] = None,
        internal: Optional[InternalFunctional] = None,
        interaction: Optional[InteractionFunctional] = None,
    ):
        self.linear = linear
        self.internal = internal
        self.interaction = interaction

    def evaluate_energy(
        self,
        parametric_model: ParametricModel,
        z_samples: Array,
        params: Optional[PyTree] = None,
    ) -> float:
        """
        Evaluate the total energy functional F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ)
        Args:
            parametric_model: ParametricModel instance
            z_samples: Reference samples (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy: Total energy functional
            x_samples: Transformed samples (batch_size, d)
            linear_energy: Linear potential energy
            internal_energy: Internal potential energy
            interaction_energy: Interaction potential energy
        """

        if params is None:
            _, params = nnx.split(parametric_model)

        # Transform reference samples. Only if internal potential is used, we need the trajectory
        if self.internal is not None:
            z_trajectory, time_steps = parametric_model(
                z_samples, history=True, params=params
            )  # (batch_size, time_steps, dim)
            x_samples = z_trajectory[:, -1, :]
        else:
            x_samples = parametric_model(z_samples, params=params)  # (batch_size, dim)
        energy = 0.0
        linear_energy = 0.0
        internal_energy = 0.0
        interaction_energy = 0.0
        # Linear potential
        if self.linear is not None:
            linear_energy, _ = self.linear.evaluate_energy(
                parametric_model, z_samples, x_samples=x_samples
            )
            linear_energy = linear_energy * self.linear.coeff
            energy += linear_energy
        # Internal potential
        if self.internal is not None:
            internal_energy = self.internal(
                parametric_model=parametric_model,
                z_samples=z_samples,
                z_trajectory=z_trajectory,
                time_steps=time_steps,
                params=params,
            )
            energy += internal_energy
        # Interaction potential
        if self.interaction is not None:
            batch_size = z_samples.shape[0]
            # Obtain samples for interaction energy computation
            part1_samples = x_samples[: batch_size // 2, :]
            part2_samples = x_samples[batch_size // 2 :, :]
            if part1_samples.shape[0] < part2_samples.shape[0]:
                part2_samples = part2_samples[: part1_samples.shape[0], :]
            elif part2_samples.shape[0] < part1_samples.shape[0]:
                part1_samples = part1_samples[: part2_samples.shape[0], :]
            interaction_energy, _ = self.interaction.evaluate_energy(
                parametric_model,
                z_samples,
                x_samples=part1_samples,
                y_samples=part2_samples,
            )
            energy += interaction_energy * self.interaction.coeff

        return energy, x_samples, linear_energy, internal_energy, interaction_energy

    def compute_energy_gradient(
        self, parametric_model: ParametricModel, z_samples: Array, params: PyTree
    ) -> PyTree:
        """
        Compute the gradient of the total energy functional F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ)
        Args:
            parametric_model: ParametricModel instance
            z_samples: Reference samples (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy_gradient: Gradient of the total energy functional
        """

        if params is None:
            _, params = nnx.split(parametric_model)

        def energy_evaluation(p: PyTree) -> Array:
            # Transform reference samples
            if self.internal is not None:
                z_trajectory, time_steps = parametric_model(
                    z_samples, history=True, params=p
                )  # (batch_size, time_steps, dim)
                x_samples = z_trajectory[:, -1, :]
            else:
                x_samples = parametric_model(z_samples, params=p)  # (batch_size, dim)
            energy = 0.0
            linear_energy = 0.0
            internal_energy = 0.0
            interaction_energy = 0.0
            # Linear potential
            if self.linear is not None:

                linear_energy, _ = self.linear.evaluate_energy(
                    parametric_model, z_samples, x_samples=x_samples
                )
                linear_energy = linear_energy * self.linear.coeff
                energy += linear_energy
            # Internal potential
            if self.internal is not None:
                internal_energy = self.internal(
                    parametric_model=parametric_model,
                    z_samples=z_samples,
                    z_trajectory=z_trajectory,
                    time_steps=time_steps,
                    params=p,
                )

                energy += internal_energy
            # Interaction potential
            if self.interaction is not None:
                batch_size = z_samples.shape[0]
                # Obtain samples for interaction energy computation
                part1_samples = x_samples[: batch_size // 2, :]
                part2_samples = x_samples[batch_size // 2 :, :]
                if part1_samples.shape[0] < part2_samples.shape[0]:
                    part2_samples = part2_samples[: part1_samples.shape[0], :]
                elif part2_samples.shape[0] < part1_samples.shape[0]:
                    part1_samples = part1_samples[: part2_samples.shape[0], :]
                interaction_energy, _ = self.interaction.evaluate_energy(
                    parametric_model,
                    z_samples,
                    x_samples=part1_samples,
                    y_samples=part2_samples,
                    params=p,
                )

                energy += interaction_energy * self.interaction.coeff

            energy_breakdown = {
                "internal_energy": internal_energy,
                "linear_energy": linear_energy,
                "interaction_energy": interaction_energy,
            }
            return energy, energy_breakdown

        (energy, energy_breakdown), energy_grad = jax.value_and_grad(
            energy_evaluation, has_aux=True
        )(params)

        return energy_grad, energy, energy_breakdown
