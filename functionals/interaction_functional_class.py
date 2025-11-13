import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union, Callable
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy.stats as stats

from parametric_model.parametric_model import ParametricModel


class InteractionPotential:
    """
    A class for handling interaction energy functionals F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy
    where W is a user-defined interaction function.
    """

    def __init__(
        self,
        interaction_fn: Union[str, Callable[[Array, Array], Array]],
        coeff: float = 1.0,
        **interaction_kwargs
    ):
        """
        Initialize InteractionPotential with an interaction function.

        Args:
            interaction_fn: Function that takes positions (batch_size, d) and (batch_size, d) and returns
                         interaction values (batch_size, batch_size). Alternatively, can be 'gaussian' or 'coulomb'.
            coeff: Coefficient for the functional
            **interaction_kwargs: Additional keyword arguments for the interaction function
        """
        self.interaction_fn = interaction_fn
        self.interaction_kwargs = interaction_kwargs
        self.coeff = coeff

    def __call__(self, x: Array, y: Array) -> Array:
        """
        Evaluate interaction function at given positions.

        The functionals are assumed to be shift-invariant, i.e., W(x,y) = W(x-y).

        Args:
            x: Positions array of shape (batch_size, d)
            y: Positions array of shape (batch_size, d)
        Returns:
            Interaction values of shape (batch_size, )
        """
        # The correct thing to do would be to compute pairwise differences
        # But for simplicity, we assume x and y have the same shape and compute element-wise

        z = x - y  # (batch_size, d)

        return self.interaction_fn(z, **self.interaction_kwargs)

    def compute_energy_gradient(
        self,
        parametric_model: ParametricModel,
        z_samples: Array,
        params: Optional[PyTree] = None,
    ) -> PyTree:
        """
        Compute gradient of energy functional

        Args:
            parametric_model: ParametricModel instance
            x_samples: Samples from the distribution ρ, shape (batch_size, d)
            y_samples: Samples from the distribution ρ, shape (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy_grad: Gradient of the energy functional w.r.t. parameters, shape (batch_size, d)
        """

        if params is None:
            _, params = nnx.split(parametric_model)

        def energy_functional(p: PyTree) -> Array:
            # with half of the z-samples define x, with the other half define y
            batch_size = z_samples.shape[0]
            mid_point = batch_size // 2
            x_samples = parametric_model(
                z_samples[:mid_point, :], params=p
            )  # (batch_size/2, d)
            y_samples = parametric_model(
                z_samples[mid_point:, :], params=p
            )  # (batch_size/2, d)
            # Make sure x_samples and y_samples have the same length
            if len(x_samples) < len(y_samples):
                y_samples = y_samples[: len(x_samples), :]
            elif len(y_samples) < len(x_samples):
                x_samples = x_samples[: len(y_samples), :]
            # Evaluate the potential
            potential_vals = self(x_samples, y_samples)  # (batch_size/2, batch_size/2)
            energy = 0.5 * jnp.mean(potential_vals)
            return energy

        values, grad = jax.value_and_grad(energy_functional)(params)

        return grad, values

    def evaluate_energy(
        self,
        parametric_model: ParametricModel,
        z_samples: Array,
        x_samples: Optional[Array] = None,
        y_samples: Optional[Array] = None,
        params: Optional[PyTree] = None,
    ) -> tuple[Array, Array]:
        """
        Evaluate current energy F(ρ_θ)

        Args:
            node: Neural ODE model
            z_samples: Reference samples
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy: Current energy value
            x_samples: Pushforward samples through the flow
        """
        if params is None:
            _, params = nnx.split(parametric_model)
        if x_samples is None or y_samples is None:
            # with half of the z-samples define x, with the other half define y
            batch_size = z_samples.shape[0]
            mid_point = batch_size // 2
            x_samples = parametric_model(
                z_samples[:mid_point, :], params=params
            )  # (batch_size/2, d)
            y_samples = parametric_model(
                z_samples[mid_point:, :], params=params
            )  # (batch_size/2, d)
            # Make sure x_samples and y_samples have the same length
            if len(x_samples) < len(y_samples):
                y_samples = y_samples[: len(x_samples), :]
            elif len(y_samples) < len(x_samples):
                x_samples = x_samples[: len(y_samples), :]

        potential_vals = self(x_samples, y_samples)  # (batch_size/2, batch_size/2)
        energy = 0.5 * jnp.mean(potential_vals)

        return energy, x_samples
