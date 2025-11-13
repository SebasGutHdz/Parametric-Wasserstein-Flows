"""
Adjoint ODE solver for parametric boundary value Hamiltonian problems.

"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from jaxtyping import Array, PyTree
from flax import nnx


from core.types import SampleArray
from architectures.utils_node import eval_model
from ODE_solvers.solvers import ODESolver, string_2_solver
from parametric_model.parametric_model import ParametricModel
from functionals.functional import Potential


def adjoint_solver(
    parametric_model: ParametricModel,
    z0: SampleArray,
    t_span: Tuple[float, float],
    potential_fn: Potential,
    solver: ODESolver,
    time_dependent: bool = False,
    parms: Optional[PyTree] = None,
    xt: Optional[SampleArray] = None,
    dt0: float = 0.1,
) -> Tuple[float, PyTree]:
    """
    Solve ODE and compute gradients using adjoint method.

    Args:
        parametric_model: ParametricMap it needs to be of vector field type
        z0: Initial condition (batch_size, dim)
        t_span: (t0, t1) integration interval
        loss_fn: L(w_final) -> scalar
        solver: ODESolver instance
        time_dependent: Whether f depends on t
        params: Parameter PyTree (if None, uses current)
        dt0: Time step size

    Returns:
        loss_value: L(w(T))
        grad_params: ∇_θ L
    """
    if parametric_model.parametric_map != "node":
        raise ValueError("parametric_model must be of type 'node' for adjoint_solver.")
    # Get parameters
    if parms is None:
        graphdef, params = nnx.split(parametric_model)
    else:
        graphdef, _ = nnx.split(parametric_model)

    # If the trajectory is not provided, solve the forward ODE
    if xt is None:
        xt, t_list = parametric_model(samples=z0, params=parms, history=True)

    x_final = xt[:, -1, :]  # Final state

    # Evaluate linear potential and internal potential if exists

    def loss_fn(x: SampleArray) -> float:
        energy = 0.0
        if potential_fn.linear is not None:
            energy += potential_fn.linear.evaluate_energy(x)
        if potential_fn.interaction is not None:
            energy += potential_fn.interaction.evaluate_energy(x)
        return energy

    loss_value, grad_ = jax.value_and_grad(loss_fn)(x_final)

    # Here we call the backward_adjoint
    return


# def backward_adjoint(
#         parametric_model: ParametricModel,
#         z0
