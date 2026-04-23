import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Union, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device

from geometry.G_matrix import G_matrix
from functionals.functional import Potential
from core.utility import _params_scalar_product

from tqdm import tqdm

from operator import add


def move_to_device(pytree: Any, device) -> Any:
    """Recursively moves all JAX arrays in a PyTree to the specified device."""
    return jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x, pytree
    )


def gradient_flow_step(
    parametric_model: nnx.Module,
    z_samples: Array,
    G_mat: G_matrix,
    potential: Potential,
    step_size: float = 0.01,
    solver: str = "minres",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    solver_x0: Optional[PyTree] = None,
    only_return_params: bool = False,
    graphdef: Optional[nnx.GraphDef] = None,
    current_params: Optional[PyTree] = None,
    outer_iter: Optional[int] = None,
    phase: str = "step_update",
    run_id: Optional[str] = None,
    method_name: Optional[str] = None,
) -> Tuple[Union[nnx.Module, PyTree], dict]:
    """
    Generic gradient flow step that works with any Potential

    Args:
        parametric_model: Current ParametricModel instance
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance
        step_size: Gradient flow step size h 
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        regularization: Regularization parameter used in regularized cg
    Returns:
        updated_parametric_model: ParametricModel with updated parameters
        step_info: Dictionary with step diagnostics
    """

    # Get current parameters (only split if not provided)
    if current_params is None or graphdef is None:
        graphdef, current_params = nnx.split(parametric_model)

    # Compute energy gradient using the potential
    energy_grad, energy, energy_breakdown = potential.compute_energy_gradient(
        parametric_model, z_samples, current_params
    )

    # Solve linear system
    # z_samples_g_mat = z_samples[::2]  # Use a subset of samples for G-matrix to save computation

    if hasattr(G_mat, "set_linear_solve_context"):
        G_mat.set_linear_solve_context(
            run_id=run_id,
            method_name=method_name,
            outer_iter=outer_iter,
            phase=phase,
        )
    eta, solver_info = G_mat.solve_system(
        z_samples,
        energy_grad,
        params=current_params,
        tol=solver_tol,
        maxiter=solver_maxiter,
        method=solver,
        regularization=regularization,
        x0=solver_x0,
    )
    # ODE solve.
    # TODO: Higher order derivative solvers.
    updated_params = jax.tree.map(lambda p, e: p - step_size * e, current_params, eta)

    # Compute diagnostics
    grad_norm = jnp.sqrt(
        sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), energy_grad)))
    )
    riemannian_grad_norm_sq = _params_scalar_product(energy_grad, eta)
    riemann_grad_norm = jnp.sqrt(jnp.maximum(riemannian_grad_norm_sq, 0.0))
    eta_norm = jnp.sqrt(
        sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), eta)))
    )
    param_norm = jnp.sqrt(
        sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), updated_params)))
    )

    step_info = {
        "eta": eta,
        "gradient_norm": grad_norm,
        "riemann_gradient_norm": riemann_grad_norm,
        "eta_norm": eta_norm,
        "param_norm": param_norm,
        "energy": energy,
        "internal_energy": energy_breakdown["internal_energy"],
        "linear_energy": energy_breakdown["linear_energy"],
        "interaction_energy": energy_breakdown["interaction_energy"],
        "step_size": step_size,
        "big_solve_iterations": solver_info.get("iterations"),
        "big_solve_iterations_estimated": solver_info.get(
            "iterations_estimated", False
        ),
        "big_solve_converged": solver_info.get("converged"),
        "big_solve_elapsed_sec": solver_info.get("elapsed_sec", 0.0),
    }

    if only_return_params:
        return updated_params, step_info

    # Create updated parametric model
    updated_parametric_model = nnx.merge(graphdef, updated_params)

    return updated_parametric_model, step_info
