import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, List, Dict, Optional
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
from jax import Device

from geometry.G_matrix import G_matrix
from geometry.lin_alg_solvers import minres
from flows.gradient_flow_step import gradient_flow_step

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel

from collections import deque


def _update_lifo_history(history, item, max_size: int):
    """
    Update a bounded LIFO history where index 0 is always the newest entry.
    """
    if isinstance(history, deque):
        updated = deque(history, maxlen=max_size)
    elif history is None:
        updated = deque(maxlen=max_size)
    else:
        updated = deque(history, maxlen=max_size)
    updated.appendleft(item)
    return updated


def anderson_step(
    parametric_model: ParametricModel,
    current_params: PyTree,
    param_history: Optional[List[PyTree]],  # at most m entries
    residual_history: Optional[List[PyTree]],  # at most m entries
    param_diff: Optional[List[PyTree]],  # at most m entries
    residual_diff: Optional[List[PyTree]],  # at most m entries
    G_mat: G_matrix,
    potential: Potential,
    z_samples: Array,
    step_size: float = 0.01,
    memory_size: int = 5,
    relaxation: float = 1.0,
    anderson_tol: float = 1e-6,
    solver: str = "cg",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    l2_reg_gamma: float = 1e-6,
    l2_inner_product: bool = False,
    graphdef: Optional[nnx.GraphDef] = None,
    outer_iter: Optional[int] = None,
    run_id: Optional[str] = None,
    method_name: Optional[str] = None,
) -> Tuple[PyTree, List[PyTree], List[PyTree], Dict]:
    """
    Anderson acceleration step for fixed-point iteration.
    Args:
        parametric_model: ParametricModel instance
        current_params: Current parameters of the model
        param_history: List of previous parameters (at most memory_size entries)
        residual_history: List of previous residuals (at most memory_size entries)
        param_diff: List of parameter differences (at most memory_size entries)
        residual_diff: List of residual differences (at most memory_size entries)
        G_mat: G_matrix object to compute inner products
        potential: Potential object to compute energy and gradient
        z_samples: Reference samples (batch_size, d)
        device: JAX device to perform computations on
        step_size: Step size for the fixed-point iteration
        memory_size: Number of previous iterations to use for Anderson acceleration
        mixing_parameter: Mixing parameter for Anderson acceleration
        anderson_tol: Tolerance for Anderson acceleration convergence
        solver: Linear solver to use ('minres' or 'cg')
        solver_tol: Tolerance for the linear solver
        regularization: Regularization parameter for the linear system
    Returns:
        new_params: Updated parameters after Anderson acceleration
        new_param_history: Updated parameter history
        new_residual_history: Updated residual history
        info: Dictionary with information about the step (e.g., energy, gradient norm)
    """
    if current_params is None or graphdef is None:
        graphdef, current_params = nnx.split(parametric_model)

    if param_history is None:
        # No history, perform a standard gradient flow step
        theta_1, _ = gradient_flow_step(
            parametric_model=parametric_model,
            z_samples=z_samples,
            G_mat=G_mat,
            potential=potential,
            step_size=step_size,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            regularization=regularization,
            only_return_params=True,
            graphdef=graphdef,
            current_params=current_params,
            outer_iter=outer_iter,
            phase="bootstrap_theta1",
            run_id=run_id,
            method_name=method_name,
        )
        r_0 = compute_fixed_point_residual(
            parametric_model,
            current_params,
            G_mat,
            potential,
            z_samples,
            step_size,
            solver,
            solver_tol,
            regularization,
            outer_iter=outer_iter,
            phase="bootstrap_r0",
            run_id=run_id,
            method_name=method_name,
        )
        r_1 = compute_fixed_point_residual(
            parametric_model,
            theta_1,
            G_mat,
            potential,
            z_samples,
            step_size,
            solver,
            solver_tol,
            regularization,
            outer_iter=outer_iter,
            phase="bootstrap_r1",
            run_id=run_id,
            method_name=method_name,
        )

        
        # Compute difference for theta and residuals
        delta_theta_0 = jax.tree.map(lambda a, b: b - a, current_params, theta_1)
        delta_r_0 = jax.tree.map(lambda a, b: b - a, r_0, r_1)
        return (
            deque([theta_1, current_params], maxlen=memory_size + 1),
            deque([r_1, r_0], maxlen=memory_size + 1),
            deque([delta_theta_0], maxlen=memory_size),
            deque([delta_r_0], maxlen=memory_size),
        )

    # For non-empty history, perform Anderson acceleration
    # Get current residual
    r_n = residual_history[0]
    # Determine hisotry length
    m_k = len(residual_diff)

    # Solve Anderson subproblem to get mixing coefficients
    gamma = compute_anderson_gamma(
        r_n,
        residual_diff,
        G_mat,
        z_samples,
        params=current_params,
        l2_regularization=l2_reg_gamma,
        l2_inner_product=l2_inner_product,
    )

    # Compute mixed residuals \bar{r}_n
    mixed_residual = r_n
    for i, gamma_i in enumerate(gamma):
        mixed_residual = jax.tree.map(
            lambda a, b: a - gamma_i * b, mixed_residual, residual_diff[i]
        )
    # Compute parameter update delta theta_n
    delta_theta_n = jax.tree.map(lambda x: relaxation * x, mixed_residual)
    for i, gamma_i in enumerate(gamma):
        delta_theta_n = jax.tree.map(
            lambda step, dx: step - gamma_i * dx, delta_theta_n, param_diff[i]
        )
    # Update parameters
    theta_candidate = jax.tree.map(lambda p, d: p + d, current_params, delta_theta_n)

    # Safeguard: if Anderson candidate does not decrease energy on the same sample
    # block, fall back to one plain gradient-flow step.
    # energy_old, _, _, _, _ = potential.evaluate_energy(
    #     parametric_model=parametric_model,
    #     z_samples=z_samples,
    #     params=current_params,
    # )
    # energy_candidate, _, _, _, _ = potential.evaluate_energy(
    #     parametric_model=parametric_model,
    #     z_samples=z_samples,
    #     params=theta_candidate,
    # )
    # fallback_to_gf = (not bool(jnp.isfinite(energy_candidate))) or (
    #     float(energy_candidate) > float(energy_old)
    # )

    # if fallback_to_gf:
    #     theta_gf, _ = gradient_flow_step(
    #         parametric_model=parametric_model,
    #         z_samples=z_samples,
    #         G_mat=G_mat,
    #         potential=potential,
    #         step_size=step_size,
    #         solver=solver,
    #         solver_tol=solver_tol,
    #         solver_maxiter=solver_maxiter,
    #         regularization=regularization,
    #         only_return_params=True,
    #         graphdef=graphdef,
    #         current_params=current_params,
    #         outer_iter=outer_iter,
    #         phase="safeguard_fallback_theta",
    #         run_id=run_id,
    #         method_name=method_name,
    #     )
    #     energy_gf, _, _, _, _ = potential.evaluate_energy(
    #         parametric_model=parametric_model,
    #         z_samples=z_samples,
    #         params=theta_gf,
    #     )
    #     # If fallback is invalid numerically, keep current iterate and restart memory.
    #     if bool(jnp.isfinite(energy_gf)):
    #         theta_new = theta_gf
    #     else:
    #         theta_new = current_params
    # else:
    theta_new = theta_candidate

    # Compute new residual at the Anderson-mixed iterate
    r_new = compute_fixed_point_residual(
        parametric_model,
        theta_new,
        G_mat,
        potential,
        z_samples,
        step_size,
        solver,
        solver_tol,
        regularization,
        outer_iter=outer_iter,
        phase="residual_eval",
        run_id=run_id,
        method_name=method_name,
        solver_x0=None,
    )
    # Compute new residual difference
    delta_r_new = jax.tree.map(lambda a, b: b - a, r_n, r_new)
    # Update histories in LIFO order (newest item at index 0)
    new_params_history = _update_lifo_history(param_history, theta_new, memory_size + 1)
    new_residual_history = _update_lifo_history(
        residual_history, r_new, memory_size + 1
    )
    new_param_diff = _update_lifo_history(param_diff, delta_theta_n, memory_size)
    new_residual_diff = _update_lifo_history(
        residual_diff, delta_r_new, memory_size
    )
    # if fallback_to_gf:
    #     # Failed AA direction: restart acceleration memory to avoid reusing
    #     # unstable/stale secant information.
    #     new_param_diff = deque(maxlen=memory_size)
    #     new_residual_diff = deque(maxlen=memory_size)

    # Note: Removed nnx.update - caller handles model state via params

    return (
        new_params_history,
        new_residual_history,
        new_param_diff,
        new_residual_diff,
    )


def compute_fixed_point_residual(
    parametric_model: nnx.Module,
    params: PyTree,
    G_mat: G_matrix,
    potential: Potential,
    z_samples: Array,
    step_size: float,
    solver: str,
    solver_tol: float,
    regularization: float,
    outer_iter: Optional[int] = None,
    phase: str = "residual_eval",
    run_id: Optional[str] = None,
    method_name: Optional[str] = None,
    solver_x0: Optional[PyTree] = None,
) -> PyTree:
    """
    Compute fixed point residual r = -h * G^{-1} grad F(p) for parameters p
    Args:
        parametric_model: Neural ODE model
        params: Current parameters of the model
        G_mat: G_matrix object to compute inner products
        potential: Potential object to compute energy and gradient
        z_samples: Reference samples (batch_size, d)
        step_size: Step size for the fixed-point iteration
        solver: Linear solver to use ('minres' or 'cg')
        solver_tol: Tolerance for the linear solver
        regularization: Regularization parameter for the linear system
    Returns:
        residual: Fixed point residual as a PyTree
    """

    # Compute energy gradient using the potential
    energy_grad, energy, energy_breakdown = potential.compute_energy_gradient(
        parametric_model, z_samples, params
    )
    # Solve linear system
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
        params=params,
        tol=solver_tol,
        maxiter=50,
        method=solver,
        regularization=regularization,
        x0=solver_x0,
    )
    # Fixed point residual
    residual = jax.tree.map(lambda x: -step_size * x, eta)
    return residual


def compute_anderson_gamma(
    current_residual: PyTree,
    residual_differences: List[PyTree],
    G_mat: G_matrix,
    z_samples: Array,
    params: PyTree,
    tol: float = 1e-6,
    l2_regularization: float = 1e-6,
    l2_inner_product: bool = False,
) -> Tuple[List[float], Dict]:
    """
    Solve Anderson mixing optimization using G-matrix norm:

    Args:
        current_residual: Current fixed-point residual r_k
        residual_differences: List of previous residual differences (r_{i+1} - r_i)
        G_mat: G_matrix object to compute inner products
        z_samples: Reference samples (batch_size, d)
        l2_regularization: needed to bound the mixing coefficients

    Returns:
        gamma: Coefficients for Anderson mixing
    """
    m = len(residual_differences)

    if m == 0:
        return []

    # Build least-squares system: A γ = b
    A = jnp.zeros((m, m))
    b = jnp.zeros((m,))

    def _tree_vdot(x: PyTree, y: PyTree) -> Array:
        leaf_dots = jax.tree.leaves(jax.tree.map(lambda a, c: jnp.vdot(a, c), x, y))
        return jnp.sum(jnp.stack(leaf_dots))

    if l2_inner_product:
        # L2 path: no G-mvp, just cheap vdots
        for i in range(m):
            for j in range(i, m):
                A = A.at[i, j].set(_tree_vdot(residual_differences[i], residual_differences[j]))
            b = b.at[i].set(_tree_vdot(current_residual, residual_differences[i]))
    else:
        # G-metric path: precompute G·Δr_i once per history entry — m mvp calls total
        # (was m(m+3)/2 calls when using inner_product separately for each A_ij and b_i)
        # A_ij = ⟨Δr_i, Δr_j⟩_G = vdot(Δr_i, G·Δr_j)
        # b_i  = ⟨r_n,  Δr_i⟩_G = vdot(r_n,  G·Δr_i)
        G_diffs = [
            G_mat.mvp(z_samples, residual_differences[i], params)
            for i in range(m)
        ]
        for i in range(m):
            for j in range(i, m):
                A = A.at[i, j].set(_tree_vdot(residual_differences[i], G_diffs[j]))
            b = b.at[i].set(_tree_vdot(current_residual, G_diffs[i]))

    A = 0.5 * (A + A.T)
    A = A + jnp.eye(A.shape[0]) * l2_regularization
    gamma = jnp.linalg.solve(A, b)

    return gamma.tolist()
