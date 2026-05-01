import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, List, Dict, Optional, Literal
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
from jax import Device

from geometry.G_matrix import G_matrix
from geometry.lin_alg_solvers import minres, dot_tree
from flows.gradient_flow_step import gradient_flow_step
from flows.work_accounting import WorkCounter

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel


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
    regularization_factor_gamma: float = 1e-6,
    regularization_method_gamma: Literal["l2", "adaptive"] = "l2",
    ensure_descent: bool = False,
    work_counter: WorkCounter | None = None,
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
    if current_params is None:
        graphdef, current_params = nnx.split(parametric_model)

    if param_history is None:
        # No history, perform a standard gradient flow step.
        # gradient_flow_step returns eta = G^{-1} grad F(theta_0) in step_info,
        # so r_0 = -h * eta is free — avoids a redundant compute_fixed_point_residual call.
        theta_1, step_info_0 = gradient_flow_step(
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
            work_counter=work_counter,
        )
        r_0 = jax.tree.map(lambda x: -step_size * x, step_info_0["eta"])
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
            solver_maxiter=solver_maxiter,
            work_counter=work_counter,
        )
        # Compute difference for theta and residuals
        delta_theta_0 = jax.tree.map(lambda a, b: b - a, current_params, theta_1)
        delta_r_0 = jax.tree.map(lambda a, b: b - a, r_0, r_1)
        return ([theta_1, current_params], [r_1, r_0], [delta_theta_0], [delta_r_0])

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
        current_params,
        regularization_factor=regularization_factor_gamma,
        regularization_method=regularization_method_gamma,
        parameter_differences=param_diff,
        work_counter=work_counter,
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

    if ensure_descent:
        if work_counter is not None:
            work_counter.record_direct_mvp(z_samples)
        descent_ip = G_mat.inner_product(
            delta_theta_n, r_n, z_samples, current_params
        )
    else:
        descent_ip = 0.0
    if ensure_descent and descent_ip < -0.0:
        delta_theta_n = jax.tree.map(lambda x: x, r_n)

    theta_new = jax.tree.map(lambda p, d: p + d, current_params, delta_theta_n)

    # Compute new residual
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
        solver_maxiter=solver_maxiter,
        work_counter=work_counter,
    )
    # Compute new residual difference
    delta_r_new = jax.tree.map(lambda a, b: b - a, r_n, r_new)
    # Update histores
    new_params_history = ([theta_new] + param_history)[: memory_size + 1]
    new_residual_history = ([r_new] + residual_history)[: memory_size + 1]
    new_param_diff = ([delta_theta_n] + param_diff)[:memory_size]
    new_residual_diff = ([delta_r_new] + residual_diff)[:memory_size]

    parametric_model = nnx.update(parametric_model, theta_new)

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
    solver_maxiter: int = 50,
    work_counter: WorkCounter | None = None,
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
    if work_counter is not None:
        work_counter.record_grad(z_samples)
    energy_grad, energy, energy_breakdown = potential.compute_energy_gradient(
        parametric_model, z_samples, params
    )
    # Solve linear system
    if work_counter is not None:
        work_counter.record_solve(z_samples, solver_maxiter)
    eta, solver_info = G_mat.solve_system(
        z_samples,
        energy_grad,
        params=params,
        tol=solver_tol,
        maxiter=solver_maxiter,
        method=solver,
        regularization=regularization,
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
    regularization_factor: float = 1e-3,
    regularization_method: Literal["l2", "adaptive"] = "l2",
    parameter_differences: List[PyTree] = None,
    work_counter: WorkCounter | None = None,
) -> Tuple[List[float], Dict]:
    """
    Solve Anderson mixing optimization using G-matrix norm.

    Args:
        current_residual: Current fixed-point residual r_k
        residual_differences: List of previous residual differences (r_{i+1} - r_i)
        G_mat: G_matrix object to compute inner products
        z_samples: Reference samples (batch_size, d)
        params: Current model parameters (passed to G_mat.mvp)
        regularization_factor: L2 regularization strength
        regularization_method: "l2" or "adaptive"
        parameter_differences: Required when regularization_method == "adaptive"

    Returns:
        gamma: Coefficients for Anderson mixing
    """

    assert (
        regularization_method == "adaptive" and parameter_differences is not None
    ) or regularization_method == "l2"

    m = len(residual_differences)

    if m == 0:
        return []

    # Precompute G @ residual_differences[j] once per j — reduces O(m²) mvp
    # calls to O(m) calls. Inner products then become cheap tree dot products.
    if work_counter is not None:
        work_counter.record_direct_mvp(z_samples, m)
    G_res_vecs = [G_mat.mvp(z_samples, r, params) for r in residual_differences]

    # Build least-squares system: A gamma = b
    A = jnp.zeros((m, m))
    M_reg = jnp.eye(m)
    b = jnp.zeros((m,))

    for i in range(m):
        for j in range(i, m):
            # A_ij = <Delta r_i, Delta r_j>_G  (symmetric, fill upper triangle)
            A = A.at[i, j].set(dot_tree(residual_differences[i], G_res_vecs[j]))
        # b_i = <r_n, Delta r_i>_G
        b = b.at[i].set(dot_tree(current_residual, G_res_vecs[i]))

    if regularization_method == "adaptive":
        # Precompute G @ parameter_differences[j] — m more mvp calls instead of O(m²)
        if work_counter is not None:
            work_counter.record_direct_mvp(z_samples, m + 1)
        G_param_vecs = [G_mat.mvp(z_samples, d, params) for d in parameter_differences]
        G_cur = G_mat.mvp(z_samples, current_residual, params)
        r_cur_norm_sq = dot_tree(current_residual, G_cur)
        for i in range(m):
            for j in range(i, m):
                M_reg = M_reg.at[i, j].set(
                    dot_tree(parameter_differences[i], G_param_vecs[j])
                )
        delta_x_norm_sq = M_reg[0, 0]
        regularization_factor *= r_cur_norm_sq / (delta_x_norm_sq + 1e-8)

    # Add l2 regularization and symmetrize
    A = A + M_reg * regularization_factor
    A = A + A.T - jnp.diag(A.diagonal())
    # For small m < 15, direct solve is faster than iterative
    gamma = jnp.linalg.solve(A, b)

    return gamma.tolist()
