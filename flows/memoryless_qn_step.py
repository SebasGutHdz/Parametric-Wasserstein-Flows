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

from geometry.G_matrix import G_matrix
from geometry.lin_alg_solvers import minres
from flows.gradient_flow_step import gradient_flow_step

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel


# comments correspond to notation from Memoryless Quasi-Newton Method ... Narushima 2023
def memoryless_qn_step(
    parametric_model: ParametricModel,
    current_params: PyTree,
    delta_theta_prev: Optional[PyTree],  # previous update direction s_{k-1}
    grad_prev: Optional[PyTree],  # previous gradient g_k
    G_mat: G_matrix,
    potential: Potential,
    z_samples: Array,
    solver: str = "cg",  # \
    solver_tol: float = 1e-6,  #  > linear solver parameters for the computation of Riemannian gradient
    solver_maxiter: int = 50,  # /
    solver_regularization: float = 1e-6,
    step_size: float = 0.01,  # constant stepsize alpha_k
    ensure_descent: bool = False,  # we use this simple heuristic to compensate the lack of line search; if the new update direction is not a descent direction, go in the direction of the gradient instead
    hessian_update_strategy: Literal[
        "preconvex", "BFGS"
    ] = "BFGS",  # choice of \varphi_{k-1}
    regularization_strategy: Literal[
        "Li-Fukushima", "Powell"
    ] = "Li-Fukushima",  # choice of z_{k-1}
    spectral_scaling: bool = True,
) -> Tuple[PyTree, List[PyTree], List[PyTree], Dict]:
    """
    Memoryless qN method

    Args:
        parametric_model: defines the pushforward map
        current_params: current parameters
        delta_x_prev: previous update vector
        grad_prev: previous grad
        G_mat: Riemannian metric tensor
        potential: functional we optimize
        z_samples: samples to compute test metrics
        solver: solver for the estimation of Riemannian gradient
        solver_tol: linear solver tolerance
        solver_maxiter: iteration limit for the solver
        step_size: step size parameter
        ensure_descent: heuristic to compensate for the lack of line search; if the new update direction is not a descent direction, go in the direction of the gradient instead
        hessian_update_strategy: choice of \\varphi_{k-1}
        regularization_strategy: choice of z_{k-1}

    Returns:
        new_params: New parameters
        delta_theta: Update vector
        grad: Residual
    """
    if current_params is None:
        graphdef, current_params = nnx.split(parametric_model)

    # if this is the first step, initialize history
    if delta_theta_prev is None:
        g_cur = compute_riemannian_grad(
            parametric_model,
            current_params,
            G_mat,
            potential,
            z_samples,
            solver,
            solver_tol,
            solver_maxiter,
            solver_regularization,
        )
        # Compute difference for theta and grads
        delta_theta = jax.tree.map(lambda _g: -step_size * _g, g_cur)
        new_params = jax.tree.map(
            lambda _theta, _dtheta: _theta + _dtheta, current_params, delta_theta
        )
        return new_params, delta_theta, g_cur

    g_cur = compute_riemannian_grad(
        parametric_model,
        current_params,
        G_mat,
        potential,
        z_samples,
        solver,
        solver_tol,
        solver_maxiter,
        solver_regularization,
    )

    # "Vector transport"
    s_cur = delta_theta_prev

    # In case we use the second strategy for beta from the paper
    beta = 1.0
    y_cur = jax.tree.map(lambda x, y: beta * x - y, g_cur, grad_prev)

    s_norm_sq = G_mat.inner_product(s_cur, s_cur, z_samples, current_params)
    sy_product = G_mat.inner_product(s_cur, y_cur, z_samples, current_params)
    # y_norm_sq = G_mat.inner_product(y_cur, y_cur, z_samples, current_params)
    match regularization_strategy:
        case "Li-Fukushima":
            nu_hat = 1e-6
            if sy_product >= nu_hat * s_norm_sq:
                nu_cur = 0.0
            else:
                nu_cur = jnp.maximum(0.0, -sy_product / s_norm_sq) + nu_hat
            z_cur = jax.tree.map(lambda _y, _s: _y + nu_cur * _s, y_cur, s_cur)
        case "Powell":
            nu_hat = 0.1
            if sy_product >= nu_hat * s_norm_sq:
                nu_cur = 1.0
            else:
                nu_cur = (1.0 - nu_hat) * s_norm_sq / (s_norm_sq - sy_product)
            z_cur = jax.tree.map(
                lambda _y, _s: nu_cur * _y + (1.0 - nu_cur) * _s, y_cur, s_cur
            )
        case _:
            raise ValueError(f"Unsupported argument {regularization_strategy=}")

    z_norm_sq = G_mat.inner_product(z_cur, z_cur, z_samples, current_params)
    sz_product = G_mat.inner_product(s_cur, z_cur, z_samples, current_params)

    # Compute phi -- the coefficient that interpolates between the differnet qN inverse hessian update formulae
    match hessian_update_strategy:
        case "BFGS":
            phi = 1.0
        case "preconvex":
            mu = s_norm_sq * z_norm_sq / sz_product**2
            theta_star = jnp.maximum(-1e5, 1.0 / (1.0 - mu))
            phi = (0.1 * theta_star - 1.0) / (0.1 * theta_star * (1.0 - mu) - 1.0)
        case _:
            raise ValueError(f"Unsupported argument {hessian_update_strategy=}")

    # compute scaling coefs
    if spectral_scaling:
        gamma = jnp.maximum(1.0, sz_product / z_norm_sq)
        tau = jnp.minimum(1.0, z_norm_sq / sz_product)
    else:
        gamma = 1.0
        tau = 1.0

    # compute the third update direction w
    w_cur = jax.tree.map(lambda _s, _z: _s / sz_product - _z / z_norm_sq, s_cur, z_cur)

    # <<apply>> the H_k operator: first compute products with current residual
    gz_product = G_mat.inner_product(g_cur, z_cur, z_samples, current_params)
    gs_product = G_mat.inner_product(g_cur, s_cur, z_samples, current_params)
    gw_product = G_mat.inner_product(g_cur, w_cur, z_samples, current_params)
    # compute coefs in front of update directions z, s, w
    z_coef = -gamma * gz_product / z_norm_sq
    s_coef = gs_product / sz_product / tau
    w_coef = gw_product * z_norm_sq * gamma * phi
    # compute the update direction
    delta_theta = jax.tree.map(
        lambda _g, _z, _s, _w: -step_size
        * (gamma * _g + z_coef * _z + s_coef * _s + w_coef * _w),
        g_cur,
        z_cur,
        s_cur,
        w_cur,
    )

    if ensure_descent and G_mat.inner_product(delta_theta, g_cur, z_samples, current_params) >= 0.0:
        delta_theta = jax.tree.map(lambda x: -step_size * x, g_cur)

    # update the parameters and the parametric model
    theta_new = jax.tree.map(lambda p, d: p + d, current_params, delta_theta)
    parametric_model = nnx.update(parametric_model, theta_new)  # does this do anything?

    return (theta_new, delta_theta, g_cur)


def compute_riemannian_grad(
    parametric_model: nnx.Module,
    params: PyTree,
    G_mat: G_matrix,
    potential: Potential,
    z_samples: Array,
    solver: str,
    solver_tol: float,
    solver_maxiter: int,
    regularization: float,
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
    return eta
