"""'
This module implements a single Parametric Hamiltonian flow step. For more details, see
PARAMETERIZED WASSERSTEIN HAMILTONIAN FLOW SIAM J. NUMER. ANAL. Vol 63, No. 1 pp 360--395
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Union
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device

from geometry.G_matrix import G_matrix
from parametric_model.parametric_model import ParametricModel
from functionals.functional import Potential

from tqdm import tqdm


def hamiltonian_flow_step(
    parametric_model: ParametricModel,
    p_n: PyTree,
    z_samples: Array,
    G_mat: G_matrix,
    potential: Potential,
    step_size: float = 0.01,
    solver: str = "minres",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    gamma: float = 1e-2,
    n_iters: int = 3,
    only_return_params: bool = False,
) -> Tuple[Union[nnx.Module, PyTree], dict]:
    """
    Generic hamiltonian flow step that works with any Potential
    This code solve a forward step of the Hamiltonian ODE system using a symplectic Euler method
    Given the (theta_n,p_n) it computes (theta_{n+1},p_{n+1}) as follows:
    (theta_{n+1}-theta_n)/h = G(theta_{n+1})^{-1} p_n
    (p_{n+1}-p_n)/h = (1/2)[ (G(theta_{n+1})^{-1} p_n)^T \nabla_theta G(theta_{n+1}) G(theta_{n+1})^{-1} p_n - \nabla_theta F(theta_{n+1}) ]

    Following equations 4.1a and 4.1b in the reference paper

    Args:
        parametric_model: Current ParametricModel instance
        p_n: Current momentum PyTree
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance
        step_size: Hamiltonian flow step size h
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        regularization: Regularization parameter used in regularized cg
        gamma: Step size gradient descent step for the fixed point interation
        n_iters: Number of fixed point iterations to perform
    Returns:
        updated_parametric_model: ParametricModel with updated parameters
        step_info: Dictionary with step diagnostics
    """

    # Get architecture and current parameters
    graph_def, theta_n = nnx.split(parametric_model)
    # Update theta using symplectic Euler step
    alpha, xi = fixed_point_solver(
        z_samples=z_samples,
        theta_n=theta_n,
        p_n=p_n,
        G_mat=G_mat,
        h=step_size,
        gamma=gamma,
        n_iters=n_iters,
        solver=solver,
        maxiter=solver_maxiter,
        tol=solver_tol,
        regularization=regularization,
    )
    xi_detached = jax.tree.map(lambda x: jax.lax.stop_gradient(x), xi)
    # Compute the terms for the update of p:
    grad_g_mat = G_mat.metric_derivative_quadratic_form(
        z_samples=z_samples, eta=xi_detached, params=alpha
    )
    # 2. Compute energy gradient using the potential
    energy_grad, energy, energy_breakdown = potential.compute_energy_gradient(
        parametric_model=parametric_model, z_samples=z_samples, params=alpha
    )
    # 3 Update p using Euler scheme
    p_new = jax.tree.map(
        lambda p, grad_g, grad_f: p + step_size * (0.5 * grad_g - grad_f),
        p_n,
        grad_g_mat,
        energy_grad,
    )

    updated_parametric_model = nnx.merge(graph_def, alpha)

    step_info = {
        "energy": energy,
        "energy": energy,
        "internal_energy": energy_breakdown["internal_energy"],
        "linear_energy": energy_breakdown["linear_energy"],
        "interaction_energy": energy_breakdown["interaction_energy"],
        "step_size": step_size,
    }

    return updated_parametric_model, p_new, step_info


def fixed_point_solver(
    z_samples: Array,
    theta_n: PyTree,
    p_n: PyTree,
    G_mat: G_matrix,
    h: float,
    gamma: float = 1e-2,
    n_iters: int = 3,
    solver: str = "cg",
    maxiter: int = 10,
    tol: float = 1e-5,
    regularization: float = 1e-6,
) -> list[PyTree, PyTree]:
    """
    This function solves the fixed point iteration problem arising from the implicit update of theta in the symplectic Euler scheme
    See 6-9 of Alg 4.1 in the reference paper.
    Args:
        z_samples: (Bs,d) Samples from reference density
        theta_n: PyTree with current parameters
        p_n: PyTree with current momentum
        G_mat: G-matrix object for linear system solving
        h: Step size of the Hamiltonian flow
        gamma: Step size gradient descent step for the fixed point interation
        n_iters: Number of fixed point iterations to perform
        solver: Linear solver to use ("cg","gmres","minres")
        maxiter: Maximum number of iterations for linear solver
        tol: Tolerance for linear solver
        regularization: Regularization parameter used in regularized cg

    Returns:
        alpha: PyTree
        xi: PyTree
    """
    # Initialize alpha and xi, alpha = theta, G xi = p
    alpha = jax.tree.map(lambda x: x, theta_n)
    xi, _ = G_mat.solve_system(
        z_samples=z_samples,
        b=p_n,
        params=alpha,
        maxiter=maxiter,
        method=solver,
        tol=tol,
        regularization=regularization,
    )
    # run n_iters of fixed point iteration
    for _ in range(n_iters):
        # update alpha
        alpha = jax.tree.map(lambda a, x: a + h * x, theta_n, xi)
        # update xi
        mvp_result = G_mat.mvp(z_samples=z_samples, eta=xi, params=alpha)
        xi = jax.tree.map(
            lambda prev, mpv, p: prev - gamma * (mpv - p), xi, mvp_result, p_n
        )
    return alpha, xi


def compute_hamiltonian(
    theta: PyTree, p: PyTree, z_samples: Array, G_mat: G_matrix, potential: Potential
) -> float:
    """Compute H = (1/2)p^T G^(-1) p + F(θ)"""
    # Kinetic energy
    h, _ = G_mat.solve_system(z_samples, p, params=theta)
    kinetic = 0.5 * sum(
        jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(a * b), p, h))
    )
    graph_def, _ = nnx.split(G_mat.mapping)
    # Potential energy
    temp_parametric_model = nnx.merge(graph_def, theta)
    _, potential_energy, _ = potential.evaluate_energy(
        temp_parametric_model, z_samples, theta
    )

    return kinetic + potential_energy
