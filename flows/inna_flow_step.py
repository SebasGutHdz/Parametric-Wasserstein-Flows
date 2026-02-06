import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Union, Optional
import jax
import jax.numpy as jnp

from geometry.G_matrix import G_matrix
from functionals.functional import Potential
from core.utility import _params_scalar_product


def inna_flow_step(
    parametric_model: nnx.Module,
    psi: PyTree,
    z_samples: Array,
    G_mat: G_matrix,
    potential: Potential,
    gamma: float = 0.01,
    a: float = 0.1,
    b: float = 0.1,
    beta: float = 1.0,
    solver: str = "cg",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    only_return_params: bool = False,
    graphdef: Optional[nnx.GraphDef] = None,
    current_params: Optional[PyTree] = None,
) -> Tuple[Union[nnx.Module, PyTree], PyTree, dict]:
    """
    INNA (Inertial Neural Network Algorithm) flow step using Riemannian gradient.

    Implements the dynamical system:
        θ_{k+1} = θ_k + γ[-a·θ_k - b·ψ_k - β·η_k]
        ψ_{k+1} = ψ_k + γ[-a·θ_k - b·ψ_k]

    where η_k = G(θ_k)^{-1} ∇F(θ_k) is the Riemannian gradient.

    Args:
        parametric_model: Current ParametricModel instance
        psi: Auxiliary momentum-like variable (same PyTree structure as params)
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance
        gamma: Step size γ
        a: Coupling coefficient for θ terms
        b: Coupling coefficient for ψ terms
        beta: Gradient coefficient β
        solver: Linear solver method ("cg" or "minres")
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        regularization: Regularization parameter for solver

    Returns:
        updated_parametric_model: ParametricModel with updated parameters (or params if only_return_params)
        updated_psi: Updated auxiliary variable
        step_info: Dictionary with step diagnostics
    """

    # Get current parameters (only split if not provided)
    if current_params is None or graphdef is None:
        graphdef, current_params = nnx.split(parametric_model)

    # Compute energy gradient using the potential
    energy_grad, energy, energy_breakdown = potential.compute_energy_gradient(
        parametric_model, z_samples, current_params
    )

    # Solve linear system: G(θ) η = ∇F(θ) to get Riemannian gradient
    eta, solver_info = G_mat.solve_system(
        z_samples,
        energy_grad,
        params=current_params,
        tol=solver_tol,
        maxiter=solver_maxiter,
        method=solver,
        regularization=regularization,
    )

    # INNA update:
    # θ_{k+1} = θ_k + γ[-a·θ_k - b·ψ_k - β·η_k]
    # ψ_{k+1} = ψ_k + γ[-a·θ_k - b·ψ_k]

    # Compute the common term: -a·θ - b·ψ
    common_term = jax.tree.map(
        lambda theta, psi_val: -a * theta - b * psi_val,
        current_params,
        psi
    )

    # Update θ: θ + γ·(common_term - β·η)
    updated_params = jax.tree.map(
        lambda theta, common, eta_val: theta + gamma * (common - beta * eta_val),
        current_params,
        common_term,
        eta
    )

    # Update ψ: ψ + γ·common_term
    updated_psi = jax.tree.map(
        lambda psi_val, common: psi_val + gamma * common,
        psi,
        common_term
    )

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
    psi_norm = jnp.sqrt(
        sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), updated_psi)))
    )

    step_info = {
        "gradient_norm": grad_norm,
        "riemann_gradient_norm": riemann_grad_norm,
        "eta_norm": eta_norm,
        "param_norm": param_norm,
        "psi_norm": psi_norm,
        "energy": energy,
        "internal_energy": energy_breakdown["internal_energy"],
        "linear_energy": energy_breakdown["linear_energy"],
        "interaction_energy": energy_breakdown["interaction_energy"],
        "solver_iterations": solver_info.get("iterations", 0),
    }

    if only_return_params:
        return updated_params, updated_psi, step_info

    # Create updated parametric model
    updated_parametric_model = nnx.merge(graphdef, updated_params)

    return updated_parametric_model, updated_psi, step_info


def initialize_psi(params: PyTree, method: str = "zeros") -> PyTree:
    """
    Initialize the auxiliary variable ψ with the same structure as params.

    Args:
        params: Parameter PyTree to match structure
        method: Initialization method ("zeros" or "copy")

    Returns:
        psi: Initialized auxiliary variable
    """
    if method == "zeros":
        return jax.tree.map(lambda x: jnp.zeros_like(x), params)
    elif method == "copy":
        return jax.tree.map(lambda x: x.copy(), params)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
