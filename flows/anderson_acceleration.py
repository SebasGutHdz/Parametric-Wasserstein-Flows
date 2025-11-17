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
from flows.anderson_acceleration_step import anderson_step

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel


def anderson_method(
    parametric_model: ParametricModel,
    batch_size: int,
    test_data_set: Array,
    G_mat: G_matrix,
    potential: Potential,
    initial_params: Optional[PyTree] = None,
    n_iterations: int = 100,
    step_size: float = 0.01,
    memory_size: int = 5,
    relaxation: float = 1.0,
    anderson_tol: float = 1e-6,
    solver: str = "minres",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    l2_reg_gamma: float = 1e-6,
    convergence_tol: float = 1e-6,
    plot_intermediate=False,
    save_param_trajectory=False,
) -> Tuple[PyTree, Dict]:
    """
    Anderson-accelerated gradient flow method for Wasserstein gradient flow.

    Iteratively applies anderson_step to solve:
        θ_{n+1} = θ_n - h · G(θ_n)^{-1} · ∇F(θ_n)
    with Anderson acceleration for improved convergence.

    Args:
        parametric_model: ParametricModel instance
        batch_size: Number of samples per iteration
        test_data_set: Test data set for evaluation (num_test_samples, d)
        G_mat: G_matrix object to compute inner products
        potential: Potential object to compute energy and gradient
        initial_params: Initial parameters θ_0
        n_iterations: Maximum number of iterations
        step_size: Step size h for the fixed-point iteration  fixed point map: (Id+h*G^{-1}*∇F)(x) = x
        memory_size: Number of previous iterations to use for Anderson acceleration (m)
        mixing_parameter: Mixing parameter β for Anderson acceleration
        anderson_tol: Tolerance for Anderson subproblem solver
        solver: Linear solver to use ('minres' or 'cg')
        solver_tol: Tolerance for the linear solver
        solver_maxiter: Maximum iterations for the linear solver
        regularization: Regularization parameter for the linear system
        convergence_tol: Tolerance for convergence (based on residual norm)
        verbose: Whether to print progress information

    Returns:
        final_params: Final parameters θ_n after n_iterations
        history: Dictionary containing:
            - 'params': List of parameters at each iteration
            - 'residuals': List of residual norms at each iteration
            - 'energies': List of energies at each iteration
            - 'gamma_history': List of gamma coefficients at each iteration
            - 'converged': Whether the method converged
            - 'final_iteration': Final iteration number
    """

    # Initialize histories
    param_history = None
    residual_history = None
    param_diffs = None
    residual_diffs = None
    if initial_params is None:
        _, current_params = nnx.split(parametric_model)
    else:
        current_params = initial_params

    # Storage for tracking progress
    params_trajectory = [initial_params]
    residual_norms = []
    energy_trajectory = []
    gamma_history = []
    # Obtain problem dimension from test data set
    problem_dim = test_data_set.shape[1]
    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)
    # Generate initial batch of reference samples

    converged = False

    print(f"Starting Anderson-accelerated gradient flow")
    print(f"  n_iterations: {n_iterations}")
    print(f"  step_size: {step_size}")
    print(f"  memory_size: {memory_size}")
    print(f"  mixing_parameter: {relaxation}")
    print("-" * 60)

    for iteration in range(n_iterations):
        key, subkey = jax.random.split(key)
        z_samples = jax.random.normal(subkey, (batch_size, problem_dim))
        # Perform Anderson acceleration step
        param_history, residual_history, param_diffs, residual_diffs = anderson_step(
            parametric_model=parametric_model,
            current_params=current_params,
            param_history=param_history,
            residual_history=residual_history,
            param_diff=param_diffs,
            residual_diff=residual_diffs,
            G_mat=G_mat,
            potential=potential,
            z_samples=z_samples,
            step_size=step_size,
            memory_size=memory_size,
            relaxation=relaxation,
            anderson_tol=anderson_tol,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            regularization=regularization,
            l2_reg_gamma=l2_reg_gamma,
        )

        # Extract new parameters (newest in history)
        current_params = param_history[0]
        current_residual = residual_history[0]

        # Compute residual norm using G-matrix inner product at the test data set
        residual_norm_sq = G_mat.inner_product(
            current_residual, current_residual, test_data_set
        )
        if residual_norm_sq >= -1e-10:  # some tolerance for numerical error
            residual_norm = jnp.sqrt(jnp.maximum(residual_norm_sq, 0.0))
        else:
            raise ValueError("Non-positive residual norm squared")

        # Compute energy at current parameters
        energy, x_samples, _, _, _ = potential.evaluate_energy(
            parametric_model=parametric_model,
            z_samples=test_data_set,
            params=current_params,
        )

        # Store trajectory information
        if save_param_trajectory or len(params_trajectory) == 1:
            params_trajectory.append(current_params)
        else:
            params_trajectory[-1] = current_params

        residual_norms.append(float(residual_norm))
        energy_trajectory.append(float(energy))

        # Print progress
        if iteration % 10 == 0 or iteration < 5:
            print(
                f"Iter {iteration:4d} | "
                f"Energy: {energy:12.6e} | "
                f"Residual: {residual_norm:12.6e} | "
            )
            if plot_intermediate:
                # Display current samples of current model
                plt.figure(figsize=(6, 6))
                plt.scatter(
                    x_samples[:, 0], x_samples[:, 1], alpha=0.5, label="Model Samples"
                )
                plt.title(f"Samples at Iteration {iteration}")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Check convergence
        if residual_norm < convergence_tol:
            converged = True

            print("-" * 60)
            print(f"Converged at iteration {iteration}!")
            print(f"Final residual norm: {residual_norm:.6e}")
            print(f"Final energy: {energy:.6e}")
            break

    # Final message if not converged
    if not converged:
        print("-" * 60)
        print(f"Reached maximum iterations ({n_iterations})")
        print(f"Final residual norm: {residual_norms[-1]:.6e}")
        print(f"Final energy: {energy_trajectory[-1]:.6e}")

    # Build history dictionary
    history = {
        "params": params_trajectory,
        "residual_norms": residual_norms,
        "riemann_grad_history": [_r / step_size for _r in residual_norms],
        "energies": energy_trajectory,
        "final_iteration": iteration if converged else n_iterations - 1,
        "param_history": param_history,
        "residual_history": residual_history,
        "param_diffs": param_diffs,
        "residual_diffs": residual_diffs,
    }

    return current_params, history
