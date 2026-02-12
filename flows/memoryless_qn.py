import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, List, Dict, Optional, Literal, Callable, Any
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
from jax import Device

from geometry.G_matrix import G_matrix
from geometry.lin_alg_solvers import minres
from flows.memoryless_qn_step import memoryless_qn_step

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel


def memoryless_qn_method(
    parametric_model: ParametricModel,
    batch_size: int,
    test_data_set: Array,
    G_mat: G_matrix,
    potential: Potential,
    initial_params: Optional[PyTree] = None,
    n_iterations: int = 100,
    step_size: float = 0.01,
    solver: str = "cg",  # \
    solver_tol: float = 1e-5,  #  > linear solver parameters for the computation of Riemannian gradient
    solver_maxiter: int = 50,  # /
    solver_regularization: float = 1e-6, 
    ensure_descent: bool = False,  
    hessian_update_strategy: Literal[
        "preconvex", "BFGS"
    ] = "BFGS",  # choice of \varphi_{k-1}
    regularization_strategy: Literal[
        "Li-Fukushima", "Powell"
    ] = "Li-Fukushima",  # choice of z_{k-1}
    spectral_scaling: bool = True,
    convergence_tol: float = 1e-6,
    plot_intermediate=False,
    plot_frequency: int = 10,
    save_param_trajectory=False,
    verbose: bool = True,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    diagnostic_sample_size: Optional[int] = None,
) -> Tuple[PyTree, Dict]:
    """
    Memoryless quasi-Newton method for Wasserstein gradient flow.

    """
    if initial_params is None:
        _, initial_params = nnx.split(parametric_model)

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

    if verbose:
        print(f"Starting memoryless qN method")
        print(f"  n_iterations: {n_iterations}")
        print(f"  step_size: {step_size}")
        print(f"  hessian strategy: {hessian_update_strategy}")
        print(f"  regularization strategy: {regularization_strategy}")
        print("-" * 60)

    # evaluate initial energy
    key, subkey = jax.random.split(key)
    z_samples = jax.random.normal(subkey, (batch_size, problem_dim))
    energy_init, _, _, _, _ = potential.evaluate_energy(
        parametric_model, z_samples=z_samples
    )
    energy_trajectory.append(float(energy_init))

    params = initial_params
    delta_params = None
    grad_prev = None

    for iteration in range(n_iterations):
        key, subkey = jax.random.split(key)
        z_samples = jax.random.normal(subkey, (batch_size, problem_dim))

        new_params, delta_params, grad_prev = memoryless_qn_step(
            parametric_model,
            params,
            delta_params,
            grad_prev,
            G_mat,
            potential,
            z_samples,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            solver_regularization=solver_regularization,
            step_size=step_size,
            ensure_descent=ensure_descent,
            hessian_update_strategy=hessian_update_strategy,
            regularization_strategy=regularization_strategy,
            spectral_scaling=spectral_scaling,
        )

        grad_norm_sq = G_mat.inner_product(
            grad_prev, grad_prev, test_data_set, params=params,
        )
        if grad_norm_sq >= -1e-10:  # some tolerance for numerical error
            residual_norm = jnp.sqrt(jnp.maximum(grad_norm_sq, 0.0)) * step_size
        else:
            raise ValueError("Non-positive residual norm squared")

        params = new_params
        # Compute energy at current parameters
        energy, x_samples, _, _, _ = potential.evaluate_energy(
            parametric_model=parametric_model,
            z_samples=test_data_set,
            params=params,
        )

        # Store trajectory information
        if save_param_trajectory or len(params_trajectory) == 1:
            params_trajectory.append(params)
        else:
            params_trajectory[-1] = params

        residual_norms.append(float(residual_norm))
        energy_trajectory.append(float(energy))

        if progress_callback is not None:
            scatter_samples = None
            if diagnostic_sample_size is not None and diagnostic_sample_size > 0:
                key, diag_key = jax.random.split(key)
                z_diag = jax.random.normal(diag_key, (diagnostic_sample_size, problem_dim))
                scatter_samples = parametric_model(z_diag, params=params)
            progress_callback(
                {
                    "method": "memoryless_qn",
                    "iteration": iteration,
                    "max_iterations": n_iterations,
                    "energy": float(energy),
                    "riemann_grad_norm": float(residual_norm / step_size),
                    "scatter_samples": scatter_samples,
                    "converged": False,
                }
            )

        # Print progress
        if (iteration % plot_frequency == 0 or iteration < 5) and verbose:
            print(
                f"Iter {iteration:4d} | "
                f"Energy: {energy:12.6e} | "
                f"Residual: {residual_norm:12.6e} | "
            )
            if plot_intermediate:
                if iteration == 0:
                    x_max = jnp.max(jnp.abs(x_samples[:, 0])) * 1.1
                    y_max = jnp.max(jnp.abs(x_samples[:, 1])) * 1.1
                # Display current samples of current model
                plt.figure(figsize=(6, 6))
                plt.scatter(
                    x_samples[:, 0], x_samples[:, 1], alpha=0.5, label="Model Samples"
                )
                plt.title(f"Samples at Iteration {iteration}")
                plt.xlabel("x1")
                plt.ylabel("x2")
                plt.xlim(-x_max, x_max)
                plt.ylim(-y_max, y_max)
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                plt.show()

        # Check convergence
        if residual_norm < convergence_tol:
            converged = True

            if verbose:
                print("-" * 60)
                print(f"Converged at iteration {iteration}!")
                print(f"Final residual norm: {residual_norm:.6e}")
                print(f"Final energy: {energy:.6e}")
            break

    # Final message if not converged
    if not converged:
        if verbose:
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
    }

    return params, history
