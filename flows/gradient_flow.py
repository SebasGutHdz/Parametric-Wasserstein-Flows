import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device

from geometry.G_matrix import G_matrix

from functionals.functional import Potential
from flows.gradient_flow_step import gradient_flow_step
from parametric_model.parametric_model import ParametricModel
from flows.visualization import plot_gradient_flow


from tqdm import tqdm
import time
import uuid


def move_to_device(pytree: Any, device) -> Any:
    """Recursively moves all JAX arrays in a PyTree to the specified device."""
    return jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x, pytree
    )


def run_gradient_flow(
    parametric_model: ParametricModel,
    test_data_set: Optional[Array] = None,
    G_mat: Optional[G_matrix] = None,
    potential: Optional[Potential] = None,
    batch_size: int = 100,
    step_size: float = 0.01,
    solver: str = "cg",
    n_iterations: int = 100,
    solver_tol: float = 1e-6,
    convergence_tol: float = 1e-6,
    regularization: float = 1e-6,
    solver_maxiter: int = 50,
    fixed_batch_steps: int = 1,
    plot_frequency: int = 10,
    initial_params: Optional[PyTree] = None,
    save_param_trajectory: bool = False,
    run_id: Optional[str] = None,
    method_name: str = "GF",
    # Legacy aliases for backward compatibility
    z_samples: Optional[Array] = None,
    N_samples: Optional[int] = None,
    h: Optional[float] = None,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
    progress_every: Optional[int] = None,
    init_params: Optional[PyTree] = None,
) -> dict:
    """
    Run complete gradient flow integration with any Potential.

    Args:
        parametric_model: Initial ParametricModel instance
        test_data_set: Fixed evaluation samples; used for energy and residual norm reporting
        G_mat: G-matrix object for linear system solving
        potential: Potential instance defining the energy functional
        batch_size: Number of training samples drawn from parametric_model.sampler per batch
        step_size: Time step size h
        solver: Linear solver type ('cg', 'minres', 'gmres')
        n_iterations: Maximum number of gradient flow steps
        solver_tol: Tolerance for the linear solver
        convergence_tol: Tolerance for convergence check (residual norm on test_data_set)
        regularization: Regularization parameter for the linear solver
        solver_maxiter: Maximum iterations for the linear solver
        fixed_batch_steps: Resample training batch every this many iterations (1 = every step)
        plot_frequency: Print/plot progress every N iterations
        initial_params: Initial parameters (optional; uses model params if None)
        save_param_trajectory: Whether to save full parameter trajectory
        run_id: Optional run identifier for instrumentation
        method_name: Method name tag for logging

    Returns:
        results: Dictionary with aligned keys matching anderson_method output
    """
    # Legacy alias resolution
    if z_samples is not None and test_data_set is None:
        test_data_set = z_samples
    if N_samples is not None:
        batch_size = N_samples
    if h is not None:
        step_size = h
    if max_iterations is not None:
        n_iterations = max_iterations
    if tolerance is not None:
        solver_tol = tolerance
        convergence_tol = tolerance
    if progress_every is not None:
        plot_frequency = progress_every
    if init_params is not None and initial_params is None:
        initial_params = init_params

    # Split ONCE at the beginning - this is the only split in the entire flow
    graphdef, current_params = nnx.split(parametric_model)
    run_id = run_id or f"{method_name}-{uuid.uuid4().hex[:10]}"
    if hasattr(G_mat, "clear_big_solve_records"):
        G_mat.clear_big_solve_records()
    if hasattr(G_mat, "set_linear_solve_context"):
        G_mat.set_linear_solve_context(run_id=run_id, method_name=method_name)

    if initial_params is not None:
        current_params = initial_params

    # Initialize tracking
    energy_history = []
    residual_norms = []
    solver_stats = []
    param_norms = []
    sample_history = []
    euclid_grad_norm_history = []
    params_trajectory = [initial_params]
    converged = False
    solver_x0 = None  # warm start for G-system solver; updated each iteration

    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)

    p_bar = tqdm(range(n_iterations), desc="Gradient Flow Progress")
    t_start = time.perf_counter()
    lsc_big_cum = 0
    ksi_big_cum = 0
    time_big_solve_sec_cum = 0.0
    iter_metrics = []

    # Draw initial training batch
    key, subkey = jax.random.split(key)
    z_train_block = parametric_model.sampler(subkey, batch_size)

    # Evaluate initial energy on test_data_set (consistent with anderson_method)
    energy_init, samples_prev, _, _, _ = potential.evaluate_energy(
        parametric_model, z_samples=test_data_set
    )

    energy_history.append(float(energy_init))
    iter_metrics.append(
        {
            "run_id": run_id,
            "method": method_name,
            "outer_iter": -1,
            "energy": float(energy_init),
            "residual_norm": float("nan"),
            "test_accuracy": None,
            "elapsed_total_sec": 0.0,
            "lsc_big_cum": 0,
            "ksi_big_cum": 0,
            "time_big_solve_sec_cum": 0.0,
        }
    )

    for iteration in p_bar:

        # Resample training batch according to fixed_batch_steps
        if fixed_batch_steps <= 0:
            key, subkey = jax.random.split(key)
            z_train_block = parametric_model.sampler(subkey, batch_size)
        elif iteration % fixed_batch_steps == 0:
            key, subkey = jax.random.split(key)
            z_train_block = parametric_model.sampler(subkey, batch_size)

        # Save params before step to compute fixed-point residual afterwards
        params_before = current_params

        # Perform gradient flow step on training batch
        current_params, step_info = gradient_flow_step(
            parametric_model,
            z_train_block,
            G_mat,
            potential,
            step_size=step_size,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            regularization=regularization,
            solver_x0=None,
            only_return_params=True,
            graphdef=graphdef,
            current_params=current_params,
            outer_iter=iteration,
            phase="step_update",
            run_id=run_id,
            method_name=method_name,
        )
        solver_x0 = step_info["eta"]  # warm start for next iteration

        # Fixed-point residual norm on test_data_set
        # step_vec = theta_before - theta_after = h * G^{-1} grad_F  (the update direction)
        # ||step_vec||_G on test_data_set matches anderson_method's residual norm convention
        step_vec = jax.tree.map(lambda a, b: a - b, params_before, current_params)
        residual_norm_sq = G_mat.inner_product(
            step_vec, step_vec, test_data_set, params=current_params
        )
        residual_norm = jnp.sqrt(jnp.maximum(residual_norm_sq, 0.0))

        # Store diagnostics
        current_energy = step_info["energy"]
        energy_history.append(float(current_energy))
        residual_norms.append(float(residual_norm))
        solver_stats.append(step_info)
        param_norms.append(float(step_info["param_norm"]))
        euclid_grad_norm_history.append(step_info["gradient_norm"])
        if save_param_trajectory:
            params_trajectory.append(current_params)
        else:
            params_trajectory[-1] = current_params

        lsc_big_cum += 1
        if step_info["big_solve_iterations"] is not None:
            ksi_big_cum += int(step_info["big_solve_iterations"])
        time_big_solve_sec_cum += float(step_info["big_solve_elapsed_sec"] or 0.0)
        iter_metrics.append(
            {
                "run_id": run_id,
                "method": method_name,
                "outer_iter": int(iteration),
                "energy": float(current_energy),
                "residual_norm": float(residual_norm),
                "test_accuracy": None,
                "elapsed_total_sec": float(time.perf_counter() - t_start),
                "lsc_big_cum": int(lsc_big_cum),
                "ksi_big_cum": int(ksi_big_cum),
                "time_big_solve_sec_cum": float(time_big_solve_sec_cum),
            }
        )

        p_bar.set_postfix(
            {
                "Energy": f"{step_info['energy']:.6f}",
                "Linear": f"{step_info['linear_energy']:.6f}",
                "Internal": f"{step_info['internal_energy']:.6f}",
                "Interaction": f"{step_info['interaction_energy']:.3e}",
            }
        )

        # Progress reporting
        if (
            iteration % plot_frequency == 0 and iteration > 0
        ) or iteration == n_iterations - 2:
            current_energy_eval, samples1, _, _, _ = potential.evaluate_energy(
                parametric_model, test_data_set, current_params
            )
            sample_history.append(samples1)
            print(
                f"Iter {iteration:3d}: Energy = {step_info['energy']:.6f}, "
                f"Residual: {residual_norm:.2e}"
            )

            try:
                fig = plot_gradient_flow(
                    samples0,
                    samples1,
                    potential,
                    current_energy_eval,
                    iteration,
                    plot_frequency,
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            except Exception as e:
                print("Plotting failed due to the following error:")
                print(e)
            samples0 = samples1

        if iteration == 0:
            # for plotting sample at previous checkpoint vs current
            _, samples0, _, _, _ = potential.evaluate_energy(
                parametric_model, test_data_set, current_params
            )

        # Check convergence (residual norm on test_data_set, same criterion as anderson_method)
        if residual_norm < convergence_tol:
            converged = True
            print("-" * 60)
            print(f"Converged at iteration {iteration}!")
            print(f"Final residual norm: {residual_norm:.6e}")
            print(f"Final energy: {current_energy:.6e}")
            break

    # Final message if not converged
    if not converged:
        print("-" * 60)
        print(f"Reached maximum iterations ({n_iterations})")
        print(f"Final residual norm: {residual_norms[-1]:.6e}")
        print(f"Final energy: {energy_history[-1]:.6e}")

    # Merge ONCE at the end to get the final model
    current_parametric_model = nnx.merge(graphdef, current_params)

    # Evaluate energy of the final iterate on test_data_set
    final_energy, samples0, _, _, _ = potential.evaluate_energy(
        current_parametric_model,
        test_data_set,
    )

    energy_history.append(float(final_energy))
    total_decrease = energy_history[0] - float(final_energy)

    print(f"\n=== Integration Complete ===")
    print(f"Total iterations:    {len(energy_history)-1}")
    print(f"Initial energy:      {energy_history[0]:.6f}")
    print(f"Final energy:        {float(final_energy):.6f}")
    print(f"Total decrease:      {total_decrease:.6f}")
    print(f"Reduction ratio:     {float(final_energy)/energy_history[0]:.4f}")
    if param_norms:
        print(f"Final param norm:    {param_norms[-1]:.6f}")

    # riemann_grad_history = residual_norms / step_size = ||G^{-1} grad_F||_G
    # matches the convention in anderson_method
    riemann_grad_history = [r / step_size for r in residual_norms]

    return {
        # Primary aligned keys (match anderson_method output)
        "energies": energy_history,
        "residual_norms": residual_norms,
        "riemann_grad_history": riemann_grad_history,
        "final_parametric_model": current_parametric_model,
        "params": params_trajectory,
        # Legacy aliases (same list objects, no extra memory cost)
        "energy_history": energy_history,
        "riemann_grad_norm_history": riemann_grad_history,
        # Other outputs
        "euclid_grad_norm_history": euclid_grad_norm_history,
        "param_norms": param_norms,
        "sample_history": sample_history,
        "potential": potential,
        "convergence_info": {
            "converged": converged,
            "final_energy": float(final_energy),
            "final_residual_norm": float(residual_norms[-1]) if residual_norms else float("nan"),
            "total_decrease": total_decrease,
            "iterations": len(energy_history) - 1,
        },
        "instrumentation": {
            "run_id": run_id,
            "method": method_name,
            "lsc_big": int(lsc_big_cum),
            "ksi_big": int(ksi_big_cum),
            "ksi_big_mean": float(ksi_big_cum / max(lsc_big_cum, 1)),
            "time_big_solve_sec": float(time_big_solve_sec_cum),
            "elapsed_total_sec": float(time.perf_counter() - t_start),
            "iter_metrics": iter_metrics,
            "big_solve_records": G_mat.get_big_solve_records()
            if hasattr(G_mat, "get_big_solve_records")
            else [],
        },
    }
