import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import uuid
from flax import nnx
from jaxtyping import Array, PyTree

from flows.aa_tgs_step import aa_tgs_step
from flows.visualization import plot_gradient_flow
from functionals.functional import Potential
from geometry.G_matrix import G_matrix
from parametric_model.parametric_model import ParametricModel


def aa_tgs_method(
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
    solver: str = "cg",
    solver_tol: float = 1e-6,
    solver_maxiter: int = 50,
    regularization: float = 1e-6,
    l2_reg_gamma: float = 1e-6,
    fixed_batch_steps: int = 25,
    restart_eta: float = float("inf"),
    convergence_tol: float = 1e-6,
    plot_intermediate: bool = False,
    plot_frequency: int = 10,
    save_param_trajectory: bool = False,
    run_id: Optional[str] = None,
    method_name: str = "AA-TGS",
) -> Tuple[PyTree, Dict]:
    """
    AA-TGS(m)-accelerated gradient flow method.

    Interface mirrors `anderson_method` so notebook migration can be import-level.
    """
    param_history = None
    residual_history = None
    param_diffs = None
    residual_diffs = None

    graphdef, initial_split_params = nnx.split(parametric_model)
    run_id = run_id or f"{method_name}-{uuid.uuid4().hex[:10]}"
    if hasattr(G_mat, "clear_big_solve_records"):
        G_mat.clear_big_solve_records()
    if hasattr(G_mat, "set_linear_solve_context"):
        G_mat.set_linear_solve_context(run_id=run_id, method_name=method_name)
    if initial_params is None:
        current_params = initial_split_params
    else:
        current_params = initial_params

    params_trajectory = [initial_params]
    residual_norms = []
    energy_trajectory = []

    problem_dim = test_data_set.shape[1]
    key = jax.random.PRNGKey(0)
    converged = False
    t_start = time.perf_counter()
    lsc_big_cum = 0
    ksi_big_cum = 0
    time_big_solve_sec_cum = 0.0
    iter_metrics = []
    processed_big_records = 0

    print("Starting AA-TGS(m)-accelerated gradient flow")
    print(f"  n_iterations: {n_iterations}")
    print(f"  step_size: {step_size}")
    print(f"  memory_size: {memory_size}")
    print("  beta_j policy: constant beta_j = step_size")
    print(f"  fixed_batch_steps: {fixed_batch_steps}")
    print(f"  restart_eta: {restart_eta}")
    print("-" * 60)

    key, subkey = jax.random.split(key)
    z_samples = jax.random.normal(subkey, (batch_size, problem_dim))
    energy_init, samples_prev, _, _, _ = potential.evaluate_energy(
        parametric_model, z_samples=test_data_set
    )
    energy_trajectory.append(float(energy_init))
    z_samples_block = z_samples

    for iteration in range(n_iterations):
        if fixed_batch_steps <= 0:
            key, subkey = jax.random.split(key)
            z_samples_block = jax.random.normal(subkey, (batch_size, problem_dim))
        elif iteration % fixed_batch_steps == 0:
            key, subkey = jax.random.split(key)
            z_samples_block = jax.random.normal(subkey, (batch_size, problem_dim))

        param_history, residual_history, param_diffs, residual_diffs = aa_tgs_step(
            parametric_model=parametric_model,
            current_params=current_params,
            param_history=param_history,
            residual_history=residual_history,
            param_diff=param_diffs,
            residual_diff=residual_diffs,
            G_mat=G_mat,
            potential=potential,
            z_samples=z_samples_block,
            step_size=step_size,
            memory_size=memory_size,
            relaxation=relaxation,
            anderson_tol=anderson_tol,
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            regularization=regularization,
            l2_reg_gamma=l2_reg_gamma,
            restart_eta=restart_eta,
            graphdef=graphdef,
            outer_iter=iteration,
            run_id=run_id,
            method_name=method_name,
        )
        if hasattr(G_mat, "get_big_solve_records"):
            all_records = G_mat.get_big_solve_records()
            new_records = all_records[processed_big_records:]
            processed_big_records = len(all_records)
            lsc_big_cum += len(new_records)
            for rec in new_records:
                if rec.get("iterations") is not None:
                    ksi_big_cum += int(rec["iterations"])
                time_big_solve_sec_cum += float(rec.get("elapsed_sec") or 0.0)

        if iteration == 0:
            init_params = param_history[-1]
            init_residual = residual_history[-1]
            init_res_norm_sq = G_mat.inner_product(
                init_residual,
                init_residual,
                z_samples_block,
                params=init_params,
            )
            residual_norms.append(float(jnp.sqrt(jnp.maximum(init_res_norm_sq, 0.0))))

        current_params = param_history[0]
        current_residual = residual_history[0]

        residual_norm_sq = G_mat.inner_product(
            current_residual,
            current_residual,
            test_data_set,
            params=current_params,
        )
        if residual_norm_sq >= -1e-10:
            residual_norm = jnp.sqrt(jnp.maximum(residual_norm_sq, 0.0))
        else:
            raise ValueError("Non-positive residual norm squared")

        energy, x_samples, _, _, _ = potential.evaluate_energy(
            parametric_model=parametric_model,
            z_samples=test_data_set,
            params=current_params,
        )

        if save_param_trajectory or len(params_trajectory) == 1:
            params_trajectory.append(current_params)
        else:
            params_trajectory[-1] = current_params

        residual_norms.append(float(residual_norm))
        energy_trajectory.append(float(energy))
        iter_metrics.append(
            {
                "run_id": run_id,
                "method": method_name,
                "outer_iter": int(iteration),
                "energy": float(energy),
                "residual_norm": float(residual_norm),
                "test_accuracy": None,
                "elapsed_total_sec": float(time.perf_counter() - t_start),
                "lsc_big_cum": int(lsc_big_cum),
                "ksi_big_cum": int(ksi_big_cum),
                "time_big_solve_sec_cum": float(time_big_solve_sec_cum),
            }
        )

        if iteration % plot_frequency == 0 or iteration < 5:
            print(
                f"Iter {iteration:4d} | "
                f"Energy: {energy:12.6e} | "
                f"Residual: {residual_norm:12.6e} | "
            )
            if plot_intermediate:
                try:
                    fig = plot_gradient_flow(
                        samples_prev,
                        x_samples,
                        potential,
                        energy,
                        iteration,
                        plot_frequency,
                    )
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                except Exception as e:
                    print("Plotting failed due to the following error:")
                    print(e)
                samples_prev = x_samples

        if residual_norm < convergence_tol:
            converged = True
            print("-" * 60)
            print(f"Converged at iteration {iteration}!")
            print(f"Final residual norm: {residual_norm:.6e}")
            print(f"Final energy: {energy:.6e}")
            break

    if not converged:
        print("-" * 60)
        print(f"Reached maximum iterations ({n_iterations})")
        print(f"Final residual norm: {residual_norms[-1]:.6e}")
        print(f"Final energy: {energy_trajectory[-1]:.6e}")

    final_parametric_model = nnx.merge(graphdef, current_params)

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
        "final_parametric_model": final_parametric_model,
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

    return current_params, history
