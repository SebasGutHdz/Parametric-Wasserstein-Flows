import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from geometry.G_matrix import G_matrix
from functionals.functional import Potential
from flows.inna_flow_step import inna_flow_step, initialize_psi
from parametric_model.parametric_model import ParametricModel
from flows.visualization import plot_gradient_flow

from tqdm import tqdm


def run_inna_flow(
    parametric_model: ParametricModel,
    z_samples: Array,
    G_mat: G_matrix,
    potential: Potential,
    N_samples: int = 100,
    gamma: float = 0.01,
    a: float = 0.1,
    b: float = 0.1,
    beta: float = 1.0,
    solver: str = "cg",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    regularization: float = 1e-6,
    progress_every: int = 10,
    plot_intermediate: bool = False,
    psi_init_method: str = "zeros",
) -> dict:
    """
    Run INNA (Inertial Neural Network Algorithm) flow for optimization.

    Implements the dynamical system:
        θ_{k+1} = θ_k + γ[-a·θ_k - b·ψ_k - β·η_k]
        ψ_{k+1} = ψ_k + γ[-a·θ_k - b·ψ_k]

    where η_k = G(θ_k)^{-1} ∇F(θ_k) is the Riemannian gradient.

    Args:
        parametric_model: Initial ParametricModel instance
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance defining the energy functional
        N_samples: Number of samples for Monte Carlo estimation
        gamma: Step size γ
        a: Coupling coefficient for θ terms
        b: Coupling coefficient for ψ terms
        beta: Gradient coefficient β
        solver: Linear solver type ("cg" or "minres")
        max_iterations: Maximum number of INNA steps
        tolerance: Convergence tolerance
        regularization: Regularization parameter for solver
        progress_every: Print/plot progress every N iterations
        plot_intermediate: Whether to plot intermediate results
        psi_init_method: Method to initialize ψ ("zeros" or "copy")

    Returns:
        results: Dictionary containing energy history, gradient norms, etc.
    """

    current_parametric_model = parametric_model

    # Initialize ψ with same structure as parameters
    _, init_params = nnx.split(parametric_model)
    psi = initialize_psi(init_params, method=psi_init_method)

    # Initialize tracking
    energy_history = []
    euclid_grad_norm_history = []
    riemann_grad_norm_history = []
    psi_norm_history = []
    param_norms = []
    sample_history = []

    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)

    p_bar = tqdm(range(max_iterations), desc="INNA Flow Progress")

    for iteration in p_bar:

        if iteration == 0 and plot_intermediate:
            _, samples0, _, _, _ = potential.evaluate_energy(
                current_parametric_model, z_samples
            )

        # Generate fresh samples for evaluation
        key, subkey = jax.random.split(key)
        z_samples_eval = jax.random.normal(
            subkey, (N_samples, current_parametric_model.problem_dimension)
        )

        # Perform INNA flow step
        current_parametric_model, psi, step_info = inna_flow_step(
            current_parametric_model,
            psi,
            z_samples_eval,
            G_mat,
            potential,
            gamma=gamma,
            a=a,
            b=b,
            beta=beta,
            solver=solver,
            solver_tol=tolerance,
            regularization=regularization,
        )

        # Get current parameters
        _, current_params = nnx.split(current_parametric_model)
        current_energy = step_info["energy"]

        # Store diagnostics
        energy_history.append(float(step_info["energy"]))
        euclid_grad_norm_history.append(float(step_info["gradient_norm"]))
        riemann_grad_norm_history.append(float(step_info["riemann_gradient_norm"]))
        psi_norm_history.append(float(step_info["psi_norm"]))
        param_norms.append(float(step_info["param_norm"]))

        p_bar.set_postfix(
            {
                "Energy": f"{step_info['energy']:.6f}",
                "||ψ||": f"{step_info['psi_norm']:.2e}",
                "||∇F||_G": f"{step_info['riemann_gradient_norm']:.2e}",
            }
        )

        # Progress reporting and visualization
        if (
            iteration % progress_every == 0 and iteration > 0
        ) or iteration == max_iterations - 1:
            print(
                f"Iter {iteration:3d}: Energy = {step_info['energy']:.6f}, "
                f"||∇F||_G = {step_info['riemann_gradient_norm']:.2e}, "
                f"||ψ|| = {step_info['psi_norm']:.2e}"
            )

            if plot_intermediate:
                current_energy_eval, samples1, _, _, _ = potential.evaluate_energy(
                    current_parametric_model, z_samples, current_params
                )
                sample_history.append(samples1)

                try:
                    fig = plot_gradient_flow(
                        samples0,
                        samples1,
                        potential,
                        current_energy_eval,
                        iteration,
                        progress_every,
                    )
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                except Exception as e:
                    print(f"Plotting failed: {e}")
                samples0 = samples1

        if iteration == 0 and plot_intermediate:
            _, samples0, _, _, _ = potential.evaluate_energy(
                current_parametric_model, z_samples, current_params
            )

        # Early stopping conditions
        if iteration > 1 and jnp.abs(current_energy) < tolerance:
            print(f"Converged! Energy below tolerance at iteration {iteration}")
            break

        if (
            iteration > 5
            and abs(energy_history[-1] - energy_history[-2]) < tolerance * 1e-2
        ):
            print(f"Energy increment below tolerance at iteration {iteration}")
            break

    # Evaluate energy of the final iterate
    final_energy, final_samples, _, _, _ = potential.evaluate_energy(
        current_parametric_model,
        z_samples,
    )

    # Final summary
    energy_history.append(float(final_energy))
    total_decrease = energy_history[0] - final_energy

    print(f"\n=== INNA Integration Complete ===")
    print(f"Total iterations:    {len(energy_history)-1}")
    print(f"Initial energy:      {energy_history[0]:.6f}")
    print(f"Final energy:        {final_energy:.6f}")
    print(f"Total decrease:      {total_decrease:.6f}")
    print(f"Reduction ratio:     {final_energy/energy_history[0]:.4f}")
    print(f"Final ||ψ||:         {psi_norm_history[-1]:.6f}")
    print(f"Final ||θ||:         {param_norms[-1]:.6f}")

    return {
        "final_parametric_model": current_parametric_model,
        "final_psi": psi,
        "energy_history": energy_history,
        "euclid_grad_norm_history": euclid_grad_norm_history,
        "riemann_grad_norm_history": riemann_grad_norm_history,
        "psi_norm_history": psi_norm_history,
        "param_norms": param_norms,
        "sample_history": sample_history,
        "final_samples": final_samples,
        "potential": potential,
        "inna_params": {"gamma": gamma, "a": a, "b": b, "beta": beta},
        "convergence_info": {
            "converged": final_energy < tolerance
            or (len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < tolerance * 1e-2),
            "final_energy": float(final_energy),
            "total_decrease": float(total_decrease),
            "iterations": len(energy_history) - 1,
        },
    }
