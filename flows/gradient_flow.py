import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array, PyTree
from typing import Tuple, Any
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


def move_to_device(pytree: Any, device) -> Any:
    """Recursively moves all JAX arrays in a PyTree to the specified device."""
    return jax.tree.map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x, pytree
    )


def run_gradient_flow(
    parametric_model: ParametricModel,
    z_samples: Array,
    G_mat: G_matrix,
    potential: Potential,
    N_samples: int = 100,
    h: float = 0.01,
    solver: str = "minres",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    regularization: float = 1e-6,
    progress_every: int = 10,
) -> dict:
    """
    Run complete gradient flow integration with any LinearPotential

    Args:
        parametric_model: Initial ParametricModel instance
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance defining the energy functional
        solver: str type of solver, choose from cg, and minres
        h: Time step size
        max_iterations: Maximum number of gradient flow steps
        tolerance: Convergence tolerance for energy
        use_regularization: Whether to use regularized CG solver
        progress_every: Print progress every N iterations

    Returns:
        results: Dictionary containing energy history, solver stats, etc.
    """

    current_parametric_model = parametric_model

    # Initialize tracking
    energy_history = []
    solver_stats = []
    param_norms = []
    sample_history = []  # Store samples at key iterations for visualization
    euclid_grad_norm_history = []
    riemann_grad_norm_history = []

    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)
    # with jax.default_device(device):

    p_bar = tqdm(range(max_iterations - 1), desc="Gradient Flow Progress")

    for iteration in p_bar:

        if iteration == 0:
            _, samples0, _, _, _ = potential.evaluate_energy(
                current_parametric_model, z_samples
            )
        # Generate key and samples for evaluation
        key, subkey = jax.random.split(key)
        z_samples_eval = jax.random.normal(subkey, (N_samples, current_parametric_model.problem_dimension))
        # Perform gradient flow step
        current_parametric_model, step_info = gradient_flow_step(
            current_parametric_model,
            z_samples_eval,
            G_mat,
            potential,
            step_size=h,
            solver=solver,
            solver_tol=tolerance,
            regularization=regularization,
        )

        # Evaluate new energy
        _, current_params = nnx.split(current_parametric_model)
        current_energy = step_info["energy"]

        # Store diagnostics
        energy_history.append(float(step_info["energy"]))
        solver_stats.append(step_info)
        param_norm = jnp.sqrt(
            sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), current_params)))
        )
        param_norms.append(float(param_norm))
        euclid_grad_norm_history.append(step_info["gradient_norm"])
        riemann_grad_norm_history.append(step_info["riemann_gradient_norm"])

        p_bar.set_postfix(
            {
                "Energy": f"{step_info['energy']:.6f}",
                "Linear": f"{step_info['linear_energy']:.6f}",
                "Internal": f"{step_info['internal_energy']:.6f}",
                "Interaction": f"{step_info['interaction_energy']:.6f}",
            }
        )

        # Progress reporting
        if (
            iteration % progress_every == 0 and iteration > 0
        ) or iteration == max_iterations - 2:
            current_energy, samples1, _, _, _ = potential.evaluate_energy(
                current_parametric_model, z_samples, current_params
            )
            sample_history.append(samples1)
            print(
                f"Iter {iteration:3d}: Energy = {step_info['energy']:.6f}, "
                f"Grad norm: {step_info['gradient_norm']:.2e}"
            )


            try: 
                fig = plot_gradient_flow(samples0, samples1, potential, current_energy, iteration, progress_every)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                # Update previous samples
            except Exception as e:
                print("PLotting failed due to the folowwing error:")
                print(e)
            samples0 = samples1
        if iteration == 0:
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
            print(f"Energy increment below tolerance at {iteration}")
            break

    # Final summary
    final_energy = energy_history[-1]
    total_decrease = energy_history[0] - final_energy

    print(f"\n=== Integration Complete ===")
    print(f"Total iterations:    {len(energy_history)-1}")
    print(f"Initial energy:      {energy_history[0]:.6f}")
    print(f"Final energy:        {final_energy:.6f}")
    print(f"Total decrease:      {total_decrease:.6f}")
    print(f"Reduction ratio:     {final_energy/energy_history[0]:.4f}")
    print(f"Final param norm:    {param_norms[-1]:.6f}")

    return {
        "final_parametric_model": current_parametric_model,
        "energy_history": energy_history,
        "euclid_grad_norm_history": euclid_grad_norm_history,
        "riemann_grad_norm_history": riemann_grad_norm_history,
        "param_norms": param_norms,
        "sample_history": sample_history,
        "potential": potential,
        "convergence_info": {
            "converged": final_energy < tolerance
            or abs(energy_history[-1] - energy_history[-2]) < tolerance * 1e-2,
            "final_energy": final_energy,
            "total_decrease": total_decrease,
            "iterations": len(energy_history) - 1,
        },
    }
