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

    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)
    n_samples = len(z_samples)
    # with jax.default_device(device):

    p_bar = tqdm(range(max_iterations - 1), desc="Gradient Flow Progress")

    for iteration in p_bar:

        if iteration == 0:
            _, samples0, _, _, _ = potential.evaluate_energy(
                current_parametric_model, z_samples
            )
        # Generate key and samples for evaluation
        key, subkey = jax.random.split(key)
        z_samples_eval = jax.random.normal(subkey, (n_samples, 2))
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

            fig = plt.figure(figsize=(12, 5))

            # 3D view
            ax3d = fig.add_subplot(1, 2, 1, projection="3d")

            # Get potential surface for region around particles
            all_samples = jnp.vstack([samples0, samples1])
            x_range = jnp.linspace(
                all_samples[:, 0].min() - 0.5, all_samples[:, 0].max() + 0.5, 100
            )
            y_range = jnp.linspace(
                all_samples[:, 1].min() - 0.5, all_samples[:, 1].max() + 0.5, 100
            )
            X, Y = jnp.meshgrid(x_range, y_range)
            # Z = jnp.array([[potential.potential_fn(x, y, **potential.potential_kwargs) for x in x_range] for y in y_range])
            Z = potential.linear.potential_fn(
                jnp.stack([X.ravel(), Y.ravel()], axis=-1),
                **potential.linear.potential_kwargs,
            ).reshape(X.shape)

            # Plot surface
            ax3d.plot_surface(X, Y, Z, alpha=0.4, cmap="viridis")

            # Particles on surface (elevated by potential)
            # surface_z0 = jnp.array([potential.potential_fn(x, y, **potential.potential_kwargs) for x, y in samples0])
            # surface_z1 = jnp.array([potential.potential_fn(x, y, **potential.potential_kwargs) for x, y in samples1])
            surface_z0 = potential.linear.potential_fn(
                samples0, **potential.linear.potential_kwargs
            )
            surface_z1 = potential.linear.potential_fn(
                samples1, **potential.linear.potential_kwargs
            )

            ax3d.scatter(
                samples0[:, 0],
                samples0[:, 1],
                surface_z0,
                c="green",
                s=10,
                alpha=0.6,
                label=f"Iteration {iteration-progress_every}",
            )
            ax3d.scatter(
                samples1[:, 0],
                samples1[:, 1],
                surface_z1,
                c="red",
                s=10,
                alpha=0.8,
                label=f"Iteration {iteration}",
            )

            # Particles on contour (base level)
            base_z = Z.min() - 0.15 * (Z.max() - Z.min())
            ax3d.scatter(
                samples0[:, 0], samples0[:, 1], base_z, c="lightgreen", s=5, alpha=0.4
            )
            ax3d.scatter(
                samples1[:, 0], samples1[:, 1], base_z, c="pink", s=5, alpha=0.4
            )

            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Potential")
            ax3d.set_title(f"3D View - Energy = {current_energy:.6f}")
            ax3d.legend()

            # 2D contour view (similar to original)
            ax2d = fig.add_subplot(1, 2, 2)
            ax2d = potential.linear.plot_function(fig=fig, ax=ax2d)
            ax2d.scatter(
                samples0[:, 0],
                samples0[:, 1],
                color="green",
                s=5,
                alpha=0.6,
                label=f"Iteration {iteration-progress_every}",
            )
            ax2d.scatter(
                samples1[:, 0],
                samples1[:, 1],
                color="red",
                s=5,
                alpha=0.8,
                label=f"Iteration {iteration}",
            )
            ax2d.set_title(f"Contour View - Iteration {iteration}")
            ax2d.legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            # Update previous samples
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
