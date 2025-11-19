import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def visualize_gradient_flow_results(results: dict, figsize: tuple = (15, 10)):
    """
    Visualize the gradient flow results

    Args:
        results: Results dictionary from run_gradient_flow
        figsize: Figure size for plots
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Energy decay plot
    axes[0, 0].plot(results["energy_history"])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].set_title("Energy Decay")
    axes[0, 0].grid(True)
    # axes[0,0].set_yscale('log')

    # Parameter norm evolution
    axes[0, 1].plot(results["param_norms"])
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Parameter Norm")
    axes[0, 1].set_title("Parameter Evolution")
    axes[0, 1].grid(True)

    # Initial vs Final samples
    initial_samples = results["sample_history"][0]
    final_samples = results["sample_history"][-1]

    x_min, x_max = min(initial_samples[:, 0].min(), final_samples[:, 0].min()), max(
        initial_samples[:, 0].max(), final_samples[:, 0].max()
    )
    y_min, y_max = min(initial_samples[:, 1].min(), final_samples[:, 1].min()), max(
        initial_samples[:, 1].max(), final_samples[:, 1].max()
    )

    axes[1, 0].scatter(
        initial_samples[:, 0], initial_samples[:, 1], alpha=0.5, s=1, label="Initial"
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_title("Initial Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(y_min, y_max)

    axes[1, 1].scatter(
        final_samples[:, 0], final_samples[:, 1], alpha=0.5, s=1, c="red", label="Final"
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_title("Final Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_gradient_flow(samples_prev, samples_cur, potential, current_energy, iteration, iteration_diff):
    fig = plt.figure(figsize=(12, 5))

    # 3D view
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # Get potential surface for region around particles
    all_samples = jnp.vstack([samples_prev, samples_cur])
    x_range = jnp.linspace(
        all_samples[:, 0].min() - 0.5, all_samples[:, 0].max() + 0.5, 100
    )
    y_range = jnp.linspace(
        all_samples[:, 1].min() - 0.5, all_samples[:, 1].max() + 0.5, 100
    )
    x_bounds = (x_range[0], x_range[-1])
    y_bounds = (x_range[0], x_range[-1])
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
        samples_prev, **potential.linear.potential_kwargs
    )
    surface_z1 = potential.linear.potential_fn(
        samples_cur, **potential.linear.potential_kwargs
    )

    ax3d.scatter(
        samples_prev[:, 0],
        samples_prev[:, 1],
        surface_z0,
        c="green",
        s=10,
        alpha=0.6,
        label=f"Iteration {iteration-iteration_diff}",
    )
    ax3d.scatter(
        samples_cur[:, 0],
        samples_cur[:, 1],
        surface_z1,
        c="red",
        s=10,
        alpha=0.8,
        label=f"Iteration {iteration}",
    )

    # Particles on contour (base level)
    base_z = Z.min() - 0.15 * (Z.max() - Z.min())
    ax3d.scatter(
        samples_prev[:, 0], samples_prev[:, 1], base_z, c="lightgreen", s=5, alpha=0.4
    )
    ax3d.scatter(samples_cur[:, 0], samples_cur[:, 1], base_z, c="pink", s=5, alpha=0.4)

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Potential")
    ax3d.set_title(f"3D View - Energy = {current_energy:.6f}")
    ax3d.legend()

    # 2D contour view (similar to original)
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d = potential.linear.plot_function(fig=fig, ax=ax2d, x_bds=x_bounds, y_bds=y_bounds)
    ax2d.scatter(
        samples_prev[:, 0],
        samples_prev[:, 1],
        color="green",
        s=5,
        alpha=0.6,
        label=f"Iteration {iteration-iteration_diff}",
    )
    ax2d.scatter(
        samples_cur[:, 0],
        samples_cur[:, 1],
        color="red",
        s=5,
        alpha=0.8,
        label=f"Iteration {iteration}",
    )
    ax2d.set_title(f"Contour View - Iteration {iteration}")
    ax2d.legend()
    return fig
