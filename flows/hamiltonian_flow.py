import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from flax import nnx
from jaxtyping import Array,PyTree
from typing import Tuple,Any,Union,Callable,Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import Device

from geometry.G_matrix import G_matrix
from flows.hamiltonian_flow_step import hamiltonian_flow_step
from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel

from tqdm import tqdm


# Function to initialize momentum 
def initialize_momentum(parametric_model: ParametricModel, z_samples: Array, phi_fn: Callable[[float,Array],Array], params: Optional[PyTree] = None) -> PyTree:
    '''
    Initialize momentum p_0 = 1/N sum_{i=1}^N \nabla_{theta} phi(0,T(z_i,\theta_0)) where z_i are samples from the reference density and T is the flow map defined by parametric_model
    Args:
        parametric_model: ParametricModel instance
        z_samples: (Bs,d) Samples from reference density
        phi_fn: Function phi(t,x): R \times R^d \to R that takes as input a time t and a batch of samples x and returns the evaluation of phi at those points
        G_mat: G-matrix object for metric tensor computations
        params: PyTree where the G matrix is computed at. If None, use current parameters of parametric_model
    Return:
        p_0: PyTree with same GraphDef as parametric_model
    '''
    if params is None:
        _, params = nnx.split(parametric_model)

    # Step 1: Compute \nabla_{theta} phi(0,T(z,\theta_0)) using vmap and grad
    def objective(p: PyTree) -> float:
        # Push samples through the flow map
        x = parametric_model(z_samples, params=p)
        # Evaluate phi at the pushed samples
        return jnp.mean(phi_fn(0.0, x))
    p_0 = jax.grad(objective)(params)
    return p_0
def compute_hamiltonian(theta: PyTree, p: PyTree, z_samples: Array, 
                       G_mat: G_matrix, potential: Potential) -> float:
    """
    Compute Hamiltonian H = (1/2)p^T G^(-1) p + F(theta)
    
    Args:
        theta: Current parameters
        p: Current momentum
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object
        potential: Potential instance
        
    Returns:
        hamiltonian: Scalar value H(theta, p)
    """
    # Kinetic energy: (1/2)p^T G^(-1) p
    # Solve G h = p to get h = G^(-1) p
    h, _ = G_mat.solve_system(z_samples, p, params=theta,regularization=1e-4)
    
    # Compute inner product p^T h = p^T G^(-1) p
    kinetic = 0.5 * sum(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(a * b), p, h)))
    

    # Potential energy: F(theta)
    # Need to create a temporary parametric_model to evaluate potential
    graphdef, _ = nnx.split(G_mat.mapping)
    temp_parametric_model = nnx.merge(graphdef, theta)
    potential_energy, _, _, _, _ = potential.evaluate_energy(temp_parametric_model, z_samples, theta)
    

    # Total Hamiltonian
    hamiltonian = kinetic + potential_energy

    return hamiltonian
def run_hamiltonian_flow(parametric_model: nnx.Module, 
                        batch_size: int,
                        test_data_set: Array, 
                        G_mat: G_matrix,
                        potential: Potential, 
                        phi_fn: Callable[[float, Array], Array],
                        h: float = 0.01, 
                        solver: str = "cg",
                        max_iterations: int = 100, 
                        tolerance: float = 1e-6,
                        regularization: float = 1e-6,
                        progress_every: int = 10,
                        # Hamiltonian-specific parameters
                        gamma: float = 1e-2,
                        n_iters: int = 3,
                        solver_tol: float = 1e-6,
                        solver_maxiter: int = 50) -> dict:
    """
    Run complete Hamiltonian flow integration using symplectic Euler method.
    
    Args:
        parametric_model: Initial Neural ODE model
        z_samples: Reference samples for Monte Carlo estimation
        G_mat: G-matrix object for linear system solving
        potential: Potential instance defining the energy functional
        phi_fn: Dual potential function for momentum initialization
        h: Time step size
        solver: Linear solver type ("minres", "cg")
        max_iterations: Maximum number of Hamiltonian flow steps
        tolerance: Convergence tolerance for Hamiltonian drift
        regularization: Regularization parameter for linear solvers
        progress_every: Print progress every N iterations
        gamma: Step size for fixed point iteration
        n_iters: Number of fixed point iterations per step
        solver_tol: Tolerance for linear solver
        solver_maxiter: Maximum iterations for linear solver
        
    Returns:
        results: Dictionary containing trajectories, conservation info, etc.
    """
    # Initialize key for sample generation
    key = jax.random.PRNGKey(0)
    # Generate initial batch of reference samples
    key, subkey = jax.random.split(key)
    problem_dim = test_data_set.shape[1]
    #TODO: take as input the sampling function of the reference density
    z_samples = jax.random.normal(subkey, (batch_size, problem_dim))

    # Initialize momentum using phi function
    current_parametric_model = parametric_model
    current_momentum = initialize_momentum(current_parametric_model, z_samples, phi_fn)

    # Initialize tracking
    energy_history = []
    hamiltonian_history = []
    momentum_norms = []
    param_norms = []
    sample_history = []
    

    # Compute initial Hamiltonian
    _, initial_params = nnx.split(current_parametric_model)
    initial_hamiltonian = compute_hamiltonian(
        initial_params, current_momentum, test_data_set, G_mat, potential
    )
    hamiltonian_history.append(float(initial_hamiltonian))
    
    print(f"Initial Hamiltonian: {initial_hamiltonian:.6f}")


    
    p_bar = tqdm(range(max_iterations-1), desc="Hamiltonian Flow Progress")

    for iteration in p_bar:
        
        # Get initial samples for visualization (first iteration only)
        if iteration == 0:
            _, samples0, _, _, _ = potential.evaluate_energy(current_parametric_model, test_data_set)

        # Generate fresh samples for this iteration
        #TODO: Use the sampling function of the reference density
        key, subkey = jax.random.split(key)
        z_samples_eval = jax.random.normal(subkey, (batch_size, problem_dim))
        _,prev_parms = nnx.split(current_parametric_model)
        
        current_parametric_model, current_momentum, step_info = hamiltonian_flow_step(
            parametric_model=current_parametric_model, 
            p_n=current_momentum, 
            z_samples=z_samples_eval, 
            G_mat=G_mat, 
            potential=potential,
            step_size=h, 
            solver=solver,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
            regularization=regularization,
            gamma=gamma,
            n_iters=n_iters
        )
        
        # Compute current Hamiltonian for conservation tracking
        _, current_params = nnx.split(current_parametric_model)
        
        
        # Compute norms for diagnostics
        momentum_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), current_momentum))))
        param_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), current_params))))
        param_increment = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum((a - b)**2), current_params, prev_parms))))
        
        # Store diagnostics
        energy_history.append(float(step_info['energy']))
        
        momentum_norms.append(float(momentum_norm))
        param_norms.append(float(param_norm))
        
        # Compute Hamiltonian drift
        
        
        # Update progress bar
        p_bar.set_postfix({
            'Energy': f"{step_info['energy']:.6f}",
            'Linear': f"{step_info['linear_energy']:.6f}",
            'Internal': f"{step_info['internal_energy']:.6f}",
            'Interaction': f"{step_info['interaction_energy']:.6f}",
            '||p||': f"{momentum_norm:.6f}",
            '||delta theta||': f"{param_increment:.6f}",
        })
        
        # Detailed progress reporting
        if (iteration % progress_every == 0 and iteration > 0) or iteration == max_iterations - 2:
            current_energy, samples1, _, _, _ = potential.evaluate_energy(current_parametric_model, test_data_set, current_params)
            sample_history.append(samples1)
            current_hamiltonian = compute_hamiltonian(
                current_params, current_momentum, test_data_set, G_mat, potential
            )
            hamiltonian_history.append(float(current_hamiltonian))
            hamiltonian_drift = float(current_hamiltonian - initial_hamiltonian)
            average_displacement = jnp.mean(jnp.linalg.norm(samples1 - samples0, axis=1))
            
            print(f"Iter {iteration:3d}: Energy = {step_info['energy']:.6f}, "
                    f"Hamiltonian = {current_hamiltonian:.6f}, "
                    f"H_drift = {hamiltonian_drift:.2e}, "
                    f"Avg_Displacement = {average_displacement:.2e}")

            # Create visualization
            fig = plt.figure(figsize=(15, 5))

            # 3D view (same as gradient flow)
            ax3d = fig.add_subplot(1, 3, 1, projection='3d')

            # Get potential surface
            all_samples = jnp.vstack([samples0, samples1])
            x_range = jnp.linspace(all_samples[:,0].min() - 0.5, all_samples[:,0].max() + 0.5, 100)
            y_range = jnp.linspace(all_samples[:,1].min() - 0.5, all_samples[:,1].max() + 0.5, 100)
            X, Y = jnp.meshgrid(x_range, y_range)
            Z = potential.linear.potential_fn(jnp.stack([X.ravel(), Y.ravel()], axis=-1), 
                                            **potential.linear.potential_kwargs).reshape(X.shape)

            # Plot surface
            ax3d.plot_surface(X, Y, Z, alpha=0.4, cmap='viridis')

            # Particles on surface
            surface_z0 = potential.linear.potential_fn(samples0, **potential.linear.potential_kwargs)
            surface_z1 = potential.linear.potential_fn(samples1, **potential.linear.potential_kwargs)

            ax3d.scatter(samples0[:,0], samples0[:,1], surface_z0, 
                        c='green', s=10, alpha=0.6, label=f'Iteration {max(0, iteration-progress_every)}')
            ax3d.scatter(samples1[:,0], samples1[:,1], surface_z1, 
                        c='red', s=10, alpha=0.8, label=f'Iteration {iteration}')

            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Potential')
            ax3d.set_title(f"3D View - Energy = {current_energy:.6f}")
            ax3d.legend()

            # 2D contour view
            ax2d = fig.add_subplot(1, 3, 2)
            ax2d = potential.linear.plot_function(fig=fig, ax=ax2d)
            ax2d.scatter(samples0[:,0], samples0[:,1], color='green', s=5, alpha=0.6, 
                        label=f'Iteration {max(0, iteration-progress_every)}')
            ax2d.scatter(samples1[:,0], samples1[:,1], color='red', s=5, alpha=0.8, 
                        label=f'Iteration {iteration}')
            ax2d.set_title(f"Contour View -time step h={h*(iteration+1):.3f}")
            ax2d.legend()

            # Hamiltonian conservation plot
            ax_cons = fig.add_subplot(1, 3, 3)
            # iterations_so_far = list(range(len(hamiltonian_history)))*h
            iterations_so_far = jnp.linspace(0,len(hamiltonian_history)-1,len(hamiltonian_history))*h
            ax_cons.plot(iterations_so_far, hamiltonian_history, 'b-', label='Hamiltonian')
            ax_cons.axhline(y=initial_hamiltonian, color='r', linestyle='--', alpha=0.5, 
                            label=f'Initial H = {initial_hamiltonian:.4f}')
            ax_cons.set_xlabel('Time Step')
            ax_cons.set_ylabel('Hamiltonian')
            ax_cons.set_title('Hamiltonian Conservation')
            ax_cons.legend()
            ax_cons.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            
            # Update samples for next comparison
            samples0 = samples1
            
        # # Early stopping conditions
        # if hamiltonian_drift > 10 * tolerance:  # Large drift indicates instability
        #     print(f"Warning: Large Hamiltonian drift {hamiltonian_drift:.2e} at iteration {iteration}")
        
        # if iteration > 5 and hamiltonian_drift < tolerance * 1e-3:
        #     print(f"Excellent Hamiltonian conservation at iteration {iteration}")
                
    # Final summary
    final_energy = energy_history[-1]
    final_hamiltonian = hamiltonian_history[-1]
    total_drift = abs(final_hamiltonian - initial_hamiltonian)
    
    print(f"\n=== Hamiltonian Integration Complete ===")
    print(f"Total iterations:         {len(energy_history)}")
    print(f"Initial Hamiltonian:      {initial_hamiltonian:.6f}")
    print(f"Final Hamiltonian:        {final_hamiltonian:.6f}")
    print(f"Total Hamiltonian drift:  {total_drift:.2e}")
    print(f"Relative drift:           {total_drift/abs(initial_hamiltonian):.2e}")
    print(f"Final energy:             {final_energy:.6f}")
    print(f"Final momentum norm:      {momentum_norms[-1]:.6f}")
    print(f"Final param norm:         {param_norms[-1]:.6f}")
    
    return {
        'final_parametric_model': current_parametric_model,
        'final_momentum': current_momentum,
        'energy_history': energy_history,
        'hamiltonian_history': hamiltonian_history,
        'momentum_norms': momentum_norms,
        'param_norms': param_norms,
        'sample_history': sample_history,
        'phi_function': phi_fn,
        'potential': potential,
        'convergence_info': {
            'hamiltonian_conserved': total_drift < tolerance,
            'initial_hamiltonian': initial_hamiltonian,
            'final_hamiltonian': final_hamiltonian,
            'hamiltonian_drift': total_drift,
            'relative_drift': total_drift/abs(initial_hamiltonian) if initial_hamiltonian != 0 else float('inf'),
            'final_energy': final_energy,
            'iterations': len(energy_history)
        }
    }

