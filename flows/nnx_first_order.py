import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxtyping import Array
from tqdm.auto import tqdm

from functionals.functional import Potential
from parametric_model.parametric_model import ParametricModel


def _run_first_order_optimizer(
    method: str,
    parametric_model: ParametricModel,
    batch_size: int,
    test_data_set: Array,
    potential: Potential,
    n_iterations: int,
    learning_rate: float,
    convergence_tol: float = 1e-6,
    progress_every: int = 10,
    verbose: bool = False,
    use_tqdm: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    diagnostic_sample_size: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> dict[str, Any]:
    if "learning_rate" in optimizer_kwargs:
        raise ValueError(
            "Use 'stepsize' in benchmark config; it is mapped to learning_rate"
        )

    if method == "sgd":
        tx = optax.sgd(learning_rate=learning_rate, **optimizer_kwargs)
    elif method == "adam":
        tx = optax.adam(learning_rate=learning_rate, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported first-order method: {method}")

    optimizer = nnx.Optimizer(parametric_model, tx, wrt=nnx.Param)

    dim = int(test_data_set.shape[1])
    key = jax.random.PRNGKey(0)

    energy_history: list[float] = []
    euclidean_grad_history: list[float] = []

    energy_init, _, _, _, _ = potential.evaluate_energy(parametric_model, test_data_set)
    energy_history.append(float(energy_init))

    iterator = tqdm(range(n_iterations), desc=f"{method.upper()} Progress")
    if not use_tqdm:
        iterator = range(n_iterations)

    for iteration in iterator:
        key, train_key = jax.random.split(key)
        z_samples = jax.random.normal(train_key, (batch_size, dim))

        _, params = nnx.split(parametric_model)
        grad, _, _ = potential.compute_energy_gradient(parametric_model, z_samples, params)
        optimizer.update(parametric_model, grad)

        grad_norm = jnp.sqrt(
            sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), grad)))
        )
        euclidean_grad_history.append(float(grad_norm))

        _, params = nnx.split(parametric_model)
        energy, _, _, _, _ = potential.evaluate_energy(
            parametric_model, test_data_set, params
        )
        energy_history.append(float(energy))

        if progress_callback is not None:
            scatter_samples = None
            if diagnostic_sample_size is not None and diagnostic_sample_size > 0:
                key, diag_key = jax.random.split(key)
                z_diag = jax.random.normal(diag_key, (diagnostic_sample_size, dim))
                scatter_samples = parametric_model(z_diag, params=params)

            progress_callback(
                {
                    "method": method,
                    "iteration": iteration,
                    "max_iterations": n_iterations,
                    "energy": float(energy),
                    "euclidean_grad_norm": float(grad_norm),
                    "scatter_samples": scatter_samples,
                    "converged": bool(grad_norm < convergence_tol),
                }
            )

        if verbose and (iteration % progress_every == 0 or iteration < 5):
            print(
                f"Iter {iteration:4d} | "
                f"Energy: {float(energy):12.6e} | "
                f"Euclidean grad: {float(grad_norm):12.6e}"
            )

        if grad_norm < convergence_tol:
            break

    return {
        "final_parametric_model": parametric_model,
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "euclidean_grad_history": np.asarray(euclidean_grad_history, dtype=np.float64),
        "convergence_info": {
            "converged": bool(
                euclidean_grad_history and euclidean_grad_history[-1] < convergence_tol
            ),
            "iterations": len(euclidean_grad_history),
        },
    }


def run_sgd(
    parametric_model: ParametricModel,
    batch_size: int,
    test_data_set: Array,
    potential: Potential,
    n_iterations: int = 100,
    learning_rate: float = 1e-3,
    convergence_tol: float = 1e-6,
    progress_every: int = 10,
    verbose: bool = False,
    use_tqdm: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    diagnostic_sample_size: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> dict[str, Any]:
    return _run_first_order_optimizer(
        method="sgd",
        parametric_model=parametric_model,
        batch_size=batch_size,
        test_data_set=test_data_set,
        potential=potential,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        progress_every=progress_every,
        verbose=verbose,
        use_tqdm=use_tqdm,
        progress_callback=progress_callback,
        diagnostic_sample_size=diagnostic_sample_size,
        **optimizer_kwargs,
    )


def run_adam(
    parametric_model: ParametricModel,
    batch_size: int,
    test_data_set: Array,
    potential: Potential,
    n_iterations: int = 100,
    learning_rate: float = 1e-3,
    convergence_tol: float = 1e-6,
    progress_every: int = 10,
    verbose: bool = False,
    use_tqdm: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    diagnostic_sample_size: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> dict[str, Any]:
    return _run_first_order_optimizer(
        method="adam",
        parametric_model=parametric_model,
        batch_size=batch_size,
        test_data_set=test_data_set,
        potential=potential,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        progress_every=progress_every,
        verbose=verbose,
        use_tqdm=use_tqdm,
        progress_callback=progress_callback,
        diagnostic_sample_size=diagnostic_sample_size,
        **optimizer_kwargs,
    )
