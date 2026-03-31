import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union, Generator, Literal
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from flax import nnx
import jax
import jax.scipy as jsp

from architectures.node import NeuralODE
from parametric_model.parametric_model import ParametricModel

def cross_entropy_of_model(model: NeuralODE, X, trace_method='hutchinson'):
    dim = X.shape[-1]
    z_trajectory, timesteps = model.pull_back(X, history=True)

 ## reverse time dimension again to evaluate log density
    z_trajectory = z_trajectory[:, ::-1, :]
    timesteps = timesteps[::-1]

    Z = z_trajectory[:, 0, :]

    log_prob_init = jsp.stats.multivariate_normal.logpdf(Z, mean=jnp.zeros(dim), cov=jnp.eye(dim))
    # Input: t, xt, log_prob_init, method, params, log_trajectory
    log_pdf_model = model.log_likelihood(
        t=timesteps,
        xt=z_trajectory,
        log_prob_init=log_prob_init,
        method=trace_method,
        log_trajectory=False,
    )  # (batch_size,)

    
    return -log_pdf_model.mean()

jitted_cross_entropy = nnx.jit(cross_entropy_of_model, static_argnames=["trace_method"])
cross_entropy_value_and_grad = nnx.jit(nnx.value_and_grad(cross_entropy_of_model), static_argnames=["trace_method"])


class CrossEntropyEnergy:
    """Computes the cross-entropy with the target distribution """
    def __init__(self, X_loader: Generator, trace_method: Literal['exact', 'hutchinson']='hutchinson'):
        self.X_loader = X_loader
        self.trace_method= trace_method

    def evaluate_energy(
        self,
        parametric_model: ParametricModel,
        z_samples: Array,
        params: Optional[PyTree] = None,
    ) -> float:
        eval_model = (
            parametric_model
            if params is None
            else nnx.merge(nnx.split(parametric_model)[0], params)
        )
        X = next(self.X_loader)

        return jitted_cross_entropy(eval_model, X, trace_method=self.trace_method), X, 0., 0., 0.



    def compute_energy_gradient(
        self, parametric_model: ParametricModel, z_samples: Array, params: PyTree
    ) -> PyTree:
        eval_model = (
            parametric_model
            if params is None
            else nnx.merge(nnx.split(parametric_model)[0], params)
        )
        X = next(self.X_loader)

        val, grad= cross_entropy_value_and_grad(eval_model, X, trace_method=self.trace_method)

        energy_breakdown = {
            "internal_energy": 0.0,
            "linear_energy": 0.0,
            "interaction_energy": 0.0,
        }

        return grad, val, energy_breakdown
