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


def cross_entropy_of_model(
    model: NeuralODE, X, norm_coef=1.0, trace_method="hutchinson"
):
    dim = X.shape[-1]
    Z = model.pull_back(X)
    x_trajectory, timesteps = model(Z, history=True)

    log_prob_init = jsp.stats.multivariate_normal.logpdf(
        Z, mean=jnp.zeros(dim), cov=jnp.eye(dim)
    )
    # Input: t, xt, log_prob_init, method, params, log_trajectory
    log_pdf_model = model.log_likelihood(
        t=timesteps,
        xt=x_trajectory,
        log_prob_init=log_prob_init,
        method=trace_method,
        log_trajectory=False,
    )  # (batch_size,)

    XX = x_trajectory[:, -1, :]

    X_norm_sq = ((X - XX) ** 2).sum(axis=-1)

    return -log_pdf_model.mean() + norm_coef * X_norm_sq.mean()


jitted_cross_entropy = nnx.jit(
    cross_entropy_of_model, static_argnames=["trace_method", "norm_coef"]
)
cross_entropy_value_and_grad = nnx.jit(
    nnx.value_and_grad(cross_entropy_of_model),
    static_argnames=["trace_method", "norm_coef"],
)


class FBCrossEntropyEnergy:
    """Computes the cross-entropy with the target distribution"""

    def __init__(
        self,
        X_loader: Generator,
        norm_coef=0.0,
        trace_method: Literal["exact", "hutchinson"] = "hutchinson",
    ):
        self.X_loader = X_loader
        self.trace_method = trace_method
        self.norm_coef = norm_coef

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

        return (
            jitted_cross_entropy(
                eval_model, X, norm_coef=self.norm_coef, trace_method=self.trace_method
            ),
            X,
            0.0,
            0.0,
            0.0,
        )

    def compute_energy_gradient(
        self, parametric_model: ParametricModel, z_samples: Array, params: PyTree
    ) -> PyTree:
        eval_model = (
            parametric_model
            if params is None
            else nnx.merge(nnx.split(parametric_model)[0], params)
        )
        X = next(self.X_loader)

        val, grad = cross_entropy_value_and_grad(
            eval_model, X, norm_coef=self.norm_coef, trace_method=self.trace_method
        )

        energy_breakdown = {
            "internal_energy": 0.0,
            "linear_energy": 0.0,
            "interaction_energy": 0.0,
        }

        return grad, val, energy_breakdown
