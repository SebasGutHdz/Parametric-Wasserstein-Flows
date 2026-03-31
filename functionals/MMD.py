import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union, Generator
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from flax import nnx
import jax

from architectures.node import NeuralODE
from functionals.linear_funcitonal_class import LinearPotential as LinearFunctional
from functionals.internal_functional_class import (
    InternalPotential as InternalFunctional,
)
from functionals.interaction_functional_class import (
    InteractionPotential as InteractionFunctional,
)
from parametric_model.parametric_model import ParametricModel


def bandwidth_median(X: jnp.array) -> float:
    """Estimate a suitable bandwidth for the kernel using the median heuristic.

    Args:
        X: array of shape `(N_samples, dim)`, representing the sample

    Returns:
        float: the bandwidth
    """
    N, d = X.shape
    X_diffs = X[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]
    idx = jnp.triu_indices(N, k=1)
    X_diffs = X_diffs[*idx, :]
    pairwise_sq_dists = (X_diffs**2).sum(axis=-1)
    H = jnp.median(pairwise_sq_dists)
    h = jnp.sqrt(0.5 * H / jnp.log(d + 1))

    return h


def gaussian_kernel(x1, x2, bw):
    return jnp.exp(-0.5 * ((x1 - x2) ** 2).sum() / bw)


gk = jnp.vectorize(gaussian_kernel, signature="(k),(k),()->()")


def gaussian_mmd(X1: jnp.ndarray, X2: jnp.ndarray, bandwidths: jnp.ndarray):
    X1 = X1.reshape(X1.shape[0], -1)
    X2 = X2.reshape(X2.shape[0], -1)

    k_x1x1 = gk(
        X1[:, None, None, :], X1[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)
    k_x1x2 = gk(
        X1[:, None, None, :], X2[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)
    k_x2x2 = gk(
        X2[:, None, None, :], X2[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)

    d1 = X1.shape[0]
    d2 = X2.shape[0]
    A = jnp.triu(k_x1x1, k=1).sum() / (d1 * (d1 - 1))
    C = jnp.triu(k_x2x2, k=1).sum() / (d2 * (d2 - 1))
    B = k_x1x2.mean()

    mmd_sq = 2.0 * (A - B + C)

    return jnp.maximum(0.0, mmd_sq) ** 0.5


mmd_jitted = nnx.jit(gaussian_mmd)
mmd_of_model = nnx.jit(
    nnx.value_and_grad(
        lambda m, X_other, z_s, bandwidths: gaussian_mmd(m(z_s), X_other, bandwidths)
    )
)


class MMDEnergy:
    def __init__(self, X_loader: Generator, bandwidths: jnp.ndarray):
        self.X_loader = X_loader
        self.bandwidths = jnp.atleast_1d(bandwidths)

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
        X1 = eval_model(z_samples)
        X2 = next(self.X_loader)

        return mmd_jitted(X1, X2, self.bandwidths), X2, 0.0, 0.0, 0.0

    def compute_energy_gradient(
        self, parametric_model: ParametricModel, z_samples: Array, params: PyTree
    ) -> PyTree:
        eval_model = (
            parametric_model
            if params is None
            else nnx.merge(nnx.split(parametric_model)[0], params)
        )
        X2 = next(self.X_loader)

        val_mmd, grad_mmd = mmd_of_model(eval_model, X2, z_samples, self.bandwidths)

        energy_breakdown = {
            "internal_energy": 0.0,
            "linear_energy": 0.0,
            "interaction_energy": 0.0,
        }

        return grad_mmd, val_mmd, energy_breakdown
