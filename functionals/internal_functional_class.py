import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy.stats as stats
from parametric_model.parametric_model import ParametricModel
from architectures.node import NeuralODE


class InternalPotential:
    """
    A class for handling internal energy functionals F(ρ) = ∫ f(ρ(x))dx
    where f is a user-defined function.
    The current implementation only support the entropy and Fisher information.
    These quantities are computed by solving ODEs, and their computation is done in the node.py file.
    """

    def __init__(
        self,
        functional: Union[str, Array] = "entropy",
        coeff: float = 1.0,
        sigma: Optional[float] = 1.0,
        method: str = "exact",
        prob_dim=None,
    ):
        """
        Initialize InternalPotential with a functional type.

        Args:
            functional: Type of functional, either 'entropy','fisher' or ['entropy','fisher']
            coeff: Coefficient for the functional
        """
        if type(functional) == str:
            functional = [functional]
        for func in functional:
            if func not in ["entropy", "fisher"]:
                raise ValueError(
                    "Unsupported functional type. Choose either 'entropy' or 'fisher'."
                )
            if func == "entropy" and method not in ["exact", "hutchinson"]:
                raise ValueError(
                    "Unsupported method for entropy functional. Choose either 'exact' or 'hutchinson'."
                )
            if func == "fisher" and method not in ["exact", "autodiff"] or coeff <= 0:
                raise ValueError(
                    "Unsupported method for fisher functional. Choose either 'exact' or 'autodiff'. Coefficient must be positive."
                )
        self.functional = functional
        self.coeff = coeff
        self.sigma = sigma
        self.prob_dim = prob_dim
        self.method = method

    def __call__(
        self,
        parametric_model: ParametricModel,
        z_samples: Array,
        z_trajectory: Optional[Array] = None,
        time_steps: Optional[Array] = None,
        params: Optional[PyTree] = None,
    ) -> Array:

        if params is not None:
            grafdef, _ = nnx.split(parametric_model)
            parametric_model = nnx.merge(grafdef, params)
        # Obtain trajectory for computation of internal energy
        if z_trajectory is None or time_steps is None:
            z_trajectory, time_steps = parametric_model(
                z_samples, history=True
            )  # (batch_size, time_steps, dim)
            # For the first call, set the prob_dim
            if self.prob_dim is None:
                self.prob_dim = z_trajectory.shape[-1]
        internal_energy = 0.0
        # Entropy and fisher information are computed in node.py
        if "entropy" in self.functional:
            log_prob_init = stats.multivariate_normal.logpdf(
                z_samples, mean=jnp.zeros(self.prob_dim), cov=jnp.eye(self.prob_dim)
            )
            # Input: t, xt, log_prob_init, method, params, log_trajectory
            entropy = parametric_model.log_likelihood(
                t=time_steps,
                xt=z_trajectory,
                log_prob_init=log_prob_init,
                method=self.method,
                params=params,
                log_trajectory=False,
            )  # (batch_size,)

            internal_energy += jnp.mean(entropy) * self.coeff  # Return the mean entropy

        elif "fisher" in self.functional:
            # Initialize score
            score_init = -z_samples  # Score of standard normal
            # Input: t, xt, score_init, params, log_trajectory
            score = parametric_model.score_function(
                t=time_steps,
                xt=z_trajectory,
                score_init=score_init,
                method=self.method,
                params=params,
                score_trajectory=False,
            )  # (batch_size, dim)
            # Compute fisher information
            fisher_info = (
                self.sigma**4
                / 8
                * jnp.mean(jnp.linalg.norm(score, axis=-1) ** 2, axis=0)
            )  # Return the mean fisher information
            internal_energy += fisher_info
        return internal_energy
