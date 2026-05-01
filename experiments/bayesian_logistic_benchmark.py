from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from datasets.bayesian_dataset import (
    BinaryDatasetName,
    load_binary_benchmark_split,
    BinaryBenchmarkSplit,
)
from functionals.functional import Potential
from functionals.linear_funcitonal_class import LinearPotential


@dataclass
class BayesianLogisticData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    dataset_name: str


@dataclass
class BayesianLogisticSetup:
    data: BinaryBenchmarkSplit
    X_train_jax: jnp.ndarray
    y_train_jax: jnp.ndarray
    X_test_jax: jnp.ndarray
    y_test_jax: jnp.ndarray
    potential_fn: callable


def prepare_binary_benchmark_dataset(
    name: BinaryDatasetName,
    data_path: str = "data",
    test_ratio: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
    allow_unverified_ssl: bool = False,
) -> BayesianLogisticData:
    """
    Load and preprocess the binary datasets used in variational WGF experiments.
    """
    split = load_binary_benchmark_split(
        name=name,
        data_path=data_path,
        test_ratio=test_ratio,
        seed=seed,
        standardize=standardize,
        allow_unverified_ssl=allow_unverified_ssl,
    )
    return BayesianLogisticData(
        X_train=np.asarray(split.X_train, dtype=np.float64),
        y_train=np.asarray(split.y_train, dtype=np.int32),
        X_test=np.asarray(split.X_test, dtype=np.float64),
        y_test=np.asarray(split.y_test, dtype=np.int32),
        n_features=split.n_features,
        dataset_name=name,
    )


def build_bayesian_logistic_setup(
    dataset: BinaryBenchmarkSplit,
    alpha_prior_shape: float = 1.0,
    alpha_prior_rate: float = 0.01,
) -> BayesianLogisticSetup:
    """
    Build data + JAX arrays + potential callable for Bayesian logistic experiments.
    """
    X_train_jax = jnp.asarray(dataset.X_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(dataset.y_train, dtype=jnp.float32)
    X_test_jax = jnp.asarray(dataset.X_test, dtype=jnp.float32)
    y_test_jax = jnp.asarray(dataset.y_test, dtype=jnp.float32)

    y_unique = set(np.unique(np.asarray(dataset.y_train).reshape(-1)).tolist())
    if not y_unique.issubset({-1.0, 1.0, -1, 1}):
        raise ValueError(
            f"Expected +/-1 labels for Bayesian logistic potential, got {sorted(y_unique)}"
        )

    def potential_fn(theta_batch: jnp.ndarray) -> jnp.ndarray:
        return bayesian_logistic_potential(
            theta_batch,
            X=X_train_jax,
            y_pm1=y_train_jax,
            alpha_prior_shape=alpha_prior_shape,
            alpha_prior_rate=alpha_prior_rate,
        )

    return BayesianLogisticSetup(
        data=dataset,
        X_train_jax=X_train_jax,
        y_train_jax=y_train_jax,
        X_test_jax=X_test_jax,
        y_test_jax=y_test_jax,
        potential_fn=potential_fn,
    )


def build_linear_potential_for_bayes_logistic(
    setup: BayesianLogisticSetup, coeff: float = 1.0
) -> LinearPotential:
    """Create a LinearPotential compatible with the existing flow code."""
    return LinearPotential(potential_fn=setup.potential_fn, coeff=coeff)


def build_potential_for_bayes_logistic(
    setup: BayesianLogisticSetup,
    *,
    linear_coeff: float = 1.0,
    internal=None,
    interaction=None,
) -> Potential:
    """
    Convenience wrapper to build Potential(linear + optional internal/interaction).
    """
    linear = build_linear_potential_for_bayes_logistic(setup, coeff=linear_coeff)
    return Potential(linear=linear, internal=internal, interaction=interaction)


def bayesian_logistic_potential(
    theta_batch: jnp.ndarray,
    X: jnp.ndarray,
    y_pm1: jnp.ndarray,
    alpha_prior_shape: float = 1.0,
    alpha_prior_rate: float = 0.01,
) -> jnp.ndarray:
    """
    Negative log unnormalized posterior for Bayesian logistic regression.

    Parameterization follows the VWGF setup:
    theta = [w, log(alpha)], with w in R^d and alpha > 0.
    Priors:
      p(w | alpha) = N(0, alpha^{-1} I)
      p(alpha) = Gamma(alpha | shape, rate)
    """
    if theta_batch.ndim != 2:
        raise ValueError("theta_batch must have shape (batch, d + 1)")
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, d)")
    if y_pm1.ndim != 1:
        raise ValueError("y_pm1 must have shape (n_samples,)")
    d = X.shape[1]
    if theta_batch.shape[1] != d + 1:
        raise ValueError(
            f"Expected theta dimension {d + 1}, got {theta_batch.shape[1]}"
        )
    if y_pm1.shape[0] != X.shape[0]:
        raise ValueError("X and y_pm1 must have the same number of samples")

    w = theta_batch[:, :-1]
    log_alpha = theta_batch[:, -1]
    alpha = jnp.exp(log_alpha)

    logits = X @ w.T
    signed_logits = y_pm1[:, None] * logits
    nll = jnp.sum(jnp.logaddexp(0.0, -signed_logits), axis=0)

    w_sq = jnp.sum(w**2, axis=1)
    gaussian_quad = 0.5 * alpha * w_sq
    gamma_rate_term = alpha_prior_rate * alpha
    log_alpha_coeff = alpha_prior_shape - 1.0 + 0.5 * d
    log_alpha_term = -log_alpha_coeff * log_alpha

    return nll + gaussian_quad + gamma_rate_term + log_alpha_term


def posterior_predictive_probability(
    theta_batch: jnp.ndarray, X: jnp.ndarray
) -> jnp.ndarray:
    """
    Mean Bernoulli probability over posterior particles.
    """
    w = theta_batch[:, :-1]
    logits = X @ w.T
    probs = jax_sigmoid(logits)
    return jnp.mean(probs, axis=1)


def classification_accuracy_from_particles(
    theta_batch: jnp.ndarray, X: jnp.ndarray, y_pm1: jnp.ndarray
) -> jnp.ndarray:
    probs = posterior_predictive_probability(theta_batch, X)
    y_hat_pm1 = jnp.where(probs >= 0.5, 1.0, -1.0)
    return jnp.mean((y_hat_pm1 == y_pm1).astype(jnp.float32))


def predictive_log_likelihood_from_particles(
    theta_batch: jnp.ndarray,
    X: jnp.ndarray,
    y_pm1: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Mean predictive log-likelihood under posterior particle averaging.

    For labels y in {-1,+1}, uses:
      p(y=+1|x) = p
      p(y=-1|x) = 1-p
    where p is the posterior predictive Bernoulli mean probability.
    """
    probs = posterior_predictive_probability(theta_batch, X)
    probs = jnp.clip(probs, eps, 1.0 - eps)
    y01 = 0.5 * (y_pm1 + 1.0)
    return jnp.mean(y01 * jnp.log(probs) + (1.0 - y01) * jnp.log(1.0 - probs))


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
