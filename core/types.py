from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Protocol,
    NamedTuple,
)
from typing_extensions import TypeAlias
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PyTree
from flax import nnx
from dataclasses import dataclass


PRNGKeyArray: TypeAlias = jax.Array
"""JAX PRNG key array."""

PyTreeParams: TypeAlias = PyTree
"""PyTree structure containing model parameters."""


# Sample and trajectory arrays
SampleArray: TypeAlias = Float[Array, "batch_size dim"]
""" Array of shape (batch_size, dim) representing samples."""

TrajectoryArray: TypeAlias = Float[Array, "batch_size time_steps dim"]
""" Array of shape (batch_size, time_steps, dim) representing trajectories."""

# Time-related arrays
TimeArray: TypeAlias = Float[Array, "batch_size"]
""" Array of shape (batch_size,) representing time points in [0,1]."""

TimeStepsArray: TypeAlias = Float[Array, "time_steps"]
""" Array of shape (time_steps,) representing discrete time steps for integration."""

# Velocity and dynamics arrays
VelocityArray: TypeAlias = Float[Array, "batch_size dim"]
""" Array of shape (batch_size, dim) representing velocities."""

VelocityFieldArray: TypeAlias = Float[Array, "batch_size time_steps dim"]
""" Array of shape (batch_size, time_steps, dim) representing velocity fields over time."""

# Score and density arrays
ScoreArray: TypeAlias = Float[Array, "batch_size dim"]
"""Array of score function values (∇ log ρ)."""

DensityArray: TypeAlias = Float[Array, "batch_size"]
"""Array of probability density values."""

LogDensityArray: TypeAlias = Float[Array, "batch_size"]
"""Array of log probability density values."""

# Energy arrays
EnergyArray: TypeAlias = Float[Array, "time_steps"]
"""Array of energy values over time."""

# =============================================================================
# Parameter and Model Types
# =============================================================================

ModelParams: TypeAlias = PyTreeParams
"""Parameters for neural network models."""

ModelState: TypeAlias = Optional[nnx.State]
"""Optional state for stateful models (e.g., batch norm)."""
