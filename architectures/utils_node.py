from flax import nnx
import jax.numpy as jnp
from core.types import TimeArray, SampleArray, VelocityArray
import jax


def eval_model(
    model: nnx.Module, t: TimeArray, x: SampleArray, time_dependent: bool = False
) -> VelocityArray:
    """
    Evaluate the velocity field model with proper time conditioning.

    Args:
        model: The neural network model
        t: Time values, float or jnp.array shape (batch_size,) or (batch_size,1)
        x: Sample positions, shape (batch_size, dim)
        batch_size: Batch size for verification

    Returns:
        Predicted velocities, shape (batch_size, dim)
    """
    if not time_dependent:
        model_input = x
    else:
        if t.ndim == 0:  # element from jnp.array
            t_expanded = jnp.full((x.shape[0], 1), t)
        elif t.ndim == 1:  # Batch of times with format (bs,)
            t_expanded = t.reshape(-1, 1)
        elif t.ndim == 2:  # Batch of times with correct format
            t_expanded = t
        else:
            raise ValueError(
                "t does not have the right shape, valid float of jnp with shapes (bs,) and (bs,1)"
            )

        model_input = jnp.concatenate([t_expanded, x], axis=-1)

    v_pred = model(model_input)
    return v_pred
