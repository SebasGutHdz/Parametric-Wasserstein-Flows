import jax
import jax.numpy as jnp

from operator import add


def _params_scalar_product(param_tree_a, param_tree_b):
    """
    Compute scalar product between two parameter trees.

    Note: The original implementation had a bug (used param_tree_a twice).
    This is now fixed to properly compute the dot product between a and b.
    """
    # Use sum over tree leaves instead of deprecated reduce_associative
    return sum(
        jax.tree.leaves(
            jax.tree.map(
                lambda a, b: jnp.dot(a.ravel(), b.ravel()),
                param_tree_a,
                param_tree_b,  # Fixed: was param_tree_a
            )
        )
    )
