import jax
import jax.numpy as jnp

from operator import add


def _params_scalar_product(param_tree_a, param_tree_b):
    return jax.tree.reduce_associative(
        add,
        jax.tree.map(
            lambda a, b: jnp.dot(a.ravel(), b.ravel()),
            param_tree_a,
            param_tree_a,
        ),
    )
