"""
In this file we include the basic architectures for the neural network.
"""

import flax.nnx as nnx
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from jax._src import api
from jax.nn.initializers import xavier_uniform, normal, xavier_normal

# initializer = jax.nn.initializers.xavier_uniform()


# Define SinTu activation fn
@api.jit
def SinTu(x: ArrayLike) -> Array:

    return jnp.sin(jnp.maximum(x, 0.0))


@api.jit
def identity(x: ArrayLike) -> Array:
    return x


# Function from string to activation function


def str_to_act_fn(name: str) -> Callable:
    if name == "relu":
        return nnx.relu
    elif name == "sigmoid":
        return nnx.sigmoid
    elif name == "tanh":
        return nnx.tanh
    elif name == "SinTu":
        return SinTu
    elif name == "identity":
        return identity
    elif name == "gelu":
        return nnx.gelu
    elif name == "swish":
        return nnx.swish
    else:
        raise ValueError(f"Unknown activation function: {name}")


# MLP
class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        num_layers: int,
        width_layers: int,
        dout: int,
        activation_fn: str,
        rngs: nnx.Rngs,
    ):

        activation_fn = str_to_act_fn(activation_fn)

        layers = []

        in_dim = din

        # hidden layers
        for _ in range(num_layers):
            layers.append(
                nnx.Linear(
                    in_dim,
                    width_layers,
                    rngs=rngs,
                    kernel_init=xavier_uniform(),
                    bias_init=normal(stddev=1e-3),
                )
            )  # ,  bias_init = normal(stddev=1e-3)
            layers.append(activation_fn)
            in_dim = width_layers

        # output layer (no activation)
        layers.append(
            nnx.Linear(
                in_dim,
                dout,
                rngs=rngs,
                kernel_init=xavier_uniform(),
                bias_init=normal(stddev=1e-3),
            )
        )

        self.layers = layers

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class ResBlock(nnx.Module):
    def __init__(
        self, din: int, width_layers: int, dout: int, activation_fn: str, rngs: nnx.Rngs
    ):

        self.din = din
        self.dout = dout

        activation_fn = str_to_act_fn(activation_fn)

        self.layer1 = nnx.Linear(
            din,
            width_layers,
            rngs=rngs,
            kernel_init=xavier_uniform(),
            bias_init=normal(stddev=1e-3),
        )
        self.activation = activation_fn
        self.layer2 = nnx.Linear(
            width_layers,
            dout,
            rngs=rngs,
            kernel_init=xavier_uniform(),
            bias_init=normal(stddev=1e-3),
        )
        if din != dout:
            self.shortcut = nnx.Linear(
                din,
                dout,
                rngs=rngs,
                kernel_init=xavier_uniform(),
                bias_init=normal(stddev=1e-3),
            )
        else:
            self.shortcut = None

    def __call__(self, x: Array) -> Array:
        identity = x

        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        return out


class ResNet(nnx.Module):
    def __init__(
        self,
        din: int,
        num_layers: int,
        width_layers: int,
        dout: int,
        activation_fn: str,
        rngs: nnx.Rngs,
    ):

        # activation_fn = str_to_act_fn(activation_fn)

        layers = []

        in_dim = din

        for _ in range(num_layers):

            layers.append(ResBlock(in_dim, width_layers, in_dim, activation_fn, rngs))
            # in_dim = width_layers
            # Q: how to use a latent intermediary space?

        # output layer (no activation)
        layers.append(ResBlock(in_dim, width_layers, dout, activation_fn, rngs))

        self.layers = layers

    def __call__(self, x: Array) -> Array:

        for layer in self.layers:

            x = layer(x)

        return x
