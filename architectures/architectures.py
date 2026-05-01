"""
In this file we include the basic architectures for the neural network.
"""

import flax.nnx as nnx
from typing import Callable, Tuple
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

        layers = nnx.List()

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

        layers = nnx.List

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


class ConcatConv2D(nnx.Module):
    """Convolutional model with time embedding via concatenation similar to one used in FFJORD paper"""

    def __init__(
        self,
        shape_x: Tuple[int],
        n_layers: int = 3,
        dim_hidden: int = 64,
        activation_fn: str = "swish",
        rngs: nnx.Rngs = None,
    ):
        self.shape_x = shape_x
        list_of_shapes = (1,) + (dim_hidden,) * n_layers + (1,)
        self._conv_layers = nnx.List(
            nnx.Conv(
                list_of_shapes[i] + 1,
                list_of_shapes[i + 1],
                kernel_size=3,
                rngs=rngs,
            )
            for i in range(n_layers + 1)
        )
        self._activations = nnx.List(
            (str_to_act_fn(activation_fn),) * n_layers + (identity,)
        )

    def __call__(self, x_in):
        # extract t from input
        t = x_in[:, 0]
        # extract x and add channel dimension
        x = x_in[:, 1:].reshape((-1, *self.shape_x, 1))

        for _layer, _activation in zip(self._conv_layers, self._activations):
            # broadcast t to x shape
            tt = jnp.broadcast_to(t[:, None, None, None], x.shape[:-1] + (1,))
            # concatenate t as another channel
            xtt = jnp.concatenate((x, tt), axis=-1)
            x = _activation(_layer(xtt))

        # flatten x again
        return x.reshape((x_in.shape[0], -1))
