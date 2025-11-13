import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.random as jrandom
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence, Callable, Optional, Dict, Any
import numpy as np
import platform


class SinActivation(nn.Module):
    """Sine activation function."""

    def __call__(self, x):
        return jnp.sin(x)


class SinTUActivation(nn.Module):
    """Sine Truncated Unit activation function: SinTU_s = sin(max(x, s))"""

    s: float = -jnp.pi  # Default truncation parameter

    def __call__(self, x):
        return jnp.sin(jnp.maximum(x, self.s))


class MMNNLayer(nn.Module):
    """MMNN Layer with JAX-safe fixed weights"""

    d_in: int
    width: int
    d_out: int
    activation: callable = SinTUActivation()
    use_bias: bool = True
    seed: int = 0
    FixWb: bool = True  # Whether to fix weights and biases

    @nn.compact
    def __call__(self, x):
        """Forward pass with JAX-safe parameter handling"""

        def init_W():
            key = jrandom.PRNGKey(self.seed)
            return jrandom.normal(key, (self.width, self.d_in)) * jnp.sqrt(
                2.0 / self.d_in
            )

        def init_b():
            key = jrandom.PRNGKey(self.seed + 1)  # Different seed for b
            return jrandom.normal(key, (self.width,)) * jnp.sqrt(2.0 / self.d_in)

        if self.FixWb:
            # Create non-trainable variables (excluded from parameter updates)
            W = self.variable("fixed", "W", init_W).value
            b = (
                self.variable("fixed", "b", init_b).value
                if self.use_bias
                else jnp.zeros(self.width)
            )
        else:
            # Create trainable parameters
            W = self.param(
                "W", nn.initializers.xavier_normal(), (self.width, self.d_in)
            )
            b = (
                self.param("b", nn.initializers.zeros, (self.width,))
                if self.use_bias
                else jnp.zeros(self.width)
            )

        # Compute fixed transformation
        z = jnp.dot(x, W.T) + b
        activated = self.activation(z)

        # Trainable linear combination
        A = self.param("A", nn.initializers.xavier_normal(), (self.d_out, self.width))
        c = self.param("c", nn.initializers.zeros, (self.d_out,))

        return jnp.dot(activated, A.T) + c


class MMNN(nn.Module):
    """Simple MMNN using JAX-safe layers"""

    ranks: list
    widths: list
    activation: callable = (SinTUActivation(),)
    ResNet: bool = (False,)
    FixWb: bool = True

    @nn.compact
    def __call__(self, x):
        # Build layers based on ranks and widths
        for i, width in enumerate(self.widths):
            if self.ResNet:
                if 0 < i < len(self.widths):
                    x_id = x + 0

            x = MMNNLayer(
                d_in=self.ranks[i] if i == 0 else self.ranks[i],
                width=width,
                d_out=self.ranks[i + 1],
                activation=self.activation,
                seed=i,
                FixWb=self.FixWb,
            )(x)
            if self.ResNet and 0 < i < len(self.widths) - 1:
                n = min(x.shape[1], x_id.shape[1])
                x = x.at[:, :n].add(
                    x_id[:, :n]
                )  # Notation for in-place addition in JAX
        return x


class Train_jax_model:
    """
    This a basic training scheme for a jax model.
    Inputs:
        model: A Flax model to train
        input_data: Input data as a jnp.ndarray
        target_data: Target data as a jnp.ndarray
        optimizer: Optimizer to use, e.g. 'adam', 'sgd'
        loss_fn: Loss function to use, e.g. 'mse', 'mae'
        learning_rate: Learning rate for the optimizer, it can also be a jax scheduler
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        random_seed: Random seed for reproducibility
    Outputs:
        params: Trained model parameters
        training_info: Dictionary containing training and validation losses
    """

    def __init__(
        self,
        model: nn.Module,
        input_data: jnp.ndarray,
        target_data: jnp.ndarray,
        optimizer: str = "adam",
        loss_fn: str = "mse",
        learning_rate: float = 0.001,
        num_epochs: int = 1000,
        batch_size: int = 32,
        random_seed: int = 0,
        device=0,
    ):
        self.model = model
        self.loss_fn = self._create_loss_fn(loss_fn)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.device = (
            jax.devices("gpu")[device] if jax.devices("gpu") else jax.devices("cpu")[0]
        )

        input_data = jax.device_put(input_data, self.device)
        target_data = jax.device_put(target_data, self.device)

        self.n_samples = jnp.shape(input_data)[0]

        self.key = jrandom.PRNGKey(random_seed)
        self.split_train_test(
            input_data, target_data
        )  # Stores self.x_train, self.y_train, self.x_test, self.y_test
        self.n_batches = jnp.shape(self.x_train)[0] // self.batch_size

        # Create optimizer
        if optimizer == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Create training step function
        self.train_step = self._create_train_step()

    # Create loss function
    def _create_loss_fn(self, loss_type):
        if loss_type == "mse":
            return lambda params, x, y: jnp.mean((self.model.apply(params, x) - y) ** 2)
        elif loss_type == "mae":
            return lambda params, x, y: jnp.mean(
                jnp.abs(self.model.apply(params, x) - y)
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    # Process input and target data
    def split_train_test(self, input_data, target_data, test_split=0.2):
        """
        Split input and target data into training and validation sets
        """
        n_test = int(self.n_samples * test_split)
        n_train = self.n_samples - n_test

        # Random indices for shuffling
        indices = jrandom.permutation(self.key, self.n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Split input and target data
        self.x_train = input_data[train_indices]
        self.y_train = target_data[train_indices]
        self.x_test = input_data[test_indices]
        self.y_test = target_data[test_indices]

    # Batch generator
    def batch_generator(self, x_data, y_data):
        """
        Generate batches of data
        Inputs:
            x_data: Input data
            y_data: Target data
        Yields:
            x_batch: Input batch
            y_batch: Target batch
        """
        # The last incomplete batch will be ignored
        for i in range(0, self.n_batches * self.batch_size, self.batch_size):
            x_batch = x_data[i : i + self.batch_size]
            y_batch = y_data[i : i + self.batch_size]
            # yield is used to create a generator
            # This allows us to iterate over batches without loading everything into memory
            yield x_batch, y_batch

    # Define training step function
    def _create_train_step(self):
        """Create a JIT-compiled training step function"""

        @jax.jit
        def train_step(params, opt_state, x_batch, y_batch):
            # Capture self variables in closure
            loss, grads = jax.value_and_grad(self.loss_fn)(params, x_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        return train_step

    def training_loop(self, print_every: int = 100):

        import matplotlib.pyplot as plt

        """
        Training loop
        """

        training_losses = []
        validation_losses = []

        # Initialize model parameters
        sample_input = self.x_train[:1]  # Use first sample for initialization
        params = self.model.init(self.key, sample_input)
        # Move parameters to the device
        params = jax.device_put(params, self.device)
        self.params_store = params
        # Create optimizer state
        opt_state = self.optimizer.init(params)

        for epoch in range(self.num_epochs):

            epoch_loss = []
            self.key, subkey = jrandom.split(self.key)
            perm = jrandom.permutation(subkey, len(self.x_train))
            x_train_shuffled = self.x_train[perm]
            y_train_shuffled = self.y_train[perm]
            # Iterate over batches
            for x_batch, y_batch in self.batch_generator(
                x_train_shuffled, y_train_shuffled
            ):

                # Perform a training step
                params, opt_state, loss = self.train_step(
                    params, opt_state, x_batch, y_batch
                )
                epoch_loss.append(loss)

            self.params_store = params  # Store the latest parameters
            # Average loss for the epoch
            avg_loss = jnp.mean(jnp.array(epoch_loss))
            training_losses.append(avg_loss)

            if print_every < jnp.inf and (
                epoch % print_every == 0 or epoch == self.num_epochs - 1
            ):
                # Compute validation loss
                val_loss = self.loss_fn(params, self.x_test, self.y_test)
                validation_losses.append(val_loss)
                # Plot the model predictions
                idx_sort = jnp.argsort(
                    self.x_test, axis=0
                )  # Sort for consistent plotting
                x_test_local = self.x_test[idx_sort]
                y_test_local = self.y_test[idx_sort]
                predictions = self.model.apply(params, x_test_local)
                plt.figure(figsize=(10, 5))
                plt.plot(
                    x_test_local.flatten(),
                    y_test_local.flatten(),
                    label="True Function",
                    color="blue",
                )
                plt.plot(
                    x_test_local.flatten(),
                    predictions.flatten(),
                    label="Model Predictions",
                    color="red",
                )
                plt.title(
                    f"Epoch {epoch + 1}/{self.num_epochs} - Validation Loss: {val_loss:.4f}"
                )
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid()
                plt.show()
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, "
                    f"Training Loss: {avg_loss:.4f}, "
                    f"Validation Loss: {val_loss:.4f}"
                )

        return params, {
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        }
