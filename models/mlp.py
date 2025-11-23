from typing import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jax


class MLP(nn.Module):
    """A multi-layer perceptron."""

    name: str
    feature_sizes: Sequence[int]
    output_size: int
    activation: Callable = nn.relu
    weight_initializer: Callable = jax.nn.initializers.he_normal()
    dropout_rate: float = 0.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=self.weight_initializer,
            )(x)
            x = self.activation(x)
            if self.dropout_rate > 0.0:
                # The layer makes it own rng inside the the dropout call (self.make_rng("dropout"), "dropout" is the default name)
                # dropout_rng = self.make_rng("dropout")
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Dense(
            features=self.output_size,
            kernel_init=self.weight_initializer,
        )(x)

        if self.layer_norm:
            x = nn.LayerNorm()(x)

        return x
