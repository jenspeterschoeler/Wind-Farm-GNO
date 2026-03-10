"""Multi-layer perceptron implementations for GNO components."""

from collections.abc import Callable, Sequence

import jax
from flax import linen as nn


class MLP(nn.Module):
    """A multi-layer perceptron.

    Supports optional LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

    Args:
        name: Name of the MLP module
        feature_sizes: Hidden layer sizes
        output_size: Output dimension
        activation: Activation function (default: ReLU)
        weight_initializer: Weight initialization function
        dropout_rate: Dropout rate (0.0 = no dropout)
        layer_norm: Whether to apply layer normalization
        use_lora: Whether to use LoRA instead of standard Dense layers
        lora_rank: LoRA rank (typical: 4-16)
        lora_alpha: LoRA scaling factor (typical: 2 * lora_rank)
    """

    name: str
    feature_sizes: Sequence[int]
    output_size: int
    activation: Callable = nn.relu
    weight_initializer: Callable = jax.nn.initializers.he_normal()
    dropout_rate: float = 0.0
    layer_norm: bool = False
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        """Forward pass through MLP.

        Args:
            inputs: Input array
            train: Whether in training mode (for dropout)

        Returns:
            Output array
        """
        # Import LoRADense if needed
        if self.use_lora:
            from models.lora_layers import LoRADense

        x = inputs

        # Hidden layers
        for size in self.feature_sizes:
            if self.use_lora:
                # Use LoRA-enabled Dense layer
                x = LoRADense(
                    features=size,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    kernel_init=self.weight_initializer,
                )(x)
            else:
                # Standard Dense layer
                x = nn.Dense(
                    features=size,
                    kernel_init=self.weight_initializer,
                )(x)

            x = self.activation(x)

            if self.dropout_rate > 0.0:
                # The layer makes its own rng inside the dropout call
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        # Output layer
        if self.use_lora:
            x = LoRADense(
                features=self.output_size,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                kernel_init=self.weight_initializer,
            )(x)
        else:
            x = nn.Dense(
                features=self.output_size,
                kernel_init=self.weight_initializer,
            )(x)

        if self.layer_norm:
            x = nn.LayerNorm()(x)

        return x
