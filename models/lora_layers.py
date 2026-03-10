"""
LoRA (Low-Rank Adaptation) layers for parameter-efficient fine-tuning.

Manual implementation for Flax Linen without external dependencies.

LoRA Formula:
    output = (W + (B @ A) * (alpha / rank)) @ input

Where:
    - W: Frozen pretrained weight matrix (in_features × out_features)
    - B: Trainable matrix (out_features × rank)
    - A: Trainable matrix (rank × in_features)
    - alpha: Scaling factor (typically 2 * rank)
    - rank: Low-rank dimension (typically 4-16)

References:
    - LoRA paper: https://arxiv.org/abs/2106.09685
    - Flax Linen docs: https://flax.readthedocs.io/en/latest/
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class LoRADense(nn.Module):
    """Dense layer with LoRA (Low-Rank Adaptation).

    This layer adds trainable low-rank matrices (A and B) to a frozen Dense layer,
    enabling parameter-efficient fine-tuning. The base kernel is frozen, and only
    the low-rank adapters are trained.

    Attributes:
        features: Number of output features
        lora_rank: Rank of the low-rank adaptation (r in the paper)
        lora_alpha: Scaling factor for LoRA (typically 2 * lora_rank)
        use_bias: Whether to include bias (frozen with base kernel)
        dtype: Data type for computations
        kernel_init: Initializer for base kernel (used only if not loading pretrained)
        lora_a_init: Initializer for A matrix (default: normal with std=1/sqrt(rank))
        lora_b_init: Initializer for B matrix (default: zeros, following LoRA paper)

    Example:
        >>> # Replace Dense layer with LoRADense
        >>> # Before:
        >>> dense = nn.Dense(features=128)
        >>>
        >>> # After:
        >>> lora_dense = LoRADense(features=128, lora_rank=8, lora_alpha=16)
    """

    features: int
    lora_rank: int = 8
    lora_alpha: float = 16.0
    use_bias: bool = True
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    lora_a_init: Callable | None = None
    lora_b_init: Callable | None = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with LoRA.

        Args:
            inputs: Input array of shape (..., in_features)

        Returns:
            Output array of shape (..., features)
        """
        in_features = inputs.shape[-1]

        # Base kernel (will be frozen via optimizer partitioning)
        kernel = self.param("kernel", self.kernel_init, (in_features, self.features), self.dtype)

        # Default initializers following LoRA paper
        # A initialized with normal distribution (std = 1/sqrt(rank))
        lora_a_initializer = (
            self.lora_a_init
            if self.lora_a_init is not None
            else nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.lora_rank))
        )
        # B initialized to zeros (so initial LoRA contribution is zero)
        lora_b_initializer = (
            self.lora_b_init if self.lora_b_init is not None else nn.initializers.zeros
        )

        # LoRA low-rank matrices (trainable)
        lora_a = self.param("lora_a", lora_a_initializer, (self.lora_rank, in_features), self.dtype)

        lora_b = self.param(
            "lora_b", lora_b_initializer, (self.features, self.lora_rank), self.dtype
        )

        # Bias (frozen with base kernel)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
        else:
            bias = None

        # Compute scaling factor
        scale = self.lora_alpha / self.lora_rank

        # LoRA forward pass: output = (W + B @ A * scale) @ input
        # Efficient computation: W @ x + B @ (A @ x) * scale
        base_output = jnp.dot(inputs, kernel)
        lora_output = jnp.dot(jnp.dot(inputs, lora_a.T), lora_b.T) * scale

        output = base_output + lora_output

        if bias is not None:
            output = output + bias

        return output
