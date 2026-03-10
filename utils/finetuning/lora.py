"""
LoRA (Low-Rank Adaptation) utilities for parameter-efficient fine-tuning.

Provides utility functions for:
- Parameter counting (count_lora_parameters)
- Parameter partitioning for optimizer (create_lora_partition_spec)

The LoRADense layer is located in models.lora_layers.

LoRA implementation approach:
    1. Set use_lora_embedder/use_lora_processor/use_lora_decoder=True in model config
    2. Model automatically uses LoRADense layers from models.lora_layers
    3. Use create_lora_partition_spec() to partition params for optimizer
    4. Base weights frozen, LoRA adapters (lora_a, lora_b) trainable

References:
    - LoRA paper: https://arxiv.org/abs/2106.09685
"""

import logging

from jax import tree_util

logger = logging.getLogger(__name__)


def create_lora_partition_spec(lora_config: dict):
    """Create partition spec for LoRA parameters.

    Returns a function that labels parameters as 'lora_params' or 'base_params'
    for use with optax.multi_transform().

    Args:
        lora_config: LoRA configuration (not used, kept for API compatibility)

    Returns:
        Function that takes params and returns label PyTree

    Example:
        >>> partition_fn = create_lora_partition_spec(lora_config)
        >>> param_labels = partition_fn(params)
        >>> optimizer = optax.multi_transform({
        ...     'lora_params': optax.adam(1e-3),  # Train LoRA adapters
        ...     'base_params': optax.set_to_zero()  # Freeze base weights
        ... }, param_labels=param_labels)
    """

    def partition_fn(params):
        """Label parameters as 'lora_params' or 'base_params'."""

        def label_param(path, _value):
            path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
            # LoRA parameters have 'lora_a' or 'lora_b' in their path
            if "lora_a" in path_str or "lora_b" in path_str:
                return "lora_params"
            return "base_params"

        param_labels = tree_util.tree_map_with_path(label_param, params)
        return param_labels

    return partition_fn


def count_lora_parameters(params: dict) -> dict[str, int | float]:
    """Count LoRA and base parameters.

    Args:
        params: Parameter dictionary (with LoRA params)

    Returns:
        Dictionary with counts:
            - base_params: Number of base model parameters
            - lora_params: Number of LoRA adapter parameters
            - total_params: Total parameter count
            - lora_ratio: Fraction of params that are LoRA adapters

    Example:
        >>> counts = count_lora_parameters(lora_params)
        >>> print(f"LoRA ratio: {100 * counts['lora_ratio']:.1f}%")
        LoRA ratio: 3.2%
    """
    import numpy as np

    base_count = 0
    lora_count = 0

    # Flatten parameters
    param_dict_flat = tree_util.tree_leaves_with_path(params)

    for path, param in param_dict_flat:
        param_size = int(np.prod(param.shape))

        # Check if this is a LoRA parameter (has 'lora' in path)
        path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        if "lora_a" in path_str or "lora_b" in path_str:
            lora_count += param_size
        else:
            base_count += param_size

    total_count = base_count + lora_count
    lora_ratio = lora_count / total_count if total_count > 0 else 0.0

    return {
        "base_params": base_count,
        "lora_params": lora_count,
        "total_params": total_count,
        "lora_ratio": lora_ratio,
    }
