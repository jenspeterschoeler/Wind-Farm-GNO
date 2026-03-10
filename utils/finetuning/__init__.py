"""
Fine-tuning utilities for transfer learning in GNO models.

This module provides modular, composable tools for advanced fine-tuning techniques:
- LoRA (Low-Rank Adaptation) - manual implementation, always available
- Parameter freezing (component-level and layer-level)
- Gradient norm clipping

All techniques can be combined via optax.chain() and optax.multi_transform().
"""

from utils.finetuning.freezing import create_frozen_mask, log_freezing_statistics, verify_freezing
from utils.finetuning.lora import (
    count_lora_parameters,
    create_lora_partition_spec,
)
from utils.finetuning.optimizer_builders import build_finetuning_optimizer
from utils.finetuning.param_partitions import (
    create_partition_spec,
    partition_params_by_component,
    partition_params_by_layer,
)
from utils.finetuning.wake_loss import (
    combined_wake_loss,
    gradient_weighted_mse_loss,
    wake_aware_mse_loss,
)

__all__ = [
    # Parameter partitioning
    "partition_params_by_component",
    "partition_params_by_layer",
    "create_partition_spec",
    # Freezing
    "create_frozen_mask",
    "verify_freezing",
    "log_freezing_statistics",
    # LoRA
    "count_lora_parameters",
    "create_lora_partition_spec",
    # Losses
    "wake_aware_mse_loss",
    "gradient_weighted_mse_loss",
    "combined_wake_loss",
    # Optimizer builders
    "build_finetuning_optimizer",
]
