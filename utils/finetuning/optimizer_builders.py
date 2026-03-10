"""
Optimizer composition for fine-tuning techniques.

Central orchestration point that builds complex optimizers by composing:
- Gradient clipping (optax.clip_by_global_norm)
- Parameter freezing (optax.multi_transform)
- LoRA partitioning (optax.multi_transform with lora params)
- Base optimizer (Adam with LR schedule)

Uses optax.chain() and optax.multi_transform() for modular composition.
"""

import logging

import optax
from omegaconf import DictConfig

from utils.finetuning.freezing import create_param_labels_for_optax, log_freezing_statistics
from utils.misc import setup_optimizer

logger = logging.getLogger(__name__)


def build_finetuning_optimizer(
    cfg: DictConfig,
    params: dict,
    pretrained_params: dict | None = None,
) -> optax.GradientTransformation:
    """Build optimizer with fine-tuning techniques applied.

    Composes multiple techniques based on cfg.finetuning:
    1. Gradient clipping (if enabled)
    2. Base optimizer (Adam + LR schedule from cfg.optimizer)
    3. Parameter partitioning (LoRA or freezing, if enabled)

    Args:
        cfg: Full configuration object with optimizer and finetuning sections
        params: Current model parameters
        pretrained_params: Pretrained parameters (required for certain techniques)

    Returns:
        optax.GradientTransformation (composed optimizer)

    Example:
        >>> optimizer = build_finetuning_optimizer(cfg, params, pretrained_params)
        >>> opt_state = optimizer.init(params)
        >>> updates, new_opt_state = optimizer.update(grads, opt_state, params)

    Note:
        Falls back to standard setup_optimizer() if fine-tuning is disabled or not configured.
    """
    # Check if fine-tuning is enabled
    if not cfg.get("finetuning", {}).get("enabled", False):
        logger.info("Fine-tuning disabled, using standard optimizer")
        return setup_optimizer(cfg)

    logger.info("=" * 80)
    logger.info("BUILDING FINE-TUNING OPTIMIZER")
    logger.info("=" * 80)

    # Build base optimizer (Adam + LR schedule)
    base_optimizer = setup_optimizer(cfg)
    logger.info(f"Base optimizer: {cfg.optimizer.algorithm}")

    # 1. Add gradient clipping (if enabled)
    if _is_enabled(cfg.finetuning, "gradient_clipping"):
        max_norm = cfg.finetuning.gradient_clipping.max_norm
        logger.info(f"Gradient clipping enabled: max_norm = {max_norm}")
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm),
            base_optimizer,
        )

    # 2. Add parameter partitioning (LoRA or freezing)
    # Check LoRA first (takes precedence over freezing)
    if _is_enabled(cfg.finetuning, "lora"):
        logger.info("LoRA enabled")
        optimizer = _build_lora_optimizer(cfg, params, base_optimizer)

    elif _is_enabled(cfg.finetuning, "freezing"):
        logger.info("Parameter freezing enabled")
        optimizer = _build_freezing_optimizer(cfg, params, base_optimizer)

    else:
        # No partitioning, use base optimizer
        logger.info("No parameter partitioning (all params trainable)")
        optimizer = base_optimizer

    logger.info("=" * 80)

    return optimizer


def _is_enabled(finetuning_cfg, technique_name: str) -> bool:
    """Check if a fine-tuning technique is enabled in config."""
    technique_cfg = finetuning_cfg.get(technique_name, {})
    return technique_cfg.get("enabled", False)


def _build_freezing_optimizer(
    cfg: DictConfig,
    params: dict,
    base_optimizer: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Build optimizer with parameter freezing via multi_transform.

    Args:
        cfg: Configuration object
        params: Model parameters
        base_optimizer: Base optimizer to apply to trainable params

    Returns:
        Composed optimizer with freezing
    """
    freeze_config = cfg.finetuning.freezing

    # Create parameter labels ('trainable' or 'frozen')
    param_labels = create_param_labels_for_optax(params, freeze_config)

    # Log freezing statistics
    from utils.finetuning.freezing import create_frozen_mask

    frozen_mask = create_frozen_mask(params, freeze_config)
    log_freezing_statistics(params, frozen_mask)

    logger.info(f"Freezing strategy: {freeze_config.get('strategy', 'component')}")
    if freeze_config.get("frozen_components"):
        logger.info(f"Frozen components: {freeze_config.frozen_components}")
    if freeze_config.get("frozen_layers"):
        logger.info(f"Frozen layers: {freeze_config.frozen_layers}")

    # Build multi_transform optimizer
    optimizer = optax.multi_transform(
        transforms={
            "trainable": base_optimizer,
            "frozen": optax.set_to_zero(),  # Zero gradients for frozen params
        },
        param_labels=param_labels,
    )

    return optimizer


def _build_lora_optimizer(
    cfg: DictConfig,
    params: dict,
    base_optimizer: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Build optimizer with LoRA parameter partitioning.

    LoRA params (lora_a, lora_b) are trainable, base params are frozen.

    Args:
        cfg: Configuration object
        params: Model parameters (should already have LoRA params from LoRADense layers)
        base_optimizer: Base optimizer to apply to LoRA params

    Returns:
        Composed optimizer with LoRA partitioning

    Note:
        LoRA is enabled by setting use_lora_embedder/processor/decoder=True in config.
        The model automatically uses LoRADense layers from models.lora_layers.
    """
    from utils.finetuning.lora import count_lora_parameters, create_lora_partition_spec

    lora_config = cfg.finetuning.lora

    # Get LoRA partition spec
    partition_fn = create_lora_partition_spec(lora_config)
    param_labels = partition_fn(params)

    # Count LoRA parameters
    counts = count_lora_parameters(params)
    logger.info(f"LoRA rank: {lora_config.rank}")
    logger.info(f"LoRA alpha: {lora_config.alpha}")
    logger.info(f"Base params: {counts['base_params']:,} (frozen)")
    logger.info(f"LoRA params: {counts['lora_params']:,} (trainable)")
    logger.info(f"Total params: {counts['total_params']:,}")
    logger.info(f"LoRA ratio: {100.0 * counts['lora_ratio']:.2f}%")

    # Build multi_transform optimizer
    optimizer = optax.multi_transform(
        transforms={
            "lora_params": base_optimizer,  # Train LoRA adapters
            "base_params": optax.set_to_zero(),  # Freeze base weights
        },
        param_labels=param_labels,
    )

    return optimizer


def build_standard_optimizer_with_clipping(
    cfg: DictConfig,
) -> optax.GradientTransformation:
    """Build standard optimizer with optional gradient clipping.

    Convenience function for non-fine-tuning workflows that still want clipping.

    Args:
        cfg: Configuration object

    Returns:
        Optimizer with optional clipping

    Example:
        >>> # In standard training (no fine-tuning)
        >>> optimizer = build_standard_optimizer_with_clipping(cfg)
    """
    base_optimizer = setup_optimizer(cfg)

    # Add gradient clipping if configured
    if cfg.get("finetuning", {}).get("gradient_clipping", {}).get("enabled", False):
        max_norm = cfg.finetuning.gradient_clipping.max_norm
        logger.info(f"Gradient clipping enabled: max_norm = {max_norm}")
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer

    return optimizer
