"""
High-level freezing utilities for parameter fine-tuning.

Provides user-friendly interface for parameter freezing with:
- Mask creation from freeze configs
- Verification that frozen params don't change
- Statistics logging

Uses param_partitions.py for PyTree manipulation.
"""

import logging

import jax.numpy as jnp
from jax import tree_util

from utils.finetuning.param_partitions import count_params_by_partition, create_partition_spec

logger = logging.getLogger(__name__)


def create_frozen_mask(params: dict, freeze_config: dict) -> dict:
    """Create boolean mask indicating frozen parameters.

    Args:
        params: Parameter dictionary (PyTree)
        freeze_config: Freezing configuration with keys:
            - strategy: 'disabled', 'component', 'layer', or 'hybrid'
            - frozen_components: List of components to freeze
            - frozen_layers: Dict mapping processor → layer indices

    Returns:
        Boolean PyTree (True = frozen, False = trainable)

    Example:
        >>> freeze_config = {
        ...     'strategy': 'component',
        ...     'frozen_components': ['embedder', 'wt_processor']
        ... }
        >>> frozen_mask = create_frozen_mask(params, freeze_config)
    """
    # Get partition function
    partition_fn = create_partition_spec(freeze_config)

    # Get parameter labels
    param_labels = partition_fn(params)

    # Convert to boolean mask (True = frozen)
    frozen_mask = tree_util.tree_map(lambda label: label == "frozen", param_labels)

    return frozen_mask


def verify_freezing(
    params_before: dict,
    params_after: dict,
    frozen_mask: dict,
    tolerance: float = 1e-8,
) -> bool:
    """Verify that frozen parameters haven't changed during training.

    Args:
        params_before: Parameters before training step
        params_after: Parameters after training step
        frozen_mask: Boolean PyTree (True = should be frozen)
        tolerance: Maximum allowed change for frozen parameters

    Returns:
        True if all frozen params unchanged, False otherwise (logs warning)

    Example:
        >>> # After a training step
        >>> is_valid = verify_freezing(old_params, new_params, frozen_mask)
        >>> if not is_valid:
        ...     logger.warning("Frozen parameters changed!")
    """
    violations = []

    def check_param(path, before, after, is_frozen):
        if is_frozen:
            max_change = jnp.max(jnp.abs(after - before))
            if max_change > tolerance:
                path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
                violations.append((path_str, float(max_change)))

    # Check all parameters
    tree_util.tree_map_with_path(
        check_param,
        params_before,
        params_after,
        frozen_mask,
    )

    if violations:
        logger.warning("=" * 60)
        logger.warning("FREEZING VIOLATION DETECTED")
        logger.warning("=" * 60)
        logger.warning(f"Found {len(violations)} frozen parameters that changed during training:")
        for path_str, max_change in violations[:5]:  # Show first 5
            logger.warning(f"  {path_str}: max_change = {max_change:.2e}")
        if len(violations) > 5:
            logger.warning(f"  ... and {len(violations) - 5} more")
        logger.warning("=" * 60)
        return False

    return True


def log_freezing_statistics(params: dict, frozen_mask: dict, logger_instance=None) -> dict:
    """Compute and log freezing statistics.

    Args:
        params: Parameter dictionary
        frozen_mask: Boolean PyTree (True = frozen)
        logger_instance: Optional logger (uses module logger if None)

    Returns:
        Dictionary with statistics:
            - total_params: Total parameter count
            - trainable_params: Count of trainable parameters
            - frozen_params: Count of frozen parameters
            - freeze_ratio: Fraction of frozen parameters (0.0 - 1.0)

    Example:
        >>> stats = log_freezing_statistics(params, frozen_mask)
        >>> print(f"Freeze ratio: {stats['freeze_ratio']:.1%}")
    """
    if logger_instance is None:
        logger_instance = logger

    # Convert mask to labels
    param_labels = tree_util.tree_map(
        lambda is_frozen: "frozen" if is_frozen else "trainable",
        frozen_mask,
    )

    # Count parameters
    counts = count_params_by_partition(params, param_labels)

    # Compute statistics
    total_params = counts["total"]
    trainable_params = counts.get("trainable", 0)
    frozen_params = counts.get("frozen", 0)
    freeze_ratio = frozen_params / total_params if total_params > 0 else 0.0

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "freeze_ratio": freeze_ratio,
    }

    # Log statistics
    logger_instance.info("=" * 60)
    logger_instance.info("PARAMETER FREEZING STATISTICS")
    logger_instance.info("=" * 60)
    logger_instance.info(f"Total parameters:     {total_params:>10,d}")
    logger_instance.info(
        f"Trainable parameters: {trainable_params:>10,d} ({100.0 * (1 - freeze_ratio):>5.1f}%)"
    )
    logger_instance.info(
        f"Frozen parameters:    {frozen_params:>10,d} ({100.0 * freeze_ratio:>5.1f}%)"
    )
    logger_instance.info("=" * 60)

    return stats


def create_param_labels_for_optax(params: dict, freeze_config: dict) -> dict:
    """Create parameter labels for optax.multi_transform().

    Convenience function that combines partition spec creation and labeling.

    Args:
        params: Parameter dictionary
        freeze_config: Freezing configuration

    Returns:
        PyTree with 'trainable' or 'frozen' labels at each leaf

    Example:
        >>> param_labels = create_param_labels_for_optax(params, freeze_config)
        >>> optimizer = optax.multi_transform({
        ...     'trainable': optax.adam(1e-3),
        ...     'frozen': optax.set_to_zero()
        ... }, param_labels=param_labels)
    """
    partition_fn = create_partition_spec(freeze_config)
    param_labels = partition_fn(params)
    return param_labels
