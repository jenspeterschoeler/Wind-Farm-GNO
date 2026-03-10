"""
Wake-aware loss functions for improved wake prediction.

This module provides loss functions that address the tendency of GNO models to
over/under predict wake effects by:
1. Asymmetric weighting of under-prediction vs over-prediction errors
2. Higher weighting for errors in wake regions (low velocity areas)

The wake-aware loss can be used as a drop-in replacement for standard MSE loss
in the training loop.
"""

import jax.numpy as jnp


def wake_aware_mse_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    node_mask: jnp.ndarray,
    deficit_threshold: float = 0.1,
    alpha_under: float = 0.7,
    alpha_over: float = 0.3,
    wake_weight: float = 2.0,
) -> jnp.ndarray:
    """
    Asymmetric MSE with wake region weighting.

    This loss function penalizes errors asymmetrically based on:
    1. Error direction: Under-predicting velocity (over-predicting deficit) can be
       weighted differently than over-predicting velocity
    2. Spatial location: Errors in wake regions (low velocity) are weighted more
       heavily than errors in freestream regions

    Args:
        predictions: Model predictions [n_nodes, 1]
        targets: Ground truth [n_nodes, 1]
        node_mask: Valid node mask [n_nodes, 1] - 1 for valid nodes, 0 for padding
        deficit_threshold: Threshold for wake region detection. A value of 0.1 means
            nodes with velocity < 90% of max are considered wake region.
        alpha_under: Weight for under-predicting velocity (over-predicting deficit).
            Higher values penalize conservative predictions more.
        alpha_over: Weight for over-predicting velocity (under-predicting deficit).
            Higher values penalize aggressive predictions more.
        wake_weight: Multiplier for errors in wake regions. Values > 1 increase
            focus on wake prediction accuracy.

    Returns:
        Scalar loss value

    Example:
        >>> # Standard MSE-like behavior (symmetric, uniform weighting)
        >>> loss = wake_aware_mse_loss(pred, target, mask,
        ...                            alpha_under=0.5, alpha_over=0.5, wake_weight=1.0)

        >>> # Focus on not under-predicting wake effects
        >>> loss = wake_aware_mse_loss(pred, target, mask,
        ...                            alpha_under=0.7, alpha_over=0.3, wake_weight=2.0)
    """
    # Compute errors: positive = under-predicted velocity, negative = over-predicted
    errors = targets - predictions
    sq_errors = errors**2

    # Asymmetric weighting based on error sign
    # errors >= 0: model under-predicted (target > prediction)
    # errors < 0: model over-predicted (target < prediction)
    alpha = jnp.where(errors >= 0, alpha_under, alpha_over)
    asymmetric_errors = alpha * sq_errors

    # Identify wake regions based on velocity deficit
    # Wake region = velocity significantly lower than maximum (freestream)
    # We use masked max to avoid including padding nodes
    masked_targets = jnp.where(node_mask > 0, targets, -jnp.inf)
    max_target = jnp.max(masked_targets) + 1e-8

    # A node is in wake region if its velocity is below (1 - deficit_threshold) * max
    is_wake = (targets / max_target) < (1.0 - deficit_threshold)

    # Apply wake region weighting
    region_weight = jnp.where(is_wake, wake_weight, 1.0)
    weighted_errors = asymmetric_errors * region_weight * node_mask

    # Compute mean over valid nodes
    return jnp.sum(weighted_errors) / (jnp.sum(node_mask) + 1e-8)


def gradient_weighted_mse_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    node_mask: jnp.ndarray,
    positions: jnp.ndarray,
    gradient_weight: float = 1.5,
) -> jnp.ndarray:
    """
    MSE loss with additional weighting for high-gradient regions.

    This loss function increases weight for nodes in regions where the flow field
    has steep gradients (wake edges), which are often the hardest to predict
    accurately.

    Args:
        predictions: Model predictions [n_nodes, 1]
        targets: Ground truth [n_nodes, 1]
        node_mask: Valid node mask [n_nodes, 1]
        positions: Node positions [n_nodes, 2]
        gradient_weight: Additional weight multiplier for high-gradient regions

    Returns:
        Scalar loss value

    Note:
        This is a simplified implementation that approximates gradient magnitude.
        For more accurate gradient computation, consider using neighboring node
        information from the graph structure.
    """
    # Base squared errors
    sq_errors = (targets - predictions) ** 2

    # Estimate local gradient magnitude using variance of targets
    # This is a simplified proxy - high variance indicates high gradient region
    mean_target = jnp.sum(targets * node_mask) / (jnp.sum(node_mask) + 1e-8)
    target_variance = (targets - mean_target) ** 2

    # Normalize variance to [0, 1] range
    max_var = jnp.max(target_variance * node_mask) + 1e-8
    normalized_variance = target_variance / max_var

    # Higher weight for high-variance (high-gradient) regions
    gradient_factor = 1.0 + (gradient_weight - 1.0) * normalized_variance

    weighted_errors = sq_errors * gradient_factor * node_mask

    return jnp.sum(weighted_errors) / (jnp.sum(node_mask) + 1e-8)


def combined_wake_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    node_mask: jnp.ndarray,
    deficit_threshold: float = 0.1,
    alpha_under: float = 0.7,
    alpha_over: float = 0.3,
    wake_weight: float = 2.0,
    mse_weight: float = 0.5,
) -> jnp.ndarray:
    """
    Combined loss: weighted average of wake-aware MSE and standard MSE.

    This allows gradual transition from standard MSE to wake-aware loss during
    training, or can be used to balance wake focus with overall accuracy.

    Args:
        predictions: Model predictions [n_nodes, 1]
        targets: Ground truth [n_nodes, 1]
        node_mask: Valid node mask [n_nodes, 1]
        deficit_threshold: Threshold for wake region detection
        alpha_under: Weight for under-predicting velocity
        alpha_over: Weight for over-predicting velocity
        wake_weight: Multiplier for errors in wake regions
        mse_weight: Weight for standard MSE component (0 to 1).
            0 = pure wake-aware loss, 1 = pure MSE

    Returns:
        Scalar loss value
    """
    # Standard MSE
    sq_errors = (targets - predictions) ** 2
    standard_mse = jnp.sum(sq_errors * node_mask) / (jnp.sum(node_mask) + 1e-8)

    # Wake-aware loss
    wake_loss = wake_aware_mse_loss(
        predictions,
        targets,
        node_mask,
        deficit_threshold=deficit_threshold,
        alpha_under=alpha_under,
        alpha_over=alpha_over,
        wake_weight=wake_weight,
    )

    # Weighted combination
    return mse_weight * standard_mse + (1.0 - mse_weight) * wake_loss


def far_wake_weighted_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    node_mask: jnp.ndarray,
    strong_wake_threshold: float = 0.15,
    weak_wake_threshold: float = 0.05,
    far_wake_weight: float = 2.0,
) -> jnp.ndarray:
    """
    Loss weighted by velocity deficit - prioritizes far-wake accuracy.

    This loss function addresses the common issue of GNO models under-predicting
    velocity recovery in far-wake regions (downstream areas where velocity
    approaches freestream). By weighting errors based on velocity deficit,
    we can improve accuracy in the critical far-wake region.

    Uses TARGET velocity to determine wake region (valid approach: we're weighting
    errors, not leaking predictions). Same methodology as existing wake_aware_mse_loss.

    Wake regions (by velocity deficit):
    - Strong wake: deficit > strong_wake_threshold (high wake effect, weight = 1.0)
    - Moderate wake: between thresholds (transition region)
    - Far-wake: deficit < weak_wake_threshold (recovering, weight = far_wake_weight)

    Args:
        predictions: Model predictions [n_nodes, 1]
        targets: Ground truth [n_nodes, 1]
        node_mask: Valid node mask [n_nodes, 1] - 1 for valid nodes, 0 for padding
        strong_wake_threshold: Velocity deficit threshold for strong wake.
            Default 0.15 means >15% deficit is strong wake.
        weak_wake_threshold: Velocity deficit threshold for far-wake.
            Default 0.05 means <5% deficit is far-wake (recovering toward freestream).
        far_wake_weight: Weight multiplier for errors in far-wake regions.
            Values > 1 prioritize far-wake accuracy.

    Returns:
        Scalar loss value

    Example:
        >>> # Standard MSE-like behavior (uniform weighting)
        >>> loss = far_wake_weighted_loss(pred, target, mask, far_wake_weight=1.0)

        >>> # Prioritize far-wake accuracy (velocity recovery)
        >>> loss = far_wake_weighted_loss(pred, target, mask,
        ...                                strong_wake_threshold=0.15,
        ...                                weak_wake_threshold=0.05,
        ...                                far_wake_weight=2.0)
    """
    # Compute squared errors
    sq_errors = (targets - predictions) ** 2

    # Compute velocity deficit from TARGETS: deficit = 1 - U/U_inf
    # Use masked max to estimate freestream velocity (avoid padding nodes)
    masked_targets = jnp.where(node_mask > 0, targets, -jnp.inf)
    max_target = jnp.max(masked_targets) + 1e-8
    deficit = 1.0 - targets / max_target

    # Calculate weight based on deficit region
    # Far-wake (low deficit) gets higher weight
    # Strong wake (high deficit) gets weight 1.0
    # Linear interpolation in between
    normalized_deficit = (strong_wake_threshold - deficit) / (
        strong_wake_threshold - weak_wake_threshold + 1e-8
    )
    normalized_deficit = jnp.clip(normalized_deficit, 0.0, 1.0)

    # Weight: 1.0 for strong wake, far_wake_weight for far-wake
    deficit_weight = 1.0 + (far_wake_weight - 1.0) * normalized_deficit

    # Apply weights and mask
    weighted_errors = sq_errors * deficit_weight * node_mask

    # Return mean over valid nodes
    return jnp.sum(weighted_errors) / (jnp.sum(node_mask) + 1e-8)


def stratified_wake_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    node_mask: jnp.ndarray,
    strong_wake_threshold: float = 0.15,
    weak_wake_threshold: float = 0.05,
    strong_wake_weight: float = 1.0,
    moderate_wake_weight: float = 1.5,
    far_wake_weight: float = 2.0,
) -> jnp.ndarray:
    """
    Stratified loss with separate weights for each wake region.

    More flexible than far_wake_weighted_loss - allows independent control
    of weight for each velocity deficit band:
    - Strong wake (>15% deficit): Near-wake, highest wake effect
    - Moderate wake (5-15% deficit): Transition region
    - Far-wake (<5% deficit): Recovery toward freestream

    Args:
        predictions: Model predictions [n_nodes, 1]
        targets: Ground truth [n_nodes, 1]
        node_mask: Valid node mask [n_nodes, 1]
        strong_wake_threshold: Deficit threshold for strong wake (default 0.15)
        weak_wake_threshold: Deficit threshold for far-wake (default 0.05)
        strong_wake_weight: Weight for strong wake region (default 1.0)
        moderate_wake_weight: Weight for moderate wake region (default 1.5)
        far_wake_weight: Weight for far-wake region (default 2.0)

    Returns:
        Scalar loss value
    """
    sq_errors = (targets - predictions) ** 2

    # Compute velocity deficit
    masked_targets = jnp.where(node_mask > 0, targets, -jnp.inf)
    max_target = jnp.max(masked_targets) + 1e-8
    deficit = 1.0 - targets / max_target

    # Categorize into wake regions
    is_strong_wake = deficit > strong_wake_threshold
    is_far_wake = deficit < weak_wake_threshold
    is_moderate_wake = ~is_strong_wake & ~is_far_wake

    # Apply stratified weights
    weights = jnp.where(
        is_strong_wake,
        strong_wake_weight,
        jnp.where(is_moderate_wake, moderate_wake_weight, far_wake_weight),
    )

    weighted_errors = sq_errors * weights * node_mask
    return jnp.sum(weighted_errors) / (jnp.sum(node_mask) + 1e-8)
