"""
Configuration Validation

Validates Hydra configuration before training to catch errors early.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


def validate_paths_exist(cfg: DictConfig) -> list[str]:
    """
    Validate that required data paths exist.

    Args:
        cfg: Configuration object

    Returns:
        List of warnings (non-fatal issues)

    Raises:
        ConfigValidationError: If critical paths don't exist
    """
    warnings = []

    # Check data paths
    if hasattr(cfg, "data"):
        if hasattr(cfg.data, "main_path"):
            data_path = Path(cfg.data.main_path)
            if not data_path.exists():
                raise ConfigValidationError(f"Data path does not exist: {data_path}")

        # Check train/val/test paths
        for split in ["train_path", "val_path", "test_path"]:
            if hasattr(cfg.data, split):
                split_path = Path(getattr(cfg.data, split))
                if not split_path.exists():
                    warnings.append(f"{split} does not exist: {split_path}")

    # Check model save path parent exists
    if hasattr(cfg, "model_save_path"):
        save_path = Path(cfg.model_save_path)
        if not save_path.parent.exists():
            raise ConfigValidationError(
                f"Parent directory for model_save_path does not exist: {save_path.parent}"
            )

    return warnings


def validate_parameter_ranges(cfg: DictConfig) -> list[str]:
    """
    Validate that parameters are within reasonable ranges.

    Args:
        cfg: Configuration object

    Returns:
        List of warnings

    Raises:
        ConfigValidationError: If parameters are out of valid ranges
    """
    warnings = []

    # Check model parameters
    if hasattr(cfg, "model"):
        # Latent size should be positive
        if hasattr(cfg.model, "latent_size") and cfg.model.latent_size <= 0:
            raise ConfigValidationError(
                f"latent_size must be positive, got {cfg.model.latent_size}"
            )

        # Hidden layer size should be positive
        if hasattr(cfg.model, "hidden_layer_size") and cfg.model.hidden_layer_size <= 0:
            raise ConfigValidationError(
                f"hidden_layer_size must be positive, got {cfg.model.hidden_layer_size}"
            )

        # Message passing steps should be positive
        for step_param in ["wt_message_passing_steps", "probe_message_passing_steps"]:
            if hasattr(cfg.model, step_param):
                value = getattr(cfg.model, step_param)
                if value <= 0:
                    raise ConfigValidationError(f"{step_param} must be positive, got {value}")

        # Dropout rates should be in [0, 1)
        if hasattr(cfg.model, "regularization"):
            for dropout_param in [
                "encoder_dropout_rate",
                "processor_dropout_rate",
                "decoder_dropout_rate",
            ]:
                if hasattr(cfg.model.regularization, dropout_param):
                    rate = getattr(cfg.model.regularization, dropout_param)
                    if not (0 <= rate < 1):
                        raise ConfigValidationError(
                            f"{dropout_param} must be in [0, 1), got {rate}"
                        )

    # Check optimizer parameters
    if hasattr(cfg, "optimizer"):
        # Number of epochs should be positive
        if hasattr(cfg.optimizer, "n_epochs") and cfg.optimizer.n_epochs <= 0:
            raise ConfigValidationError(f"n_epochs must be positive, got {cfg.optimizer.n_epochs}")

        # Learning rate should be positive
        if hasattr(cfg.optimizer, "lr") and cfg.optimizer.lr <= 0:
            raise ConfigValidationError(f"Learning rate must be positive, got {cfg.optimizer.lr}")

        # Batch size should be positive
        if hasattr(cfg.optimizer, "batch_size") and cfg.optimizer.batch_size <= 0:
            raise ConfigValidationError(
                f"batch_size must be positive, got {cfg.optimizer.batch_size}"
            )

    return warnings


def validate_compatibility(cfg: DictConfig) -> list[str]:
    """
    Validate that configuration options are compatible with each other.

    Args:
        cfg: Configuration object

    Returns:
        List of warnings

    Raises:
        ConfigValidationError: If incompatible options are detected
    """
    warnings = []

    # Check that validation rate divides evenly into epochs
    if (
        hasattr(cfg, "optimizer")
        and hasattr(cfg.optimizer, "validation")
        and hasattr(cfg.optimizer.validation, "rate_of_validation")
    ):
        rate = cfg.optimizer.validation.rate_of_validation
        n_epochs = cfg.optimizer.n_epochs
        if n_epochs % rate != 0:
            warnings.append(
                f"n_epochs ({n_epochs}) is not evenly divisible by "
                f"rate_of_validation ({rate}). Last validation may be skipped."
            )

    # Check early stopping patience vs validation rate
    if (
        hasattr(cfg, "optimizer")
        and hasattr(cfg.optimizer, "early_stop")
        and hasattr(cfg.optimizer.early_stop, "patience")
        and hasattr(cfg.optimizer.validation, "rate_of_validation")
    ):
        patience = cfg.optimizer.early_stop.patience
        val_rate = cfg.optimizer.validation.rate_of_validation
        if patience < val_rate:
            warnings.append(
                f"early_stop.patience ({patience}) is less than "
                f"rate_of_validation ({val_rate}). Early stopping may trigger prematurely."
            )

    # Check that data IO type matches model expectations
    if hasattr(cfg, "data") and hasattr(cfg.data, "io") and hasattr(cfg.data.io, "type"):
        io_type = cfg.data.io.type
        if io_type not in ["GNN", "GNO_probe"]:
            warnings.append(f"Unknown data.io.type: {io_type}. Expected 'GNN' or 'GNO_probe'.")

    return warnings


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration before training.

    Performs comprehensive validation including:
    - Path existence checks
    - Parameter range validation
    - Compatibility checks

    Args:
        cfg: Configuration object to validate

    Raises:
        ConfigValidationError: If validation fails on critical issues

    Note:
        Non-critical issues are logged as warnings but don't prevent execution.
    """
    all_warnings = []

    logger.info("Validating configuration...")

    # Run all validation checks
    try:
        all_warnings.extend(validate_paths_exist(cfg))
        all_warnings.extend(validate_parameter_ranges(cfg))
        all_warnings.extend(validate_compatibility(cfg))
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Log all warnings
    if all_warnings:
        logger.warning("Configuration validation warnings:")
        for warning in all_warnings:
            logger.warning(f"  - {warning}")
    else:
        logger.info("Configuration validation passed with no warnings.")
