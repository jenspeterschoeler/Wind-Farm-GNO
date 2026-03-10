"""
Validate LoRA configuration before training starts.

This module provides validation to catch LoRA configuration mismatches
before expensive cluster jobs are submitted.

Usage in train script:
    from utils.finetuning.validate_lora_config import validate_lora_config, LoRAConfigError

    try:
        lora_settings = validate_lora_config(cfg)
    except LoRAConfigError as e:
        logger.error(f"LoRA Configuration Error: {e}")
        raise SystemExit(1)
"""

import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LoRAConfigError(Exception):
    """Raised when LoRA configuration is invalid or will not be applied."""

    pass


def validate_lora_config(cfg: DictConfig) -> dict:
    """
    Validate LoRA configuration and return what will actually be used.

    This function checks for common configuration mismatches that cause
    LoRA to silently fail to inject adapter layers.

    Args:
        cfg: Hydra configuration object

    Raises:
        LoRAConfigError: If configuration is invalid or incomplete

    Returns:
        dict with actual LoRA settings that will be applied:
            - enabled: bool - whether LoRA is active
            - use_lora_embedder: bool
            - use_lora_processor: bool
            - use_lora_decoder: bool
            - rank: int or None
            - alpha: float or None
            - config_source: str - where config came from
    """
    result = {
        "enabled": False,
        "use_lora_embedder": False,
        "use_lora_processor": False,
        "use_lora_decoder": False,
        "rank": None,
        "alpha": None,
        "config_source": None,
    }

    # Check finetuning.lora config (new format)
    if hasattr(cfg, "finetuning") and cfg.finetuning.get("enabled", False):
        lora_cfg = cfg.finetuning.get("lora", {})
        if lora_cfg.get("enabled", False):
            result["enabled"] = True
            result["config_source"] = "finetuning.lora"
            result["rank"] = lora_cfg.get("rank")
            result["alpha"] = lora_cfg.get("alpha")

            # Check if model initialization will actually use this
            # (This is where the bug was - config exists but not used)
            if not (
                cfg.model.get("use_lora_embedder", False)
                or cfg.model.get("use_lora_processor", False)
                or cfg.model.get("use_lora_decoder", False)
            ):
                # Config says LoRA enabled but model won't use it!
                raise LoRAConfigError(
                    "LoRA enabled in finetuning.lora config but model.use_lora_* flags not set!\n"
                    "The model initialization code expects model.use_lora_embedder/processor/decoder.\n"
                    "Either:\n"
                    "  1. Add these flags to model config, OR\n"
                    "  2. Update utils/model_tools.py to read from finetuning.lora config\n"
                    f"Current config has:\n"
                    f"  finetuning.lora.enabled: {lora_cfg.get('enabled')}\n"
                    f"  finetuning.lora.rank: {result['rank']}\n"
                    f"  model.use_lora_embedder: {cfg.model.get('use_lora_embedder', False)}\n"
                    f"  model.use_lora_processor: {cfg.model.get('use_lora_processor', False)}\n"
                    f"  model.use_lora_decoder: {cfg.model.get('use_lora_decoder', False)}"
                )

    # Check model.use_lora_* config (old format)
    use_lora_all = cfg.model.get("use_lora", False)
    result["use_lora_embedder"] = cfg.model.get("use_lora_embedder", use_lora_all)
    result["use_lora_processor"] = cfg.model.get("use_lora_processor", use_lora_all)
    result["use_lora_decoder"] = cfg.model.get("use_lora_decoder", use_lora_all)

    if any(
        [
            result["use_lora_embedder"],
            result["use_lora_processor"],
            result["use_lora_decoder"],
        ]
    ):
        result["enabled"] = True
        if result["config_source"] is None:
            result["config_source"] = "model.use_lora_*"
        result["rank"] = cfg.model.get("lora_rank", 8)
        result["alpha"] = cfg.model.get("lora_alpha", 16.0)

    # Validation checks
    if result["enabled"]:
        if result["rank"] is None or result["rank"] <= 0:
            raise LoRAConfigError(f"LoRA enabled but invalid rank: {result['rank']}")

        if result["alpha"] is None or result["alpha"] <= 0:
            raise LoRAConfigError(f"LoRA enabled but invalid alpha: {result['alpha']}")

        if not any(
            [
                result["use_lora_embedder"],
                result["use_lora_processor"],
                result["use_lora_decoder"],
            ]
        ):
            raise LoRAConfigError(
                "LoRA enabled but no components will use it! "
                "Set at least one of: use_lora_embedder, use_lora_processor, use_lora_decoder"
            )

        logger.info("✅ LoRA config validation passed:")
        logger.info(f"  Source: {result['config_source']}")
        logger.info(f"  Rank: {result['rank']}, Alpha: {result['alpha']}")
        logger.info(f"  Embedder: {result['use_lora_embedder']}")
        logger.info(f"  Processor: {result['use_lora_processor']}")
        logger.info(f"  Decoder: {result['use_lora_decoder']}")
    else:
        logger.info("ℹ️  LoRA disabled")

    return result
