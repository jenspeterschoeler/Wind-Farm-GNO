"""Miscellaneous utility functions for GNO training and evaluation."""

import logging
import os
import platform
import subprocess
from datetime import datetime

import jax
import numpy as np
import optax
from jax import numpy as jnp
from omegaconf import OmegaConf

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_run_info():
    """Get the run info for the experiment"""
    # Get git commit hash if available
    # Try 'git' first, then common system paths as fallback (for isolated envs like pixi)
    git_commit = "unknown"
    for git_cmd in [
        "git",
        "/usr/bin/git",
        "/usr/local/bin/git",
        "/apps/software/git/2.41.0-GCCcore-12.3.0-nodocs/bin/git",  # Sophia cluster
    ]:
        try:
            git_commit = (
                subprocess.check_output([git_cmd, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )
            break  # Success, stop trying
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            continue

    run_info = {
        "TimeInitiatedTraining": datetime.now().replace(microsecond=0).isoformat(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "available_devices": str(jax.devices()),
        "git_commit": git_commit,
    }
    return run_info


def add_to_hydra_cfg(cfg, key, value):
    """Function to add a key-value pair to a hydra config object, do not use for replacement of existing keys (unless not part of a struct)"""
    OmegaConf.set_struct(cfg, False)
    cfg.update(OmegaConf.create({key: value}))
    OmegaConf.set_struct(cfg, True)
    return cfg


def setup_optimizer(cfg):
    """Setup the optimizer for the model"""
    if cfg.optimizer.algorithm == "adam":
        if "lr_schedule" not in cfg.optimizer:
            assert "learning_rate" in cfg.optimizer, "learning_rate is not defined"
            lr_schedule = optax.constant_schedule(cfg.optimizer.learning_rate)
            logging.info(
                "No learning rate schedule is defined, using default constant learning rate"
            )

        elif cfg.optimizer.lr_schedule.type == "constant":
            assert "learning_rate" in cfg.optimizer.lr_schedule, "learning_rate is not defined"
            lr_schedule = optax.constant_schedule(cfg.optimizer.lr_schedule.learning_rate)

        elif cfg.optimizer.lr_schedule.type == "piecewise_constant":
            assert "init_learning_rate" in cfg.optimizer.lr_schedule, (
                "init_learning_rate is not defined"
            )

            bounds_and_scales = dict(
                zip(
                    cfg.optimizer.lr_schedule.boundaries,
                    cfg.optimizer.lr_schedule.scales,
                )
            )

            lr_schedule = optax.piecewise_constant_schedule(
                init_value=cfg.optimizer.lr_schedule.init_learning_rate,
                boundaries_and_scales=bounds_and_scales,
            )

        else:
            raise NotImplementedError(
                "The schedule is not implemented, chose from 'constant', 'piecewise_constant'"
            )
        print(lr_schedule)
        optimizer = optax.adam(lr_schedule)

    else:
        raise NotImplementedError("The optimizer is not implemented, chose from 'adam'")
    return optimizer


def convert_to_wandb_format(nested_dict, parent_key="", separator="/"):
    """
    Convert nested dictionary to W&B compatible flat format with slash-separated keys
    Args:
        nested_dict: Input dictionary (possibly nested)
        parent_key: Internal use for recursion
        separator: Character to use for path separation (default '/')
    Returns:
        dict: Flat dictionary with slash-separated keys
    """
    items = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(convert_to_wandb_format(v, new_key, separator))
        else:
            items[new_key] = v
    return items


def _get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Get the latest checkpoint from a checkpoint directory.

    Orbax saves checkpoints with epoch number as directory name.
    Returns the path to the highest-numbered checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints_best_mse/)

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    list_checkpoints = os.listdir(checkpoint_dir)
    if not list_checkpoints:
        return None

    # Filter to numeric directories only (Orbax format)
    numeric_dirs = []
    for d in list_checkpoints:
        try:
            numeric_dirs.append((int(d), d))
        except ValueError:
            continue

    if not numeric_dirs:
        return None

    # Get highest epoch
    best_epoch, best_dir = max(numeric_dirs, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, best_dir)


def get_model_save_paths(model_path: str) -> dict:
    """
    Get paths to all model checkpoints (multi-metric aware).

    Supports both legacy single-checkpoint format and new multi-metric format:
    - Legacy: model/checkpoints/ (single best checkpoint)
    - New: model/checkpoints_best_mse/, checkpoints_best_mae/, checkpoints_best_hybrid/

    Args:
        model_path: Path to model save directory (e.g., outputs/experiment/model/)

    Returns:
        Dict with keys:
        - 'final': Path to final model (or None)
        - 'best_mse': Path to best MSE checkpoint (or None)
        - 'best_mae': Path to best MAE checkpoint (or None)
        - 'best_hybrid': Path to best hybrid checkpoint (or None)
        - 'best': Legacy key pointing to best_mse (for backward compatibility)
        - 'periodic': List of all periodic checkpoint paths (or empty list)
    """
    paths = {
        "final": None,
        "best_mse": None,
        "best_mae": None,
        "best_hybrid": None,
        "best": None,  # Legacy compatibility
        "periodic": [],
    }

    if not os.path.exists(model_path):
        raise ValueError(f"Model save path {model_path} does not exist")

    list_dir = os.listdir(model_path)

    for dir_ in list_dir:
        full_path = os.path.join(model_path, dir_)

        if "plots" in dir_:
            continue
        elif "final" in dir_:
            paths["final"] = full_path
        elif dir_ == "checkpoints_best_mse":
            paths["best_mse"] = _get_latest_checkpoint(full_path)
        elif dir_ == "checkpoints_best_mae":
            paths["best_mae"] = _get_latest_checkpoint(full_path)
        elif dir_ == "checkpoints_best_hybrid":
            paths["best_hybrid"] = _get_latest_checkpoint(full_path)
        elif dir_ == "checkpoints_periodic":
            # Get all periodic checkpoints
            periodic: list[str] = []
            periodic_dir = full_path
            if os.path.exists(periodic_dir):
                for ckpt_dir in os.listdir(periodic_dir):
                    try:
                        int(ckpt_dir)  # Validate it's a numeric epoch dir
                        periodic.append(os.path.join(periodic_dir, ckpt_dir))
                    except ValueError:
                        continue
            # Sort by epoch number
            periodic.sort(key=lambda x: int(os.path.basename(x)))
            paths["periodic"] = periodic
        elif dir_ == "checkpoints":
            # Legacy single checkpoint format - use as fallback for best_mse if not set
            legacy_path = _get_latest_checkpoint(full_path)
            if legacy_path is not None and paths["best_mse"] is None:
                paths["best_mse"] = legacy_path

    # Set 'best' to best_mse for backward compatibility
    paths["best"] = paths["best_mse"]

    # Validate we found something
    has_any = (
        paths["final"] is not None
        or paths["best_mse"] is not None
        or paths["best_mae"] is not None
        or paths["best_hybrid"] is not None
    )

    if not has_any:
        raise ValueError(f"No final model or checkpoints found in {model_path}")

    return paths


def get_model_save_paths_legacy(model_path):
    """
    Legacy function for backward compatibility.

    Returns [final_model_path, best_model_path] tuple.
    Use get_model_save_paths() for new code.
    """
    paths = get_model_save_paths(model_path)
    return [paths["final"], paths["best"]]


def get_model_paths(cfg) -> dict:
    """
    Get the model paths from the config file.

    Args:
        cfg: Configuration object with model_save_path

    Returns:
        Dict with checkpoint paths (see get_model_save_paths for keys)
    """
    model_path = cfg.model_save_path
    if model_path is None:
        raise ValueError("No model save path found in config file")

    if not os.path.exists(model_path):
        raise ValueError(f"Model save path {model_path} does not exist")

    # Get the model save paths (returns dict with all checkpoint types)
    paths = get_model_save_paths(model_path)

    return paths


# Convert ndarray to list if necessary
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray | jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj
