"""
Sophia HPC path utilities for mapping WandB paths to local mount points.

WandB stores experiment paths as they appear on Sophia (remote HPC).
These utilities convert between remote and local mount paths for
offline analysis of experiment results.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Sophia HPC path mapping
SOPHIA_REMOTE_PREFIX = "/work/users/jpsch/gno/"
SOPHIA_LOCAL_MOUNT = Path("/home/jpsch/Documents/Sophia_work/gno/")


def convert_wandb_path_to_local(wandb_path: str) -> tuple[Path | None, str]:
    """
    Convert a WandB run path (from Sophia) to local mount point.

    Args:
        wandb_path: Path as stored in WandB (e.g., /work/users/jpsch/gno/outputs/...)

    Returns:
        Tuple of (local_path or None, status_message)
        - If Sophia is mounted and path exists: (Path, "available")
        - If Sophia is mounted but path doesn't exist: (Path, "path_not_found")
        - If Sophia is not mounted: (None, "sophia_not_mounted")
        - If path doesn't match expected format: (None, "invalid_path_format")
    """
    if not wandb_path.startswith(SOPHIA_REMOTE_PREFIX):
        return None, "invalid_path_format"

    relative_path = wandb_path[len(SOPHIA_REMOTE_PREFIX) :]
    local_path = SOPHIA_LOCAL_MOUNT / relative_path

    if not SOPHIA_LOCAL_MOUNT.exists():
        return None, "sophia_not_mounted"

    if local_path.exists():
        return local_path, "available"
    else:
        return local_path, "path_not_found"


def local_path_to_sophia_path(local_path: Path | str) -> str:
    """
    Convert a local mount path back to Sophia remote path.

    Args:
        local_path: Path on local mount (e.g., /home/jpsch/Documents/Sophia_work/gno/...)

    Returns:
        Sophia remote path (e.g., /work/users/jpsch/gno/...)
    """
    local_str = str(local_path)
    local_mount_str = str(SOPHIA_LOCAL_MOUNT)

    if local_str.startswith(local_mount_str):
        relative = local_str[len(local_mount_str) :]
        if relative.startswith("/"):
            relative = relative[1:]
        return SOPHIA_REMOTE_PREFIX + relative

    return local_str


def find_checkpoint_step_dir(checkpoint_parent: Path) -> Path | None:
    """
    Find the step-numbered subdirectory inside a checkpoint folder.

    Orbax CheckpointManager saves checkpoints in step-numbered subdirectories
    (e.g., checkpoints_best_mse/2961/) where the actual _CHECKPOINT_METADATA lives.

    Args:
        checkpoint_parent: Path to checkpoint parent dir (e.g., checkpoints_best_mse/)

    Returns:
        Path to the step directory containing the actual checkpoint, or None if not found
    """
    if not checkpoint_parent.exists():
        return None

    step_dirs = []
    for item in checkpoint_parent.iterdir():
        if item.is_dir():
            try:
                int(item.name)
                step_dirs.append(item)
            except ValueError:
                continue

    if not step_dirs:
        return None

    step_dirs.sort(key=lambda x: int(x.name), reverse=True)
    return step_dirs[0]


def get_checkpoint_paths(experiment_path: Path) -> dict[str, Path | None]:
    """
    Get checkpoint paths for an experiment, including step subdirectories.

    Orbax checkpoints are stored in step-numbered subdirectories like:
        checkpoints_best_mse/2961/
    This function finds the actual checkpoint path including the step number.

    Args:
        experiment_path: Path to the experiment directory

    Returns:
        Dictionary with checkpoint paths (or None if not found)
    """
    model_dir = experiment_path / "model"
    if not model_dir.exists():
        return {
            "best_mse": None,
            "best_mae": None,
            "best_hybrid": None,
            "final": None,
        }

    checkpoints = {}

    best_mse_dir = model_dir / "checkpoints_best_mse"
    checkpoints["best_mse"] = find_checkpoint_step_dir(best_mse_dir)

    best_mae_dir = model_dir / "checkpoints_best_mae"
    checkpoints["best_mae"] = find_checkpoint_step_dir(best_mae_dir)

    best_hybrid_dir = model_dir / "checkpoints_best_hybrid"
    checkpoints["best_hybrid"] = find_checkpoint_step_dir(best_hybrid_dir)

    final_dirs = list(model_dir.glob("final_*"))
    checkpoints["final"] = final_dirs[0] if final_dirs else None

    return checkpoints
