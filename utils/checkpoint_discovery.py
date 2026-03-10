"""
Checkpoint Discovery Utility

Functions for discovering and selecting checkpoints across multirun experiments.
Supports the multi-metric checkpoint structure (MSE, MAE, hybrid).
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)

MetricType = Literal["best_mse", "best_mae", "best_hybrid", "final"]


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""

    path: str
    metric_type: MetricType
    epoch: int
    experiment_dir: str
    val_mse: float | None = None
    val_mae: float | None = None
    val_hybrid: float | None = None

    @property
    def experiment_name(self) -> str:
        """Extract experiment name from path (e.g., '+experiment=phase1_vj8')."""
        # Look for experiment identifier in path components
        parts = self.experiment_dir.split(os.sep)
        for part in parts:
            if "+experiment=" in part or "experiment=" in part:
                return part.split("=")[-1]
        # Fallback to directory name
        return os.path.basename(self.experiment_dir)


def load_checkpoint_metrics(checkpoint_path: str) -> dict:
    """
    Load metrics from a checkpoint without loading full state.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dict with available metrics (val_mse, val_mae, etc.)
    """
    try:
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(checkpoint_path)

        metrics = {}
        if "metrics" in restored:
            raw_metrics = restored["metrics"]
            # Handle both flat and nested metric formats
            for key in [
                "val_mse",
                "val_mae",
                "val_RMSE",
                "val_mse_scaled_rel_ws",
                "val_mae_scaled_rel_ws",
            ]:
                if key in raw_metrics:
                    val = raw_metrics[key]
                    # Convert JAX arrays to float
                    if hasattr(val, "item"):
                        val = float(val.item())
                    elif hasattr(val, "__float__"):
                        val = float(val)
                    metrics[key] = val

        return metrics
    except Exception as e:
        logger.warning(f"Failed to load metrics from {checkpoint_path}: {e}")
        return {}


def discover_experiment_checkpoints(
    multirun_dir: str,
    metric_type: MetricType = "best_mse",
) -> list[CheckpointInfo]:
    """
    Discover all checkpoints of a given type in a Hydra multirun directory.

    Scans the multirun directory structure:
        multirun_dir/
            +experiment=exp1/
                model/
                    checkpoints_best_mse/
                    checkpoints_best_mae/
                    ...
            +experiment=exp2/
                model/
                    ...

    Args:
        multirun_dir: Path to Hydra multirun output directory
        metric_type: Type of checkpoint to find ("best_mse", "best_mae", "best_hybrid", "final")

    Returns:
        List of CheckpointInfo objects for found checkpoints
    """
    checkpoints = []

    if not os.path.exists(multirun_dir):
        logger.warning(f"Multirun directory does not exist: {multirun_dir}")
        return checkpoints

    # Scan experiment directories
    for exp_name in os.listdir(multirun_dir):
        exp_dir = os.path.join(multirun_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # Look for model directory
        model_dir = os.path.join(exp_dir, "model")
        if not os.path.exists(model_dir):
            continue

        # Find checkpoint based on metric_type
        if metric_type == "final":
            # Look for final_e_* directory
            for item in os.listdir(model_dir):
                if item.startswith("final_e_"):
                    checkpoint_path = os.path.join(model_dir, item)
                    try:
                        epoch = int(item.split("_")[-1])
                        metrics = load_checkpoint_metrics(checkpoint_path)
                        info = CheckpointInfo(
                            path=checkpoint_path,
                            metric_type="final",
                            epoch=epoch,
                            experiment_dir=exp_dir,
                            val_mse=metrics.get("val_mse"),
                            val_mae=metrics.get("val_mae"),
                        )
                        checkpoints.append(info)
                    except (ValueError, Exception) as e:
                        logger.warning(f"Failed to parse checkpoint {checkpoint_path}: {e}")
        else:
            # Look for checkpoints_best_* directory
            checkpoint_dir = os.path.join(model_dir, f"checkpoints_{metric_type}")
            if not os.path.exists(checkpoint_dir):
                continue

            # Get latest checkpoint in this directory
            latest_epoch = -1
            latest_path = None
            for item in os.listdir(checkpoint_dir):
                try:
                    epoch = int(item)
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_path = os.path.join(checkpoint_dir, item)
                except ValueError:
                    continue

            if latest_path is not None:
                metrics = load_checkpoint_metrics(latest_path)
                info = CheckpointInfo(
                    path=latest_path,
                    metric_type=metric_type,
                    epoch=latest_epoch,
                    experiment_dir=exp_dir,
                    val_mse=metrics.get("val_mse"),
                    val_mae=metrics.get("val_mae"),
                )
                checkpoints.append(info)

    return checkpoints


def find_best_checkpoint(
    multirun_dir: str,
    metric_type: MetricType = "best_mse",
    selection_metric: str = "val_mse",
) -> CheckpointInfo | None:
    """
    Find the best checkpoint across all experiments in a multirun.

    Args:
        multirun_dir: Path to Hydra multirun output directory
        metric_type: Type of checkpoint to search ("best_mse", "best_mae", "best_hybrid", "final")
        selection_metric: Metric to use for selection ("val_mse", "val_mae")

    Returns:
        CheckpointInfo for best checkpoint, or None if not found
    """
    checkpoints = discover_experiment_checkpoints(multirun_dir, metric_type)

    if not checkpoints:
        return None

    # Filter to checkpoints with the selection metric
    valid_checkpoints = []
    for ckpt in checkpoints:
        metric_val = getattr(ckpt, selection_metric.replace("val_", "val_"), None)
        if metric_val is not None:
            valid_checkpoints.append((ckpt, metric_val))

    if not valid_checkpoints:
        # Fall back to any checkpoint if metrics aren't available
        logger.warning(f"No checkpoints have {selection_metric} metric, returning first found")
        return checkpoints[0] if checkpoints else None

    # Sort by metric (lower is better)
    valid_checkpoints.sort(key=lambda x: x[1])

    best_ckpt, best_metric = valid_checkpoints[0]
    logger.info(
        f"Best checkpoint by {selection_metric}: {best_ckpt.experiment_name} "
        f"(epoch {best_ckpt.epoch}, {selection_metric}={best_metric:.6f})"
    )

    return best_ckpt


def print_checkpoint_summary(multirun_dir: str) -> None:
    """
    Print a summary of all checkpoints in a multirun directory.

    Shows metrics for each experiment to help with informed selection.

    Args:
        multirun_dir: Path to Hydra multirun output directory
    """
    print(f"\n{'=' * 80}")
    print(f"Checkpoint Summary: {multirun_dir}")
    print(f"{'=' * 80}\n")

    # Collect all checkpoints by experiment
    experiments = {}

    all_metric_types: list[MetricType] = ["best_mse", "best_mae", "best_hybrid", "final"]
    for metric_type in all_metric_types:
        checkpoints = discover_experiment_checkpoints(multirun_dir, metric_type)
        for ckpt in checkpoints:
            exp_name = ckpt.experiment_name
            if exp_name not in experiments:
                experiments[exp_name] = {}
            experiments[exp_name][metric_type] = ckpt

    if not experiments:
        print("No checkpoints found.")
        return

    # Print header
    print(f"{'Experiment':<40} {'Type':<12} {'Epoch':<8} {'Val MSE':<15} {'Val MAE':<15}")
    print("-" * 90)

    # Print each experiment
    for exp_name in sorted(experiments.keys()):
        exp_ckpts = experiments[exp_name]
        first = True

        for metric_type in all_metric_types:
            if metric_type in exp_ckpts:
                ckpt = exp_ckpts[metric_type]
                mse_str = f"{ckpt.val_mse:.8f}" if ckpt.val_mse is not None else "N/A"
                mae_str = f"{ckpt.val_mae:.8f}" if ckpt.val_mae is not None else "N/A"

                if first:
                    print(
                        f"{exp_name:<40} {metric_type:<12} {ckpt.epoch:<8} {mse_str:<15} {mae_str:<15}"
                    )
                    first = False
                else:
                    print(f"{'':<40} {metric_type:<12} {ckpt.epoch:<8} {mse_str:<15} {mae_str:<15}")

        print()  # Blank line between experiments

    # Print best overall
    print("-" * 90)
    print("\nBest checkpoints overall:\n")

    best_only_types: list[MetricType] = ["best_mse", "best_mae"]
    for best_metric_type in best_only_types:
        best = find_best_checkpoint(
            multirun_dir, best_metric_type, f"val_{best_metric_type.split('_')[1]}"
        )
        if best:
            metric_val = getattr(best, f"val_{best_metric_type.split('_')[1]}", "N/A")
            if isinstance(metric_val, float):
                metric_val = f"{metric_val:.8f}"
            print(
                f"  {best_metric_type}: {best.experiment_name} (epoch {best.epoch}, val={metric_val})"
            )
            print(f"    Path: {best.path}")


def export_checkpoint_summary(multirun_dir: str, output_path: str | None = None) -> str:
    """
    Export checkpoint summary to JSON file.

    Args:
        multirun_dir: Path to Hydra multirun output directory
        output_path: Optional output path (defaults to multirun_dir/checkpoint_summary.json)

    Returns:
        Path to created JSON file
    """
    if output_path is None:
        output_path = os.path.join(multirun_dir, "checkpoint_summary.json")

    from typing import Any

    summary: dict[str, Any] = {
        "multirun_dir": multirun_dir,
        "experiments": {},
        "best_overall": {},
    }

    # Collect all checkpoints
    all_metric_types: list[MetricType] = ["best_mse", "best_mae", "best_hybrid", "final"]
    for export_metric_type in all_metric_types:
        checkpoints = discover_experiment_checkpoints(multirun_dir, export_metric_type)
        for ckpt in checkpoints:
            exp_name = ckpt.experiment_name
            if exp_name not in summary["experiments"]:
                summary["experiments"][exp_name] = {}

            summary["experiments"][exp_name][export_metric_type] = {
                "path": ckpt.path,
                "epoch": ckpt.epoch,
                "val_mse": ckpt.val_mse,
                "val_mae": ckpt.val_mae,
            }

    # Find best overall
    best_only_types: list[MetricType] = ["best_mse", "best_mae"]
    for best_metric_type in best_only_types:
        best = find_best_checkpoint(
            multirun_dir, best_metric_type, f"val_{best_metric_type.split('_')[1]}"
        )
        if best:
            summary["best_overall"][best_metric_type] = {
                "experiment": best.experiment_name,
                "path": best.path,
                "epoch": best.epoch,
                "val_mse": best.val_mse,
                "val_mae": best.val_mae,
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Checkpoint summary exported to: {output_path}")
    return output_path
