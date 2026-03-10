"""
WandB data fetching utilities with CSV caching.

Provides functions to fetch experiment runs from Weights & Biases
with local CSV caching for offline access and faster repeated access.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# WandB entity for this project
ENTITY = "jpsch-dtu-wind-and-energy-systems"

# Project name mapping for multi-phase transfer learning experiments
WANDB_PROJECTS = {
    "phase0": "GNO_phase0_baselines",
    "phase1": "GNO_phase1_arch_search",
    "phase1_global": "GNO_phase1_arch_search_global",
    "phase2": "GNO_phase2_techniques",
    "phase2_global": "GNO_phase2_global",
    "phase3": "GNO_phase3_hero",
    "phase3_global": "GNO_phase3_hero_global",
}


def extract_summary_metrics(run) -> dict[str, Any]:
    """
    Extract key metrics from a WandB run summary.

    Args:
        run: WandB Run object

    Returns:
        Dictionary with extracted metrics:
        - train_mse: Training MSE at end of training
        - val_mse: Validation MSE at best checkpoint
        - val_mae: Validation MAE at best checkpoint
        - best_val_mse: Best validation MSE during training
        - best_val_mae: Best validation MAE during training
        - final_epoch: Final training epoch
    """
    summary = run.summary._json_dict

    # Common metric keys in GNO training
    # Note: WandB uses "val/loss(mse)" and "train/loss" for MSE values
    metrics = {
        "train_mse": summary.get("train/loss") or summary.get("train/mse"),
        "val_mse": summary.get("val/loss(mse)") or summary.get("val/mse"),
        "val_mae": summary.get("val/mae"),
        "val_rmse": summary.get("val/rmse"),
        "best_val_mse": summary.get("best_val_mse"),
        "best_val_mae": summary.get("best_val_mae"),
        "final_epoch": summary.get("epoch") or summary.get("_step"),
        "runtime_seconds": summary.get("_runtime") or summary.get("_wandb", {}).get("runtime"),
    }

    # Handle alternate metric names
    if metrics["val_mse"] is None:
        metrics["val_mse"] = summary.get("validation/mse") or summary.get("val_mse")
    if metrics["val_mae"] is None:
        metrics["val_mae"] = summary.get("validation/mae") or summary.get("val_mae")

    return metrics


def extract_config(run, exclude_prefixes: tuple[str, ...] = ("_",)) -> dict[str, Any]:
    """
    Extract configuration from a WandB run.

    Args:
        run: WandB Run object
        exclude_prefixes: Tuple of key prefixes to exclude (default: underscore-prefixed)

    Returns:
        Dictionary with configuration values, excluding internal keys
    """
    config = {}
    for key, value in run.config.items():
        if not any(key.startswith(prefix) for prefix in exclude_prefixes):
            config[key] = value
    return config


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary with dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fetch_wandb_runs(
    project: str,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    entity: str = ENTITY,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    Fetch runs from WandB with CSV caching.

    Args:
        project: WandB project name (can be short name like "phase1" or full name)
        cache_dir: Directory to store CSV cache (creates if doesn't exist)
        refresh: If True, force refetch from WandB even if cache exists
        entity: WandB entity (default: ENTITY constant)
        filters: Optional WandB API filters (e.g., {"state": "finished"})

    Returns:
        DataFrame with run data including:
        - Run metadata (id, name, state, created_at, etc.)
        - Summary metrics (val_mse, val_mae, etc.)
        - Flattened configuration values

    Raises:
        ImportError: If wandb is not installed
        RuntimeError: If both WandB fetch fails and no cache exists
    """
    # Resolve project name if short form used
    if project in WANDB_PROJECTS:
        project = WANDB_PROJECTS[project]

    # Setup cache path
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{project}_runs.csv"
    else:
        cache_path = None

    # Try to load from cache if not refreshing
    if not refresh and cache_path is not None and cache_path.exists():
        logger.info(f"Loading cached runs from {cache_path}")
        return pd.read_csv(cache_path)

    # Fetch from WandB
    try:
        import wandb

        api = wandb.Api()
        logger.info(f"Fetching runs from WandB: {entity}/{project}")

        runs = api.runs(f"{entity}/{project}", filters=filters or {})

        records = []
        for run in runs:
            record = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "tags": ",".join(run.tags) if run.tags else "",
            }

            # Add summary metrics
            metrics = extract_summary_metrics(run)
            record.update(metrics)

            # Add flattened config
            config = extract_config(run)
            flat_config = _flatten_dict(config, parent_key="config")
            record.update(flat_config)

            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} runs from WandB")

        # Save to cache
        if cache_path is not None:
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached runs to {cache_path}")

        return df

    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        if cache_path is not None and cache_path.exists():
            logger.warning(f"Falling back to cached data from {cache_path}")
            return pd.read_csv(cache_path)
        raise

    except Exception as e:
        logger.error(f"Failed to fetch from WandB: {e}")
        if cache_path is not None and cache_path.exists():
            logger.warning(f"Falling back to cached data from {cache_path}")
            return pd.read_csv(cache_path)
        raise RuntimeError(f"Failed to fetch from WandB and no cache exists at {cache_path}") from e


def get_run_history(
    run_id: str,
    project: str,
    entity: str = ENTITY,
    keys: list[str] | None = None,
    samples: int | None = None,
) -> pd.DataFrame:
    """
    Fetch training history for a specific run.

    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity
        keys: Optional list of metric keys to fetch (default: all)
        samples: Number of samples to fetch. If None, uses run.historyLineCount
                 to fetch all data points (avoids WandB's default 500-sample downsampling).

    Returns:
        DataFrame with training history (epoch/step as index)
    """
    import wandb

    # Resolve project name if short form used
    if project in WANDB_PROJECTS:
        project = WANDB_PROJECTS[project]

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Default to all samples to avoid WandB's 500-sample downsampling
    if samples is None:
        samples = getattr(run, "historyLineCount", 10000)

    kwargs: dict[str, Any] = {"samples": samples}
    if keys:
        kwargs["keys"] = keys

    return run.history(**kwargs)


def enrich_with_best_metrics(
    df: pd.DataFrame,
    project: str,
    entity: str = ENTITY,
) -> pd.DataFrame:
    """
    Overwrite val_mse/val_mae/val_rmse with best (min) values from run history.

    WandB summary stores the last-logged value by default, not the best.
    This function fetches the full history for each run and replaces the
    summary values with the true minimums.

    Args:
        df: DataFrame from fetch_wandb_runs (must have 'run_id' column)
        project: WandB project name (short or full form)
        entity: WandB entity

    Returns:
        DataFrame with val_mse, val_mae, val_rmse overwritten by min values
    """
    import wandb

    # Resolve project name
    resolved_project = WANDB_PROJECTS.get(project, project)

    api = wandb.Api()
    df = df.copy()

    metrics_map = {
        "val/loss(mse)": "val_mse",
        "val/mae": "val_mae",
        "val/rmse": "val_rmse",
    }
    history_keys = list(metrics_map.keys())

    for idx, row in df.iterrows():
        run_id = row["run_id"]
        try:
            run = api.run(f"{entity}/{resolved_project}/{run_id}")
            num_samples = getattr(run, "historyLineCount", 10000)
            history = run.history(keys=history_keys, samples=num_samples)

            for wandb_key, df_col in metrics_map.items():
                if wandb_key in history.columns:
                    series = history[wandb_key].dropna()
                    if not series.empty:
                        best_val = float(series.min())
                        old_val = row.get(df_col)
                        if pd.notna(old_val) and old_val != best_val:
                            logger.info(
                                f"  {row.get('run_name', run_id)}: "
                                f"{df_col} {old_val:.6e} -> {best_val:.6e}"
                            )
                        df.at[idx, df_col] = best_val

        except Exception as e:
            logger.warning(f"Failed to fetch history for run {run_id}: {e}")

    logger.info(f"Enriched {len(df)} runs with best (min) validation metrics")
    return df


def load_test_metrics(experiment_path: Path) -> dict[str, float] | None:
    """
    Load test metrics from error_metrics_all.json.

    Args:
        experiment_path: Path to the experiment directory (containing 'model' folder)

    Returns:
        Dictionary with flattened test metrics, or None if file not found.
        Keys are formatted as 'test_{checkpoint}_{metric}' (e.g., 'test_best_mse_mse').
        If metrics are per-feature dicts, values are averaged.
    """
    metrics_path = experiment_path / "model" / "error_metrics_all.json"
    if not metrics_path.exists():
        logger.debug(f"Test metrics not found: {metrics_path}")
        return None

    try:
        import json

        with open(metrics_path) as f:
            raw_metrics = json.load(f)

        # Flatten metrics with simplified keys
        # e.g., "test/best_mse/mse" -> "test_best_mse_mse"
        flattened = {}
        for key, value in raw_metrics.items():
            # Convert key format
            flat_key = key.replace("/", "_")

            # Handle dict values (per-feature metrics) by averaging
            if isinstance(value, dict):
                # Average across features
                values = list(value.values())
                if values and all(isinstance(v, int | float) for v in values):
                    flattened[flat_key] = sum(values) / len(values)
            elif isinstance(value, list):
                # convert_ndarray() produces single-element lists from 1D arrays
                if len(value) == 1 and isinstance(value[0], int | float):
                    flattened[flat_key] = float(value[0])
                elif value and all(isinstance(v, int | float) for v in value):
                    flattened[flat_key] = sum(value) / len(value)
            elif isinstance(value, int | float):
                flattened[flat_key] = float(value)

        return flattened if flattened else None

    except Exception as e:
        logger.warning(f"Failed to load test metrics from {metrics_path}: {e}")
        return None


def extract_convergence_from_history(
    history: pd.DataFrame,
    metric: str = "val/loss(mse)",
    window_size: int = 5,
) -> dict[str, Any]:
    """
    Extract convergence metrics from training history.

    Args:
        history: DataFrame with training history (from get_run_history)
        metric: Metric column to analyze
        window_size: Window size for stability calculation

    Returns:
        Dictionary with convergence metrics:
        - best_epoch: Epoch with best metric value
        - best_value: Best metric value
        - final_epoch: Final training epoch
        - final_value: Final metric value
        - stability_std: Standard deviation of last 'window_size' values
        - converged: Whether training appears to have converged
    """
    if metric not in history.columns:
        logger.warning(f"Metric '{metric}' not found in history")
        return {}

    # Filter to rows where metric is available
    valid_history = history[history[metric].notna()].copy()
    if valid_history.empty:
        return {}

    # Get epoch column (prefer _step, fallback to index)
    if "_step" in valid_history.columns:
        epochs = valid_history["_step"].values
    else:
        epochs = valid_history.index.values

    values = valid_history[metric].values

    # Find best value
    best_idx = values.argmin()
    best_epoch = int(epochs[best_idx])
    best_value = float(values[best_idx])

    # Final values
    final_epoch = int(epochs[-1])
    final_value = float(values[-1])

    # Stability: std of last N values
    last_values = values[-window_size:] if len(values) >= window_size else values
    stability_std = float(last_values.std())

    # Convergence heuristic: std < 10% of final value and best is in last 20%
    relative_std = stability_std / (final_value + 1e-10)
    best_in_last_20pct = best_idx >= 0.8 * len(values)
    converged = relative_std < 0.1 and best_in_last_20pct

    return {
        "best_epoch": best_epoch,
        "best_value": best_value,
        "final_epoch": final_epoch,
        "final_value": final_value,
        "stability_std": stability_std,
        "converged": converged,
    }
