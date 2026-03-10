"""
Phase 1: Architecture Search Analysis and Selection (Global Conditioning).

This script analyzes the results of the Phase 1 architecture search experiments
with global conditioning enabled (WindfarmGNO_probe, use_global_conditioning=True)
and generates tables, figures, and a selection JSON for Phase 2.

WandB Project: GNO_phase1_arch_search_global

Experiments analyzed:
- Vj8, S1-S3, L1-L3 with dropout + layernorm
- Learning rate variants (lr001)

Outputs (named by configuration, e.g., phase1_global_2500layouts_*):
- outputs/phase1_global_{config}_model_configs.tex: Architecture parameters table
- outputs/phase1_global_{config}_results.tex: Results table ranked by val_MSE
- figures/phase1_global_{config}_architecture_comparison.pdf: Bar chart comparison
- outputs/phase1_global_{config}_selection.json: Selection output for Phase 2

Usage:
    python phase1_architecture.py [--config CONFIG] [--refresh] [--select EXPERIMENT_ID]

Options:
    --config: Configuration to use (default: global_2500layouts)
    --refresh: Force refetch from WandB (ignore cache)
    --select: Manually select experiment (default: best val_MSE)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import matplotlib_set_rcparams  # noqa: E402
from utils.reporting import (  # noqa: E402
    SOPHIA_LOCAL_MOUNT,
    WANDB_PROJECTS,
    convert_wandb_path_to_local,
    create_latex_table,
    enrich_with_best_metrics,
    fetch_wandb_runs,
    get_checkpoint_paths,
    get_run_history,
    load_test_metrics,
    local_path_to_sophia_path,
    save_dataframe_csv,
    save_table,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Run configurations for different experiment sets
# =============================================================================
RUN_CONFIGS = {
    "global_2500layouts": {
        "name": "Global Context (2500 layouts)",
        "wandb_project": "phase1_global",
        "data_path_filter": "turbopark_2500layouts",
        "output_prefix": "phase1_global_2500layouts",
    },
}

# Default configuration
DEFAULT_CONFIG = "global_2500layouts"

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
CACHE_DIR = SCRIPT_DIR / "cache"

# Create directories
OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# GitLab URL for table warnings
GITLAB_URL = "https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/article_2/phase1_architecture.py"

# Model config column mapping (WandB config keys -> display names)
MODEL_CONFIG_COLUMNS = {
    "config.model.latent_size": "Latent",
    "config.model.hidden_layer_size": "Hidden",
    "config.model.num_mlp_layers": "MLP Layers",
    "config.model.wt_message_passing_steps": "WT MP",
    "config.model.probe_message_passing_steps": "Probe MP",
    "config.model.decoder_hidden_layer_size": "Dec Hidden",
    "config.model.num_decoder_layers": "Dec Layers",
}

# Known model ID patterns for parsing experiment names
KNOWN_MODEL_IDS = ["vj8", "s1", "s2", "s3", "l1", "l2", "l3"]

# Regularization display names
REG_DISPLAY = {
    "dropout_layernorm": "DO+LN",
    "lr001": "DO+LN (lr001)",
    "dropout": "DO",
    "unknown": "?",
}


def _format_regularization(reg: str) -> str:
    """Convert regularization key to display string."""
    return REG_DISPLAY.get(reg, reg)


def parse_experiment_id(run_name: str) -> tuple[str, str]:
    """
    Parse run name to extract model ID and regularization strategy.

    Expected formats:
    - "vj8_dropout_layernorm" -> ("vj8", "dropout_layernorm")
    - "s1_dropout" -> ("s1", "dropout")
    - "phase1_l2_dropout_layernorm" -> ("l2", "dropout_layernorm")

    Returns:
        Tuple of (model_id, regularization)
    """
    name_lower = run_name.lower()

    # Try to find model ID
    model_id = None
    for mid in KNOWN_MODEL_IDS:
        if mid in name_lower:
            model_id = mid
            break

    if model_id is None:
        model_id = "unknown"

    # Determine regularization
    # Check lr001 first (these runs use dropout+layernorm but different learning rate)
    if "lr001" in name_lower:
        regularization = "lr001"
    elif "layernorm" in name_lower:
        regularization = "dropout_layernorm"
    elif "dropout" in name_lower:
        regularization = "dropout"
    else:
        regularization = "unknown"

    return model_id, regularization


def create_model_configs_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create table of model architecture configurations extracted from WandB.

    Args:
        df: DataFrame with run data including config columns

    Returns:
        DataFrame with architecture parameters
    """
    records = []
    for _, row in df.iterrows():
        record = {
            "Model": row["model_id"].upper(),
            "Reg.": _format_regularization(row["regularization"]),
        }

        # Extract model config columns
        for wandb_col, display_name in MODEL_CONFIG_COLUMNS.items():
            if wandb_col in df.columns:
                value = row[wandb_col]
                # Convert to int if it's a whole number
                if pd.notna(value) and isinstance(value, float) and value == int(value):
                    value = int(value)
                record[display_name] = value

        records.append(record)

    result_df = pd.DataFrame(records)

    # Sort by model ID for consistent ordering
    if "Model" in result_df.columns:
        result_df = result_df.sort_values("Model")

    return result_df


def create_results_table(df: pd.DataFrame, include_test: bool = False) -> pd.DataFrame:
    """
    Create results table ranked by MSE with all available metrics.

    When include_test=True and test metrics are available, uses test metrics
    instead of validation metrics. Column names reflect the data source
    (e.g., "Test MSE" vs "Val MSE").

    Args:
        df: DataFrame with run data and metrics
        include_test: Use test metrics when available

    Returns:
        DataFrame with results, sorted by MSE, including rank
    """
    # Determine whether to use test or validation metrics
    use_test = (
        include_test
        and "test_mse" in df.columns
        and "test_mae" in df.columns
        and df["test_mse"].notna().any()
    )

    if use_test:
        mse_col, mae_col = "test_mse", "test_mae"
        rmse_col = "test_rmse" if "test_rmse" in df.columns else None
        prefix = "Test"
    else:
        mse_col, mae_col = "val_mse", "val_mae"
        rmse_col = "val_rmse" if "val_rmse" in df.columns else None
        prefix = "Val"

    # Select columns that have data
    cols_to_select = ["model_id", "regularization", mse_col, mae_col]
    if rmse_col and rmse_col in df.columns and df[rmse_col].notna().any():
        cols_to_select.append(rmse_col)

    results = df[cols_to_select].copy()

    # Compute normalized hybrid metric for comparison
    # Using max values as baseline (so all values are <= 1.0)
    mse_max = results[mse_col].max()
    mae_max = results[mae_col].max()
    if mse_max > 0 and mae_max > 0:
        results["hybrid"] = ((results[mse_col] / mse_max) * (results[mae_col] / mae_max)).apply(
            lambda x: x**0.5
        )

    # Sort by MSE
    results = results.sort_values(mse_col)

    # Add rank column
    results.insert(0, "Rank", range(1, len(results) + 1))

    # Rename columns for display
    rename_map = {
        "model_id": "Model",
        "regularization": "Reg.",
        mse_col: f"{prefix} MSE",
        mae_col: f"{prefix} MAE",
        "hybrid": "Hybrid*",
    }
    if rmse_col and rmse_col in results.columns:
        rename_map[rmse_col] = f"{prefix} RMSE"
    results = results.rename(columns=rename_map)

    # Uppercase model IDs and format regularization
    results["Model"] = results["Model"].str.upper()
    results["Reg."] = results["Reg."].map(_format_regularization)

    return results


def create_compact_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create compact results table with Val and Test RMSE + MAE side-by-side.

    RMSE is computed as sqrt(MSE) from the existing MSE columns.
    Sorted by Test RMSE ascending.

    Args:
        df: DataFrame with run data including val_mse, val_mae, test_mse, test_mae

    Returns:
        DataFrame with columns: Rank, Model, Reg., Val RMSE, Val MAE, Test RMSE, Test MAE
    """
    results = pd.DataFrame()
    results["model_id"] = df["model_id"]
    results["regularization"] = df["regularization"]
    results["Val RMSE"] = np.sqrt(df["val_mse"].astype(float))
    results["Val MAE"] = df["val_mae"].astype(float)
    results["Test RMSE"] = np.sqrt(df["test_mse"].astype(float))
    results["Test MAE"] = df["test_mae"].astype(float)

    # Sort by Test RMSE ascending
    results = results.sort_values("Test RMSE")
    results.insert(0, "Rank", range(1, len(results) + 1))

    # Format display columns
    results = results.rename(columns={"model_id": "Model", "regularization": "Reg."})
    results["Model"] = results["Model"].str.upper()
    results["Reg."] = results["Reg."].map(_format_regularization)

    return results


def add_test_and_param_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add test metrics from JSON files and parameter counts from checkpoints.

    Args:
        df: DataFrame with run data (must include run_name column with WandB paths)

    Returns:
        DataFrame with additional columns:
        - test_mse: Test MSE from best_mse checkpoint
        - test_mae: Test MAE from best_mse checkpoint
        - param_count: Total parameter count (if checkpoint available)
    """
    df = df.copy()

    # Initialize new columns
    df["test_mse"] = None
    df["test_mae"] = None
    df["param_count"] = None

    for idx, row in df.iterrows():
        wandb_path = row["run_name"]
        local_path, path_status = convert_wandb_path_to_local(wandb_path)

        if local_path is None or path_status != "available":
            continue

        # Load test metrics
        test_metrics = load_test_metrics(local_path)
        if test_metrics:
            # Prefer best_mse checkpoint metrics, fallback to final
            if "test_best_mse_mse" in test_metrics:
                df.at[idx, "test_mse"] = test_metrics["test_best_mse_mse"]
            elif "test_final_mse" in test_metrics:
                df.at[idx, "test_mse"] = test_metrics["test_final_mse"]

            if "test_best_mse_mae" in test_metrics:
                df.at[idx, "test_mae"] = test_metrics["test_best_mse_mae"]
            elif "test_final_mae" in test_metrics:
                df.at[idx, "test_mae"] = test_metrics["test_final_mae"]

    return df


def create_comprehensive_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive results table with test metrics and generalization gap.

    Columns: Rank, Model, Reg., Params(K), Val MSE, Val MAE, Test MSE, Test MAE,
             Hybrid, Gen Gap MSE, Gen Gap MAE

    Args:
        df: DataFrame with run data including test metrics

    Returns:
        DataFrame formatted for display/export
    """
    # Select and compute columns
    results = pd.DataFrame()
    results["model_id"] = df["model_id"]
    results["regularization"] = df["regularization"]
    results["val_mse"] = df["val_mse"]
    results["val_mae"] = df["val_mae"]
    results["test_mse"] = df.get("test_mse")
    results["test_mae"] = df.get("test_mae")

    # Compute normalized hybrid metric
    mse_max = results["val_mse"].max()
    mae_max = results["val_mae"].max()
    if mse_max > 0 and mae_max > 0:
        results["hybrid"] = ((results["val_mse"] / mse_max) * (results["val_mae"] / mae_max)).apply(
            lambda x: x**0.5
        )

    # Compute generalization gaps (Test - Val)
    results["gap_mse"] = results["test_mse"] - results["val_mse"]
    results["gap_mae"] = results["test_mae"] - results["val_mae"]

    # Sort by val_mse and add rank
    results = results.sort_values("val_mse")
    results.insert(0, "Rank", range(1, len(results) + 1))

    # Rename columns for display
    rename_map = {
        "model_id": "Model",
        "regularization": "Reg.",
        "val_mse": "Val MSE",
        "val_mae": "Val MAE",
        "test_mse": "Test MSE",
        "test_mae": "Test MAE",
        "hybrid": "Hybrid",
        "gap_mse": "Gap MSE",
        "gap_mae": "Gap MAE",
    }
    results = results.rename(columns=rename_map)

    # Format model IDs and regularization
    results["Model"] = results["Model"].str.upper()
    results["Reg."] = results["Reg."].map(_format_regularization)

    return results


def plot_architecture_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create bar chart comparing architectures by val_MSE.

    Args:
        df: DataFrame with run data
        output_path: Path to save the figure
    """
    matplotlib_set_rcparams("paper")

    # Prepare data
    plot_df = df.copy()
    plot_df["label"] = (
        plot_df["model_id"].str.upper() + "\n" + plot_df["regularization"].str.replace("_", "\n")
    )

    # Sort by val_mse
    plot_df = plot_df.sort_values("val_mse")

    # Create figure
    fig, ax = plt.subplots(figsize=(6.7, 3.5))

    # Color by regularization
    colors = []
    for reg in plot_df["regularization"]:
        if reg == "dropout_layernorm":
            colors.append("#2ecc71")  # Green
        elif reg == "lr001":
            colors.append("#3498db")  # Blue
        else:
            colors.append("#95a5a6")  # Gray for unknown/other

    bars = ax.bar(range(len(plot_df)), plot_df["val_mse"], color=colors)

    # Labels
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["model_id"].str.upper(), rotation=45, ha="right")
    ax.set_ylabel("Validation MSE")
    ax.set_xlabel("Model Architecture")
    ax.set_title("Phase 1: Architecture Search Results (Global Conditioning)")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="DO+LN"),
        Patch(facecolor="#3498db", label="DO+LN (lr001)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Annotate best
    best_idx = plot_df["val_mse"].idxmin()
    for i, (idx, _row) in enumerate(plot_df.iterrows()):
        if idx == best_idx:
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure to {output_path}")
    plt.close()


def plot_validation_history(
    df: pd.DataFrame,
    output_path: Path,
    models_to_plot: list[str] | None = None,
    reference_model: str = "vj8",
    wandb_project: str = "phase1",
) -> None:
    """
    Create validation history plot with subplots for MSE, MAE, and Hybrid.

    Plots the training curves for models that were best in any category,
    plus the reference model (VJ8) in black for comparison.

    Args:
        df: DataFrame with run data (must include run_id column)
        output_path: Path to save the figure
        models_to_plot: List of run_ids to plot. If None, auto-selects best models.
        reference_model: Model ID to always include as reference (default: "vj8")
        wandb_project: WandB project name for fetching run histories
    """
    matplotlib_set_rcparams("paper")

    # Auto-select models that were best in any category
    if models_to_plot is None:
        best_models = set()
        # Best by MSE
        best_mse_idx = df["val_mse"].idxmin()
        best_models.add(df.loc[best_mse_idx, "run_id"])
        # Best by MAE
        best_mae_idx = df["val_mae"].idxmin()
        best_models.add(df.loc[best_mae_idx, "run_id"])
        models_to_plot = list(best_models)

    # Always include reference model (VJ8 with dropout_layernorm) for comparison
    # Find the reference model run (prefer dropout_layernorm variant)
    reference_runs = df[df["model_id"] == reference_model]
    if not reference_runs.empty:
        # Prefer dropout_layernorm variant
        ref_with_ln = reference_runs[reference_runs["regularization"] == "dropout_layernorm"]
        if not ref_with_ln.empty:
            reference_run_id = ref_with_ln.iloc[0]["run_id"]
        else:
            reference_run_id = reference_runs.iloc[0]["run_id"]
        # Add to list if not already present
        if reference_run_id not in models_to_plot:
            models_to_plot.append(reference_run_id)

    # Fetch history for each model
    histories = {}
    labels = {}
    is_reference = {}
    for run_id in models_to_plot:
        row = df[df["run_id"] == run_id].iloc[0]
        model_id = row["model_id"]
        model_label = f"{model_id.upper()} ({row['regularization'].replace('_', '+')})"
        labels[run_id] = model_label
        is_reference[run_id] = model_id == reference_model

        logger.info(f"Fetching history for {model_label}...")
        try:
            history = get_run_history(run_id, project=wandb_project)
            histories[run_id] = history
        except Exception as e:
            logger.warning(f"Failed to fetch history for {run_id}: {e}")

    if not histories:
        logger.warning("No history data fetched, skipping validation history plot")
        return

    # Create figure with 2 rows x 3 columns (linear on top, log scale on bottom)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Color palette for non-reference models
    non_ref_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]  # Distinct colors
    non_ref_idx = 0

    # Define metrics for each column
    metric_configs = [
        ("val/loss(mse)", "Validation MSE"),
        ("val/mae", "Validation MAE"),
        ("hybrid", "Hybrid Metric"),
    ]

    # Sort so reference model is plotted first (appears in background)
    sorted_run_ids = sorted(histories.keys(), key=lambda x: not is_reference[x])

    for run_id in sorted_run_ids:
        history = histories[run_id]
        label = labels[run_id]

        # Reference model in black, others in distinct colors
        if is_reference[run_id]:
            color = "black"
            linestyle = "--"
            linewidth = 2.0
            alpha = 0.7
            plot_label = f"{label} (reference)"
        else:
            color = non_ref_colors[non_ref_idx % len(non_ref_colors)]
            non_ref_idx += 1
            linestyle = "-"
            linewidth = 1.5
            alpha = 1.0
            plot_label = label

        # Filter to validation steps (where val metrics are logged)
        val_history = history[history["val/loss(mse)"].notna()].copy()

        if val_history.empty:
            continue

        # Use _step column directly as x-axis (it's the epoch number from wandb.log(step=epoch))
        if "_step" in val_history.columns:
            x = val_history["_step"].values
        else:
            # Fallback: use index if _step not available
            x = range(len(val_history))

        # Compute hybrid metric (normalized by first epoch values)
        mse_vals = val_history["val/loss(mse)"].values
        mae_vals = val_history["val/mae"].values
        mse_baseline = mse_vals[0] if len(mse_vals) > 0 else 1.0
        mae_baseline = mae_vals[0] if len(mae_vals) > 0 else 1.0
        hybrid_vals = ((mse_vals / mse_baseline) * (mae_vals / mae_baseline)) ** 0.5
        val_history["hybrid"] = hybrid_vals

        # Plot each metric in both rows
        for col_idx, (metric_col, ylabel) in enumerate(metric_configs):
            if metric_col not in val_history.columns:
                continue

            y_vals = val_history[metric_col].values

            # Top row: linear scale
            ax_linear = axes[0, col_idx]
            ax_linear.plot(
                x,
                y_vals,
                label=plot_label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax_linear.set_ylabel(ylabel)
            ax_linear.grid(True, alpha=0.3)

            # Bottom row: log scale
            ax_log = axes[1, col_idx]
            ax_log.semilogy(
                x,
                y_vals,
                label=plot_label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax_log.set_ylabel(f"{ylabel} (log)")
            ax_log.set_xlabel("Epoch")
            ax_log.grid(True, alpha=0.3, which="both")

    # Add legends and titles
    for col_idx, (_metric_col, ylabel) in enumerate(metric_configs):
        # Title on top row only
        axes[0, col_idx].set_title(ylabel)
        # Legend on top row only (to avoid duplication)
        axes[0, col_idx].legend(loc="upper right", fontsize=8)

    fig.suptitle("Phase 1: Validation Loss History", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved validation history plot to {output_path}")
    plt.close()


def plot_generalization_gap(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create grouped bar chart comparing Validation vs Test metrics side-by-side.

    Shows gap values annotated (red if positive/worse, green if negative/better).

    Args:
        df: DataFrame with val_mse, val_mae, test_mse, test_mae columns
        output_path: Path to save the figure
    """
    matplotlib_set_rcparams("paper")

    # Filter to rows that have test metrics
    plot_df = df.dropna(subset=["test_mse", "test_mae"]).copy()
    if plot_df.empty:
        logger.warning("No test metrics available, skipping generalization gap plot")
        return

    # Sort by val_mse for consistent ordering
    plot_df = plot_df.sort_values("val_mse")

    # Create labels
    plot_df["label"] = plot_df["model_id"].str.upper()
    if "regularization" in plot_df.columns:
        # Add short reg indicator for models with multiple variants
        reg_short = plot_df["regularization"].apply(
            lambda x: " (LN)" if "layernorm" in str(x) else ""
        )
        plot_df["label"] = plot_df["label"] + reg_short

    # Set up figure with 2 subplots (MSE and MAE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    x = range(len(plot_df))
    width = 0.35

    # MSE comparison
    ax1 = axes[0]
    ax1.bar(
        [i - width / 2 for i in x],
        plot_df["val_mse"],
        width,
        label="Validation",
        color="#3498db",
        alpha=0.8,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        plot_df["test_mse"],
        width,
        label="Test",
        color="#e74c3c",
        alpha=0.8,
    )

    # Annotate gaps
    for i, (_, row) in enumerate(plot_df.iterrows()):
        gap = row["test_mse"] - row["val_mse"]
        color = "#c0392b" if gap > 0 else "#27ae60"
        max_val = max(row["val_mse"], row["test_mse"])
        ax1.annotate(
            f"{gap:+.2e}",
            xy=(i, max_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            rotation=45,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE: Validation vs Test")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # MAE comparison
    ax2 = axes[1]
    ax2.bar(
        [i - width / 2 for i in x],
        plot_df["val_mae"],
        width,
        label="Validation",
        color="#3498db",
        alpha=0.8,
    )
    ax2.bar(
        [i + width / 2 for i in x],
        plot_df["test_mae"],
        width,
        label="Test",
        color="#e74c3c",
        alpha=0.8,
    )

    # Annotate gaps
    for i, (_, row) in enumerate(plot_df.iterrows()):
        gap = row["test_mae"] - row["val_mae"]
        color = "#c0392b" if gap > 0 else "#27ae60"
        max_val = max(row["val_mae"], row["test_mae"])
        ax2.annotate(
            f"{gap:+.3f}",
            xy=(i, max_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            rotation=45,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE: Validation vs Test")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 1: Generalization Gap (Test - Validation)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved generalization gap plot to {output_path}")
    plt.close()


def plot_metric_correlation(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create scatter plot showing Val MSE vs Val MAE correlation.

    Shows whether MSE and MAE rankings correspond well with regression line.

    Args:
        df: DataFrame with val_mse and val_mae columns
        output_path: Path to save the figure
    """
    from scipy import stats

    matplotlib_set_rcparams("paper")

    plot_df = df.dropna(subset=["val_mse", "val_mae"]).copy()
    if len(plot_df) < 3:
        logger.warning("Not enough data points for correlation plot")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Create labels
    plot_df["label"] = plot_df["model_id"].str.upper()

    # Color by regularization
    colors = []
    for _, row in plot_df.iterrows():
        if "layernorm" in str(row.get("regularization", "")):
            colors.append("#2ecc71")
        else:
            colors.append("#3498db")

    # Scatter plot
    ax.scatter(
        plot_df["val_mse"],
        plot_df["val_mae"],
        c=colors,
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidth=1,
    )

    # Add labels for each point
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["label"],
            (row["val_mse"], row["val_mae"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Regression line
    x = plot_df["val_mse"].values
    y = plot_df["val_mae"].values
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x, y)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"R² = {r_value**2:.3f}")

    # Add Spearman rank correlation
    spearman_corr, _spearman_p = stats.spearmanr(x, y)

    ax.set_xlabel("Validation MSE")
    ax.set_ylabel("Validation MAE")
    ax.set_title(f"Phase 1: MSE vs MAE Correlation\n(Spearman ρ = {spearman_corr:.3f})")

    # Legend for regularization
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="DO + LN"),
        Patch(facecolor="#3498db", label="DO only"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Regularization")

    # Add R² annotation
    ax.annotate(
        f"Pearson R² = {r_value**2:.3f}\nSpearman ρ = {spearman_corr:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved metric correlation plot to {output_path}")
    plt.close()


def create_selection_json(
    df: pd.DataFrame,
    selected_id: str | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Create selection JSON for Phase 2.

    Includes checkpoint paths for both Sophia remote and local mount access.

    Args:
        df: DataFrame with run data
        selected_id: Manually selected experiment ID (default: best val_MSE)
        output_path: Path to save JSON (optional)

    Returns:
        Selection dictionary
    """
    if selected_id is None:
        # Select by best val_MSE
        best_idx = df["val_mse"].idxmin()
        selected_row = df.loc[best_idx]
    else:
        # Manual selection
        mask = (df["model_id"] == selected_id.lower()) | (df["run_name"].str.contains(selected_id))
        if mask.any():
            selected_row = df[mask].iloc[0]
        else:
            logger.warning(f"Selected ID '{selected_id}' not found. Using best val_MSE.")
            best_idx = df["val_mse"].idxmin()
            selected_row = df.loc[best_idx]

    # Convert WandB path to local path
    wandb_path = selected_row["run_name"]
    local_path, path_status = convert_wandb_path_to_local(wandb_path)

    # Build checkpoint paths with both Sophia and local paths
    checkpoint_paths = {}
    if local_path is not None and path_status == "available":
        checkpoints = get_checkpoint_paths(local_path)
        for ckpt_name, ckpt_path in checkpoints.items():
            if ckpt_path is not None:
                checkpoint_paths[ckpt_name] = {
                    "sophia_path": local_path_to_sophia_path(ckpt_path),
                    "local_path": str(ckpt_path),
                }
            else:
                checkpoint_paths[ckpt_name] = None

    selection = {
        "phase": "phase1",
        "selected_experiment": selected_row["run_name"],
        "run_id": selected_row["run_id"],
        "model_id": selected_row["model_id"],
        "regularization": selected_row["regularization"],
        "val_mse": float(selected_row["val_mse"]) if pd.notna(selected_row["val_mse"]) else None,
        "val_mae": float(selected_row["val_mae"]) if pd.notna(selected_row["val_mae"]) else None,
        "selection_metric": "val_mse",
        "selection_date": datetime.now().isoformat(),
        # Path information
        "sophia_remote_path": wandb_path,
        "sophia_local_path": str(local_path) if local_path else None,
        "path_status": path_status,
        "checkpoints": checkpoint_paths,
    }

    # Log path status
    if path_status == "sophia_not_mounted":
        logger.warning(
            f"Sophia is not mounted at {SOPHIA_LOCAL_MOUNT}. " "Mount Sophia to access checkpoints."
        )
    elif path_status == "path_not_found":
        logger.warning(
            f"Experiment path not found: {local_path}. "
            "The experiment may still be running or the path may be incorrect."
        )
    elif path_status == "available":
        logger.info(f"Checkpoint path available: {local_path}")
        if checkpoint_paths.get("best_mse"):
            logger.info(
                f"  Best MSE checkpoint (Sophia): {checkpoint_paths['best_mse']['sophia_path']}"
            )
            logger.info(
                f"  Best MSE checkpoint (local):  {checkpoint_paths['best_mse']['local_path']}"
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(selection, f, indent=2)
        logger.info(f"Saved selection to {output_path}")

    return selection


def process_wandb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process WandB data to add parsed experiment info.

    Args:
        df: Raw DataFrame from WandB

    Returns:
        Processed DataFrame with model_id and regularization columns
    """
    df = df.copy()

    # Parse experiment IDs
    parsed = df["run_name"].apply(parse_experiment_id)
    df["model_id"] = [p[0] for p in parsed]
    df["regularization"] = [p[1] for p in parsed]

    # Filter to completed runs only
    if "state" in df.columns:
        df = df[df["state"] == "finished"]

    return df


def filter_runs_by_config(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter runs based on configuration data path filter.

    Args:
        df: DataFrame with run data (must include config.data.main_path column)
        config: Configuration dictionary with data_path_filter key

    Returns:
        Filtered DataFrame containing only runs matching the data path filter
    """
    data_path_col = "config.data.main_path"
    if data_path_col in df.columns:
        mask = df[data_path_col].str.contains(config["data_path_filter"], na=False)
        filtered_df = df[mask].copy()
        logger.info(
            f"Filtered to {len(filtered_df)}/{len(df)} runs matching "
            f"'{config['data_path_filter']}'"
        )
        return filtered_df
    else:
        logger.warning(
            f"Column '{data_path_col}' not found in data. " "Returning all runs without filtering."
        )
        return df


def main(
    config_name: str = DEFAULT_CONFIG,
    refresh: bool = False,
    select: str | None = None,
    include_test: bool = False,
    table_format: str = "compact",
) -> None:
    """
    Main analysis function.

    Args:
        config_name: Configuration name from RUN_CONFIGS (default: full_2500layouts)
        refresh: Force refetch from WandB
        select: Manual experiment selection
        include_test: Include test metrics from JSON files (requires Sophia mount)
        table_format: "compact" (Val+Test RMSE/MAE, default) or "full" (legacy behavior)
    """
    # Compact format requires test metrics
    if table_format == "compact":
        include_test = True
    # Load configuration
    if config_name not in RUN_CONFIGS:
        logger.error(f"Unknown config '{config_name}'. Available: {list(RUN_CONFIGS.keys())}")
        return
    config = RUN_CONFIGS[config_name]
    output_prefix = config["output_prefix"]

    logger.info("=" * 60)
    logger.info(f"Phase 1: Architecture Search Analysis - {config['name']}")
    logger.info("=" * 60)

    # Fetch data from WandB
    wandb_project = config.get("wandb_project", "phase1")
    logger.info(
        f"Fetching data from WandB project: {WANDB_PROJECTS.get(wandb_project, wandb_project)}"
    )
    df = fetch_wandb_runs(
        project=wandb_project,
        cache_dir=CACHE_DIR,
        refresh=refresh,
        filters={"state": "finished"},
    )

    if df.empty:
        logger.error("No runs found. Please check WandB project and run status.")
        return

    # Enrich with best (min) validation metrics from run histories
    if refresh:
        logger.info("Enriching with best (min) validation metrics from run histories...")
        df = enrich_with_best_metrics(df, project=wandb_project)
        # Re-save cache with corrected min values
        cache_path = CACHE_DIR / f"{WANDB_PROJECTS.get(wandb_project, wandb_project)}_runs.csv"
        df.to_csv(cache_path, index=False)
        logger.info(f"Re-saved cache with best metrics to {cache_path}")

    # Process data
    df = process_wandb_data(df)
    logger.info(f"Found {len(df)} total completed runs")

    # Filter runs by configuration (data path)
    df = filter_runs_by_config(df, config)

    # Add test metrics if requested
    if include_test:
        logger.info("Loading test metrics from experiment directories...")
        df = add_test_and_param_metrics(df)
        test_available = df["test_mse"].notna().sum()
        logger.info(f"Loaded test metrics for {test_available}/{len(df)} runs")

    # Print summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    summary_cols = ["run_name", "model_id", "regularization", "val_mse", "val_mae"]
    if include_test:
        summary_cols.extend(["test_mse", "test_mae"])
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].sort_values("val_mse").to_string(index=False))

    # Create model configs table
    logger.info("Creating model configs table...")
    configs_df = create_model_configs_table(df)
    if not configs_df.empty:
        # Dynamic column format: first two columns left-aligned, rest centered
        num_cols = len(configs_df.columns)
        col_format = "l|l|" + "c" * (num_cols - 2)
        configs_latex = create_latex_table(
            configs_df,
            caption="Model architecture configurations for Phase 1 architecture search.",
            label="tab:phase1_model_configs",
            column_format=col_format,
        )
        save_table(
            configs_latex,
            OUTPUTS_DIR / f"{output_prefix}_model_configs.tex",
            warning_url=GITLAB_URL,
        )
        # Also save as CSV
        save_dataframe_csv(configs_df, OUTPUTS_DIR / f"{output_prefix}_model_configs.csv")

    # Create results table
    logger.info("Creating results table...")
    if table_format == "compact" and df["test_mse"].notna().any():
        results_df = create_compact_results_table(df)
        num_result_cols = len(results_df.columns)
        result_col_format = "c|l|l|" + "c" * (num_result_cols - 3)
        highlight_cols = ["Test RMSE", "Test MAE"]
        results_latex = create_latex_table(
            results_df,
            caption="Phase 1 architecture search results, ranked by test RMSE.",
            label="tab:phase1_results",
            column_format=result_col_format,
            highlight_columns=highlight_cols,
            highlight_direction="min",
            float_format="%.6f",
        )
    else:
        results_df = create_results_table(df, include_test=include_test)
        num_result_cols = len(results_df.columns)
        result_col_format = "c|l|l|" + "c" * (num_result_cols - 3)
        use_test = include_test and "Test MSE" in results_df.columns
        metric_prefix = "Test" if use_test else "Val"
        highlight_cols = [f"{metric_prefix} MSE", f"{metric_prefix} MAE"]
        if f"{metric_prefix} RMSE" in results_df.columns:
            highlight_cols.append(f"{metric_prefix} RMSE")
        if "Hybrid*" in results_df.columns:
            highlight_cols.append("Hybrid*")
        ranked_by = "test MSE" if use_test else "validation MSE"
        results_latex = create_latex_table(
            results_df,
            caption=f"Phase 1 architecture search results, ranked by {ranked_by}. *Hybrid metric computed as geometric mean of normalized MSE and MAE.",
            label="tab:phase1_results",
            column_format=result_col_format,
            highlight_columns=highlight_cols,
            highlight_direction="min",
            float_format="%.6f",
        )
    save_table(
        results_latex,
        OUTPUTS_DIR / f"{output_prefix}_results.tex",
        warning_url=GITLAB_URL,
    )
    save_dataframe_csv(results_df, OUTPUTS_DIR / f"{output_prefix}_results.csv")

    # Create comprehensive results table with test metrics
    if include_test:
        logger.info("Creating comprehensive results table with test metrics...")
        comprehensive_df = create_comprehensive_results_table(df)
        save_dataframe_csv(
            comprehensive_df, OUTPUTS_DIR / f"{output_prefix}_comprehensive_results.csv"
        )

        # Create comprehensive LaTeX table
        comp_highlight_cols = ["Val MSE", "Val MAE", "Test MSE", "Test MAE", "Hybrid"]
        comp_latex = create_latex_table(
            comprehensive_df,
            caption="Comprehensive Phase 1 results with test metrics and generalization gaps.",
            label="tab:phase1_comprehensive",
            highlight_columns=comp_highlight_cols,
            highlight_direction="min",
            float_format="%.6f",
        )
        save_table(
            comp_latex,
            OUTPUTS_DIR / f"{output_prefix}_comprehensive_results.tex",
            warning_url=GITLAB_URL,
        )

    # Create figures
    logger.info("Creating architecture comparison figure...")
    plot_architecture_comparison(df, FIGURES_DIR / f"{output_prefix}_architecture_comparison.pdf")

    logger.info("Creating validation history figure...")
    plot_validation_history(
        df,
        FIGURES_DIR / f"{output_prefix}_validation_history.pdf",
        wandb_project=wandb_project,
    )

    # Create metric correlation plot
    logger.info("Creating metric correlation figure...")
    plot_metric_correlation(df, FIGURES_DIR / f"{output_prefix}_metric_correlation.pdf")

    # Create generalization gap plot if test metrics available
    if include_test and df["test_mse"].notna().any():
        logger.info("Creating generalization gap figure...")
        plot_generalization_gap(df, FIGURES_DIR / f"{output_prefix}_generalization_gap.pdf")

    # Create selection JSON
    logger.info("Creating selection JSON...")
    selection = create_selection_json(
        df,
        selected_id=select,
        output_path=OUTPUTS_DIR / f"{output_prefix}_selection.json",
    )

    # Print selection
    print("\n" + "=" * 60)
    print("SELECTION FOR PHASE 2")
    print("=" * 60)
    print(json.dumps(selection, indent=2))

    logger.info("=" * 60)
    logger.info("Phase 1 analysis complete!")
    logger.info(f"Outputs saved to: {OUTPUTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Architecture Search Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        choices=list(RUN_CONFIGS.keys()),
        help=f"Run configuration to use (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refetch from WandB (ignore cache)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Manually select experiment ID for Phase 2",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test metrics from JSON files (requires Sophia mount)",
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="compact",
        choices=["compact", "full"],
        help="Table format: 'compact' (Val+Test RMSE/MAE, default) or 'full' (legacy)",
    )

    args = parser.parse_args()
    main(
        config_name=args.config,
        refresh=args.refresh,
        select=args.select,
        include_test=args.include_test,
        table_format=args.table_format,
    )
