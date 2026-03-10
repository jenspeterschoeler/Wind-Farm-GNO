"""
Phase 2 Global: Fine-tuning Technique Comparison (Global Conditioning).

Analyzes results of Phase 2 fine-tuning experiments with global-conditioned L1 model.
Compares 25 techniques x 2 clipping options = ~50 runs.

WandB Project: GNO_phase2_global

Experiments analyzed:
- 1 model: L1 global (latent_size=128)
- 25 techniques: F1-F6, L1-L18, SCRATCH
- 2 clipping: clip_1.0, disabled

Outputs:
- outputs/phase2_global_results.{tex,csv}: Grouped results table
- outputs/phase2_global_flat_ranking.csv: All experiments ranked by Val MSE
- outputs/phase2_global_trainable_params.{tex,csv}: Trainable parameter counts
- figures/phase2_global_technique_comparison.{pdf,png}: Horizontal bar chart ranked by val_mse
- figures/phase2_global_validation_history.{pdf,png}: Training curves (top-N)
- figures/phase2_global_clipping_effect.{pdf,png}: Clip vs no-clip scatter
- figures/phase2_global_generalization_gap.{pdf,png}: Val vs Test (with --include-test)
- figures/phase2_global_technique_groups.{pdf,png}: Box plot by technique group

Usage:
    python phase2_global_results.py [--refresh] [--include-test] [--top-n N]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import matplotlib_set_rcparams  # noqa: E402
from utils.reporting import (  # noqa: E402
    WANDB_PROJECTS,
    convert_wandb_path_to_local,
    create_comparison_table,
    create_latex_table,
    enrich_with_best_metrics,
    fetch_wandb_runs,
    get_run_history,
    load_test_metrics,
    save_dataframe_csv,
    save_table,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
CACHE_DIR = SCRIPT_DIR / "cache"

OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

GITLAB_URL = "https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/article_2/phase2_global_results.py"

OUTPUT_PREFIX = "phase2_global"
WANDB_PROJECT = "phase2_global"

# Known models and techniques (L1 only for article)
KNOWN_MODELS = ["L1"]
KNOWN_TECHNIQUES = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L9",
    "L10",
    "L11",
    "L12",
    "L13",
    "L14",
    "L15",
    "L16",
    "L17",
    "L18",
    "SCRATCH",
]

# Concise display names for techniques
TECHNIQUE_DISPLAY = {
    "F1": "Full FT",
    "F2": "Frz Emb",
    "F3": "Frz Emb+Proc",
    "F4": "Dec Only",
    "F5": "Full+Anch",
    "F6": "Frz Emb+Anch",
    "L1": "LoRA-4",
    "L2": "LoRA-8",
    "L3": "LoRA-16",
    "L4": "LoRA-8 FrzE",
    "L5": "LoRA-4 FrzE",
    "L6": "LoRA-4 Anch",
    "L7": "LoRA-16 FrzE",
    "L8": "LoRA-8 Anch",
    "L9": "LoRA-16 Anch",
    "L10": "LoRA-4 FrzE+A",
    "L11": "LoRA-8 FrzE+A",
    "L12": "LoRA-16 FrzE+A",
    "L13": "LoRA-4 FrzEP",
    "L14": "LoRA-8 FrzEP",
    "L15": "LoRA-16 FrzEP",
    "L16": "LoRA-4 FrzEP+A",
    "L17": "LoRA-8 FrzEP+A",
    "L18": "LoRA-16 FrzEP+A",
    "SCRATCH": "Scratch",
}

# Technique groups for article table and box plots
TECHNIQUE_GROUPS = {
    "From scratch": ["SCRATCH"],
    "Full fine-tune": ["F1"],
    "LoRA": ["L1", "L2", "L3", "L6", "L8", "L9"],
    "Freeze embedder": ["F2", "F6", "L4", "L5", "L7", "L10", "L11", "L12"],
    "Freeze emb+proc": ["F3", "F4", "L13", "L14", "L15", "L16", "L17", "L18"],
    "Weight anchoring": ["F5"],
}

# Reverse mapping: technique -> group
TECHNIQUE_TO_GROUP = {}
for _group_name, _techniques in TECHNIQUE_GROUPS.items():
    for _tech in _techniques:
        TECHNIQUE_TO_GROUP[_tech] = _group_name

# LoRA config -> technique code mapping
# Key: (rank, freeze_emb, freeze_proc, anchoring)
# Alpha is always 2 * rank (see configurations/finetuning/lora/*.yaml)
LORA_MAP = {
    (4, False, False, False): "L1",
    (8, False, False, False): "L2",
    (16, False, False, False): "L3",
    (8, True, False, False): "L4",
    (4, True, False, False): "L5",
    (4, False, False, True): "L6",
    (16, True, False, False): "L7",
    (8, False, False, True): "L8",
    (16, False, False, True): "L9",
    (4, True, False, True): "L10",
    (8, True, False, True): "L11",
    (16, True, False, True): "L12",
    (4, True, True, False): "L13",
    (8, True, True, False): "L14",
    (16, True, True, False): "L15",
    (4, True, True, True): "L16",
    (8, True, True, True): "L17",
    (16, True, True, True): "L18",
}

# Derived mappings: technique code -> LoRA rank, and set of anchoring techniques
LORA_RANK = {code: rank for (rank, _, _, _), code in LORA_MAP.items()}
ANCHORING_TECHNIQUES = {"F5", "F6"} | {code for (_, _, _, anch), code in LORA_MAP.items() if anch}


# =============================================================================
# Helpers
# =============================================================================
def _best_model_description(technique: str, clipping: str, group: str) -> str:
    """Build 'Best model options' description, avoiding redundancy with the group name."""
    parts = []

    rank = LORA_RANK.get(technique)
    if rank:
        alpha = 2 * rank
        parts.append(rf"LoRA ($r={rank}$, $\alpha={alpha}$)")

    # Mention anchoring only when the group doesn't already imply it
    if technique in ANCHORING_TECHNIQUES and group != "Weight anchoring":
        parts.append("anchoring")

    # Clipping
    if clipping == "clip_1.0":
        parts.append("clip 1.0")
    else:
        parts.append("no clip")

    return ", ".join(parts)


def _natural_sort_key(s: str) -> list:
    """Sort key for natural ordering: L1, L2, ..., L9, L10, ..., L18."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def get_technique_display(technique: str) -> str:
    """Return display name for a technique code, defaulting to the code itself."""
    return TECHNIQUE_DISPLAY.get(technique, technique)


def get_technique_colors(techniques: list[str]) -> dict[str, str]:
    """Auto-generate colors: warm for F-series, cool for L-series, gray for SCRATCH."""
    colors = {}
    f_techs = sorted([t for t in techniques if t.startswith("F")], key=_natural_sort_key)
    l_techs = sorted([t for t in techniques if t.startswith("L")], key=_natural_sort_key)

    # Warm colors for F-series (Oranges)
    if f_techs:
        f_cmap = plt.colormaps["Oranges"]
        for i, t in enumerate(f_techs):
            colors[t] = mcolors.to_hex(f_cmap(0.35 + 0.55 * i / max(len(f_techs) - 1, 1)))

    # Cool colors for L-series (winter: blue->green)
    if l_techs:
        l_cmap = plt.colormaps["winter"]
        for i, t in enumerate(l_techs):
            colors[t] = mcolors.to_hex(l_cmap(0.05 + 0.9 * i / max(len(l_techs) - 1, 1)))

    # Gray for SCRATCH
    if "SCRATCH" in techniques:
        colors["SCRATCH"] = "#7f8c8d"

    return colors


# =============================================================================
# Run Name Parsing
# =============================================================================
def parse_phase2_run_name(run_name: str) -> dict[str, str]:
    """
    Parse a Phase 2 Global run name to extract model, technique, and clipping.

    Expected format (Hydra override dirname):
        .../N_+experiment=phase2/L1_global_F1,finetuning/gradient_clipping=clip_1.0

    Args:
        run_name: WandB run name (= Hydra output_dir)

    Returns:
        Dictionary with keys: model, technique, clipping
    """
    result = {"model": "unknown", "technique": "unknown", "clipping": "unknown"}

    # Extract experiment part: phase2/XXX_global_YYY
    exp_match = re.search(r"experiment=phase2/(\w+)_global_(\w+)", run_name)
    if exp_match:
        result["model"] = exp_match.group(1).upper()
        result["technique"] = exp_match.group(2).upper()

    # Extract clipping: gradient_clipping=XXX
    clip_match = re.search(r"gradient_clipping=(\S+)", run_name)
    if clip_match:
        result["clipping"] = clip_match.group(1)
        # Clean up trailing path separators or commas
        result["clipping"] = result["clipping"].rstrip("/,")

    return result


def parse_phase2_from_config(row: pd.Series) -> dict[str, str]:
    """
    Fallback: parse model/technique/clipping from WandB config columns.

    Args:
        row: DataFrame row with config.* columns

    Returns:
        Dictionary with keys: model, technique, clipping
    """
    result = {"model": "unknown", "technique": "unknown", "clipping": "unknown"}

    # Model from latent_size
    latent = row.get("config.model.latent_size")
    if pd.notna(latent):
        latent = int(latent)
        if latent == 128:
            result["model"] = "L1"
        elif latent == 100:
            result["model"] = "VJ8"
        elif latent == 50:
            result["model"] = "S1"

    # Clipping
    clip_val = row.get("config.finetuning.gradient_clipping.max_norm")
    if pd.notna(clip_val):
        result["clipping"] = f"clip_{clip_val}"
    else:
        result["clipping"] = "disabled"

    # SCRATCH detection: no pretrained checkpoint
    pretrained = row.get("config.pretrained_checkpoint_path")
    if pretrained is None or (isinstance(pretrained, float) and pd.isna(pretrained)):
        # Also check alternative key
        pretrained = row.get("config.finetuning.pretrained_checkpoint_path")
    if pretrained is None or (isinstance(pretrained, float) and pd.isna(pretrained)):
        result["technique"] = "SCRATCH"
        return result
    pretrained_str = str(pretrained).strip().lower()
    if pretrained_str in ("null", "none", "nan", ""):
        result["technique"] = "SCRATCH"
        return result

    # Detect LoRA vs F-series
    lora_enabled = row.get("config.finetuning.lora.enabled", False)
    if isinstance(lora_enabled, str):
        lora_enabled = lora_enabled.lower() == "true"
    lora_enabled = bool(lora_enabled)

    anchoring_enabled = row.get("config.finetuning.weight_anchoring.enabled", False)
    if isinstance(anchoring_enabled, str):
        anchoring_enabled = anchoring_enabled.lower() == "true"
    anchoring_enabled = bool(anchoring_enabled)

    # Detect frozen components from config
    frozen_str = str(row.get("config.finetuning.freezing.frozen_components", "")).lower()
    freeze_emb = "embedder" in frozen_str
    freeze_proc = "wt_processor" in frozen_str
    freeze_probe = "probe_processor" in frozen_str

    if lora_enabled:
        rank = int(row.get("config.finetuning.lora.rank", 0))
        key = (rank, freeze_emb, freeze_proc, anchoring_enabled)
        result["technique"] = LORA_MAP.get(key, f"L_r{rank}")
    elif freeze_emb and freeze_proc and freeze_probe:
        # Decoder only trainable (F4)
        result["technique"] = "F4"
    elif freeze_emb and freeze_proc and not freeze_probe:
        if anchoring_enabled:
            result["technique"] = "F3"  # F3 has no anchoring in current configs
        else:
            result["technique"] = "F3"
    elif freeze_emb and not freeze_proc:
        result["technique"] = "F6" if anchoring_enabled else "F2"
    elif anchoring_enabled:
        result["technique"] = "F5"
    else:
        result["technique"] = "F1"

    return result


# =============================================================================
# Data Processing
# =============================================================================
def process_phase2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process WandB data: parse run names, filter to L1, handle duplicates.

    Args:
        df: Raw DataFrame from fetch_wandb_runs

    Returns:
        Processed DataFrame with model, technique, clipping columns
    """
    df = df.copy()

    # Parse run names
    parsed = df["run_name"].apply(parse_phase2_run_name)
    df["model"] = [p["model"] for p in parsed]
    df["technique"] = [p["technique"] for p in parsed]
    df["clipping"] = [p["clipping"] for p in parsed]

    # Fallback for rows where parsing failed
    unknown_mask = (df["model"] == "unknown") | (df["technique"] == "unknown")
    if unknown_mask.any():
        logger.warning(
            f"Run name parsing failed for {unknown_mask.sum()} runs, trying config fallback"
        )
        for idx in df[unknown_mask].index:
            fallback = parse_phase2_from_config(df.loc[idx])
            if df.at[idx, "model"] == "unknown":
                df.at[idx, "model"] = fallback["model"]
            if df.at[idx, "technique"] == "unknown":
                df.at[idx, "technique"] = fallback["technique"]
            if df.at[idx, "clipping"] == "unknown":
                df.at[idx, "clipping"] = fallback["clipping"]

    # Filter to finished runs only
    if "state" in df.columns:
        n_before = len(df)
        df = df[df["state"] == "finished"].copy()
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            logger.info(f"Filtered out {n_filtered} non-finished runs")

    # Filter to L1 model only
    n_before = len(df)
    df = df[df["model"] == "L1"].copy()
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.info(f"Filtered out {n_filtered} non-L1 runs")

    # Handle duplicate runs (keep latest for each model/technique/clipping combo)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
        n_before = len(df)
        df = df.drop_duplicates(subset=["model", "technique", "clipping"], keep="first")
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.info(f"Removed {n_dupes} duplicate runs (kept latest)")

    return df


# =============================================================================
# Test Metrics & Trainable Params
# =============================================================================
def add_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add test metrics from experiment JSON files.

    Args:
        df: DataFrame with run_name column containing WandB paths

    Returns:
        DataFrame with test_mse and test_mae columns added
    """
    df = df.copy()
    df["test_mse"] = None
    df["test_mae"] = None
    df["test_rmse"] = None

    for idx, row in df.iterrows():
        local_path, status = convert_wandb_path_to_local(row["run_name"])
        if local_path is None or status != "available":
            continue

        test_metrics = load_test_metrics(local_path)
        if test_metrics:
            if "test_best_mse_mse" in test_metrics:
                df.at[idx, "test_mse"] = test_metrics["test_best_mse_mse"]
            elif "test_final_mse" in test_metrics:
                df.at[idx, "test_mse"] = test_metrics["test_final_mse"]

            if "test_best_mse_mae" in test_metrics:
                df.at[idx, "test_mae"] = test_metrics["test_best_mse_mae"]
            elif "test_final_mae" in test_metrics:
                df.at[idx, "test_mae"] = test_metrics["test_final_mae"]

            if "test_best_mse_rmse" in test_metrics:
                df.at[idx, "test_rmse"] = test_metrics["test_best_mse_rmse"]
            elif "test_final_rmse" in test_metrics:
                df.at[idx, "test_rmse"] = test_metrics["test_final_rmse"]

    return df


def add_trainable_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trainable parameter counts from WandB summary/config.

    Strategies:
    - F1/F5/SCRATCH: all params are trainable
    - Other F-series (F2-F4, F6): try logged decoder/trainable params
    - L-series: try logged lora/trainable params

    Args:
        df: DataFrame with config columns from WandB

    Returns:
        DataFrame with total_params, trainable_params, trainable_pct columns
    """
    df = df.copy()
    df["total_params"] = None
    df["trainable_params"] = None
    df["trainable_pct"] = None

    # Try to get total params from WandB summary
    for col_name in ["config.params.total_params", "config.model.total_params"]:
        if col_name in df.columns:
            mask = df[col_name].notna()
            df.loc[mask, "total_params"] = df.loc[mask, col_name].astype(float)

    # Estimate trainable params based on technique
    for idx, row in df.iterrows():
        total = row.get("total_params")
        if pd.isna(total) or total is None:
            continue
        total = float(total)

        technique = row["technique"]

        if technique in ("F1", "F5", "SCRATCH"):
            # All parameters trainable (anchoring is regularization, not freezing)
            df.at[idx, "trainable_params"] = total
        elif technique.startswith("F"):
            # Frozen F-series: try logged trainable/decoder params
            for col in [
                "config.params.trainable_params",
                "config.params.decoder_params",
            ]:
                if col in df.columns and pd.notna(row.get(col)):
                    df.at[idx, "trainable_params"] = float(row[col])
                    break
        elif technique.startswith("L"):
            # LoRA: try logged lora/trainable params
            for col in [
                "config.params.lora_params",
                "config.params.trainable_params",
                "config.finetuning.lora.trainable_params",
            ]:
                if col in df.columns and pd.notna(row.get(col)):
                    df.at[idx, "trainable_params"] = float(row[col])
                    break

    # Compute percentage
    mask = df["total_params"].notna() & df["trainable_params"].notna()
    df.loc[mask, "trainable_pct"] = df.loc[mask, "trainable_params"].astype(float) / df.loc[
        mask, "total_params"
    ].astype(float)

    return df


# =============================================================================
# Tables
# =============================================================================
def _fmt_time(hours: float) -> str:
    """Format wall time as 'Xh YYm' string."""
    if pd.isna(hours):
        return "--"
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h {m:02d}m"


def create_grouped_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create compact results table with one row per technique group, showing the best config.

    For each of the 6 technique groups, finds the configuration with the lowest
    validation RMSE and reports its RMSE and MAE.

    Args:
        df: Processed DataFrame with technique, clipping columns

    Returns:
        DataFrame with 6 rows (one per group), sorted by RMSE ascending
    """
    # Compute RMSE from MSE
    df = df.copy()
    df["val_rmse"] = np.sqrt(df["val_mse"])

    # Check if wall time and epoch data is available
    has_runtime = "runtime_seconds" in df.columns and df["runtime_seconds"].notna().any()
    has_epoch = "final_epoch" in df.columns and df["final_epoch"].notna().any()

    # Map each row to its technique group
    df["group"] = df["technique"].map(TECHNIQUE_TO_GROUP)

    records = []
    for group_name, group_df in df.groupby("group"):
        if group_df.empty:
            continue
        # Best = lowest val RMSE
        best_idx = group_df["val_rmse"].idxmin()
        best = group_df.loc[best_idx]

        best_desc = _best_model_description(best["technique"], best["clipping"], group_name)

        record = {
            "Group": group_name,
            "$N$": len(group_df),
            "Best model options": best_desc,
            "RMSE": best["val_rmse"],
            "MAE": best["val_mae"],
        }

        if has_epoch:
            epoch_val = best.get("final_epoch")
            record["Final epoch"] = int(epoch_val) if pd.notna(epoch_val) else "--"

        if has_runtime and pd.notna(best.get("runtime_seconds")):
            record["Time"] = _fmt_time(float(best["runtime_seconds"]) / 3600)
        elif has_runtime:
            record["Time"] = "--"

        records.append(record)

    result = pd.DataFrame(records)
    result = result.sort_values("RMSE").reset_index(drop=True)

    return result


def create_flat_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flat ranking of all experiments by Val MSE.

    Args:
        df: Processed DataFrame

    Returns:
        Ranked DataFrame with all metrics
    """
    metric_cols = ["val_mse", "val_mae"]
    if "val_rmse" in df.columns and df["val_rmse"].notna().any():
        metric_cols.append("val_rmse")
    if "test_mse" in df.columns and df["test_mse"].notna().any():
        metric_cols.extend(["test_mse", "test_mae"])
    if "test_rmse" in df.columns and df["test_rmse"].notna().any():
        metric_cols.append("test_rmse")

    id_cols = ["model", "technique", "clipping"]
    if "runtime_seconds" in df.columns and df["runtime_seconds"].notna().any():
        id_cols.append("runtime_seconds")

    return create_comparison_table(
        df,
        metric_columns=metric_cols,
        id_columns=id_cols,
        rank_by="val_mse",
        ascending=True,
    )


def create_trainable_params_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create table showing parameter counts per technique.

    Args:
        df: DataFrame with total_params, trainable_params, trainable_pct

    Returns:
        Summary DataFrame
    """
    records = []
    for technique in KNOWN_TECHNIQUES:
        tech_rows = df[df["technique"] == technique]
        if tech_rows.empty:
            continue

        # Use first available row for param counts (same technique = same structure)
        row = tech_rows.iloc[0]
        total = row.get("total_params")
        trainable = row.get("trainable_params")
        pct = row.get("trainable_pct")

        record = {
            "Code": technique,
            "Technique": get_technique_display(technique),
            "Group": TECHNIQUE_TO_GROUP.get(technique, ""),
            "Total (K)": f"{total / 1000:.1f}" if pd.notna(total) else "--",
            "Trainable (K)": f"{trainable / 1000:.1f}" if pd.notna(trainable) else "--",
            "Trainable %": f"{pct * 100:.1f}" if pd.notna(pct) else "--",
        }
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# Figures
# =============================================================================
def plot_technique_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Horizontal bar chart of all techniques ranked by val_mse.

    Each technique gets two bars (clip and no-clip), hatching distinguishes them.

    Args:
        df: Processed DataFrame
        output_path: Path to save the figure
    """
    matplotlib_set_rcparams("paper")

    colors = get_technique_colors(df["technique"].unique().tolist())

    # Rank techniques by best val_mse across clipping variants
    best_per_tech = df.groupby("technique")["val_mse"].min().sort_values()
    ranked_techniques = best_per_tech.index.tolist()
    n_techs = len(ranked_techniques)

    fig, ax = plt.subplots(figsize=(8, max(5, n_techs * 0.45)))

    y_positions = np.arange(n_techs)
    height = 0.35

    for i, technique in enumerate(ranked_techniques):
        tech_df = df[df["technique"] == technique]
        color = colors.get(technique, "#95a5a6")

        clip_row = tech_df[tech_df["clipping"] == "clip_1.0"]
        noclip_row = tech_df[tech_df["clipping"] == "disabled"]

        if not clip_row.empty:
            ax.barh(
                i + height / 2,
                clip_row.iloc[0]["val_mse"],
                height,
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )
        if not noclip_row.empty:
            ax.barh(
                i - height / 2,
                noclip_row.iloc[0]["val_mse"],
                height,
                color=color,
                alpha=0.6,
                hatch="//",
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([get_technique_display(t) for t in ranked_techniques], fontsize=8)
    ax.set_xlabel("Validation MSE")
    ax.invert_yaxis()  # Best at top
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#666", alpha=0.9, label="Clip 1.0"),
        Patch(
            facecolor="#666",
            alpha=0.6,
            hatch="//",
            edgecolor="white",
            label="No clip",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_title("Phase 2 Global: Technique Comparison (ranked by Val MSE)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved technique comparison to {output_path}")
    plt.close()


def plot_validation_history(
    df: pd.DataFrame,
    output_path: Path,
    prefer_clipping: str = "clip_1.0",
    top_n: int = 10,
) -> None:
    """
    Validation training curves in 2x2 layout: (MSE, MAE) x (linear, log).

    Only plots top-N best techniques to keep the figure readable.

    Args:
        df: Processed DataFrame with run_id column
        output_path: Path to save the figure
        prefer_clipping: Show only this clipping variant to reduce clutter
        top_n: Number of top techniques to plot
    """
    matplotlib_set_rcparams("paper")

    # Select runs to plot (prefer specified clipping variant)
    plot_df = df[df["clipping"] == prefer_clipping].copy()
    if plot_df.empty:
        plot_df = df.copy()
        logger.warning(f"No runs with clipping={prefer_clipping}, plotting all")

    # Keep only top-N by val_mse
    if len(plot_df) > top_n:
        plot_df = plot_df.nsmallest(top_n, "val_mse")

    colors = get_technique_colors(plot_df["technique"].unique().tolist())

    # Fetch histories
    histories = {}
    for _, row in plot_df.iterrows():
        run_id = row["run_id"]
        label = row["technique"]
        logger.info(f"Fetching history for {label}...")
        try:
            history = get_run_history(run_id, project=WANDB_PROJECT)
            histories[run_id] = {
                "history": history,
                "technique": row["technique"],
                "label": label,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch history for {run_id}: {e}")

    if not histories:
        logger.warning("No history data fetched, skipping validation history plot")
        return

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    metric_configs = [
        ("val/loss(mse)", "Validation MSE"),
        ("val/mae", "Validation MAE"),
    ]

    for _run_id, info in histories.items():
        history = info["history"]
        technique = info["technique"]
        label = info["label"]

        color = colors.get(technique, "#95a5a6")

        val_history = history[history["val/loss(mse)"].notna()].copy()
        if val_history.empty:
            continue

        x = (
            val_history["_step"].values
            if "_step" in val_history.columns
            else range(len(val_history))
        )

        for col_idx, (metric_col, ylabel) in enumerate(metric_configs):
            if metric_col not in val_history.columns:
                continue

            y_vals = val_history[metric_col].values

            # Top row: linear
            ax_lin = axes[0, col_idx]
            ax_lin.plot(x, y_vals, label=label, color=color, linewidth=1.5)
            ax_lin.set_ylabel(ylabel)
            ax_lin.grid(True, alpha=0.3)

            # Bottom row: log
            ax_log = axes[1, col_idx]
            ax_log.semilogy(x, y_vals, label=label, color=color, linewidth=1.5)
            ax_log.set_ylabel(f"{ylabel} (log)")
            ax_log.set_xlabel("Epoch")
            ax_log.grid(True, alpha=0.3, which="both")

    # Add titles and legends
    for col_idx, (_, ylabel) in enumerate(metric_configs):
        axes[0, col_idx].set_title(ylabel)
        axes[0, col_idx].legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"Phase 2 Global: Validation History ({prefer_clipping}, top {top_n})",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved validation history to {output_path}")
    plt.close()


def plot_clipping_effect(df: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter plot of clip_val (x) vs noclip_val (y) with diagonal reference.

    Points above the diagonal: clipping helps (lower MSE with clip).
    Different markers for F-series vs L-series.

    Args:
        df: Processed DataFrame
        output_path: Path to save the figure
    """
    matplotlib_set_rcparams("paper")

    # Build pairs: one point per technique with both clip and noclip results
    pairs = []
    for technique in df["technique"].unique():
        tech_df = df[df["technique"] == technique]
        clip_row = tech_df[tech_df["clipping"] == "clip_1.0"]
        noclip_row = tech_df[tech_df["clipping"] == "disabled"]

        if not clip_row.empty and not noclip_row.empty:
            pairs.append(
                {
                    "technique": technique,
                    "clip_mse": clip_row.iloc[0]["val_mse"],
                    "noclip_mse": noclip_row.iloc[0]["val_mse"],
                    "series": (
                        "F"
                        if technique.startswith("F")
                        else ("L" if technique.startswith("L") else "other")
                    ),
                }
            )

    if not pairs:
        logger.warning("No clip/noclip pairs found, skipping clipping effect plot")
        return

    pairs_df = pd.DataFrame(pairs)
    colors = get_technique_colors(pairs_df["technique"].tolist())

    fig, ax = plt.subplots(figsize=(6, 6))

    markers = {"F": "s", "L": "o", "other": "D"}

    for _, row in pairs_df.iterrows():
        marker = markers.get(row["series"], "o")
        color = colors.get(row["technique"], "#95a5a6")
        ax.scatter(
            row["clip_mse"],
            row["noclip_mse"],
            marker=marker,
            color=color,
            s=60,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            row["technique"],
            (row["clip_mse"], row["noclip_mse"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    # Diagonal reference (y=x)
    all_vals = list(pairs_df["clip_mse"]) + list(pairs_df["noclip_mse"])
    lim_min = min(all_vals) * 0.95
    lim_max = max(all_vals) * 1.05
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        alpha=0.3,
        linewidth=1,
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel("Val MSE (clip 1.0)")
    ax.set_ylabel("Val MSE (no clip)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend for marker types
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="s", color="gray", linestyle="", markersize=8, label="F-series"),
        Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=8, label="L-series"),
        Line2D([0], [0], marker="D", color="gray", linestyle="", markersize=8, label="Other"),
        Line2D([0], [0], color="k", linestyle="--", alpha=0.3, label="y = x"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.set_title("Phase 2 Global: Clipping Effect")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved clipping effect plot to {output_path}")
    plt.close()


def plot_generalization_gap(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """
    Grouped bars comparing Val vs Test MSE/MAE per experiment.

    Args:
        df: DataFrame with val_mse, val_mae, test_mse, test_mae columns
        output_path: Path to save the figure
        top_n: Number of top techniques to show
    """
    matplotlib_set_rcparams("paper")

    plot_df = df.dropna(subset=["test_mse", "test_mae"]).copy()
    if plot_df.empty:
        logger.warning("No test metrics available, skipping generalization gap plot")
        return

    if len(plot_df) > top_n:
        plot_df = plot_df.nsmallest(top_n, "val_mse")
    plot_df = plot_df.sort_values("val_mse")
    plot_df["label"] = plot_df["technique"] + "\n" + plot_df["clipping"]

    n = len(plot_df)
    fig_width = max(8, n * 0.6)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))

    x = range(n)
    width = 0.35

    for ax, (val_col, test_col, ylabel, title) in zip(
        axes,
        [
            ("val_mse", "test_mse", "MSE", "MSE: Validation vs Test"),
            ("val_mae", "test_mae", "MAE", "MAE: Validation vs Test"),
        ],
    ):
        ax.bar(
            [i - width / 2 for i in x],
            plot_df[val_col],
            width,
            label="Validation",
            color="#3498db",
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            plot_df[test_col],
            width,
            label="Test",
            color="#e74c3c",
            alpha=0.8,
        )

        # Annotate gaps
        for i, (_, row) in enumerate(plot_df.iterrows()):
            gap = row[test_col] - row[val_col]
            ann_color = "#c0392b" if gap > 0 else "#27ae60"
            max_val = max(row[val_col], row[test_col])
            fmt = "+.2e" if "mse" in val_col else "+.3f"
            ax.annotate(
                f"{gap:{fmt}}",
                xy=(i, max_val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color=ann_color,
                rotation=45,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 2 Global: Generalization Gap (top {top_n})", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved generalization gap plot to {output_path}")
    plt.close()


def plot_technique_group_summary(df: pd.DataFrame, output_path: Path) -> None:
    """
    Box plot of val_mse by technique group.

    Shows the distribution of performance within each regularization strategy group,
    with individual data points overlaid.

    Args:
        df: Processed DataFrame
        output_path: Path to save the figure
    """
    matplotlib_set_rcparams("paper")

    df = df.copy()
    df["group"] = df["technique"].map(TECHNIQUE_TO_GROUP)
    df = df[df["group"].notna()]

    if df.empty:
        logger.warning("No group assignments found, skipping technique group summary")
        return

    # Group order matches TECHNIQUE_GROUPS definition
    group_order = list(TECHNIQUE_GROUPS.keys())

    data_by_group = []
    labels = []
    for group in group_order:
        group_df = df[df["group"] == group]
        if not group_df.empty:
            data_by_group.append(group_df["val_mse"].dropna().values)
            labels.append(group)

    if not data_by_group:
        logger.warning("No data for box plot, skipping")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(data_by_group, labels=labels, patch_artist=True, vert=True)

    # Color boxes
    group_colors = ["#bdc3c7", "#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
    for patch, color in zip(bp["boxes"], group_colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, group_data in enumerate(data_by_group):
        x_jitter = rng.uniform(-0.15, 0.15, len(group_data))
        ax.scatter(
            i + 1 + x_jitter,
            group_data,
            alpha=0.5,
            s=20,
            color="black",
            zorder=5,
        )

    ax.set_ylabel("Validation MSE")
    ax.set_title("Phase 2 Global: Technique Group Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved technique group summary to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main(refresh: bool = False, include_test: bool = False, top_n: int = 10) -> None:
    """
    Main analysis function.

    Args:
        refresh: Force refetch from WandB
        include_test: Include test metrics from JSON files (requires Sophia mount)
        top_n: Number of top techniques to show in training curves
    """
    logger.info("=" * 60)
    logger.info("Phase 2 Global: Fine-tuning Technique Comparison")
    logger.info("=" * 60)

    # Fetch data from WandB
    wandb_project_full = WANDB_PROJECTS.get(WANDB_PROJECT, WANDB_PROJECT)
    logger.info(f"Fetching data from WandB project: {wandb_project_full}")
    df = fetch_wandb_runs(
        project=WANDB_PROJECT,
        cache_dir=CACHE_DIR,
        refresh=refresh,
        filters={"state": "finished"},
    )

    if df.empty:
        logger.error("No runs found. Please check WandB project and run status.")
        return

    # Enrich with best (min) validation metrics
    if refresh:
        logger.info("Enriching with best (min) validation metrics from run histories...")
        df = enrich_with_best_metrics(df, project=WANDB_PROJECT)
        cache_path = CACHE_DIR / f"{wandb_project_full}_runs.csv"
        df.to_csv(cache_path, index=False)
        logger.info(f"Re-saved cache with best metrics to {cache_path}")

    # Process data (parses run names, filters to L1 model only)
    df = process_phase2_data(df)
    logger.info(f"Found {len(df)} completed L1 runs")

    # Print parsed summary
    print("\n" + "=" * 60)
    print("PARSED RUN SUMMARY")
    print("=" * 60)
    summary_cols = ["model", "technique", "clipping", "val_mse", "val_mae"]
    print(df[summary_cols].sort_values("val_mse").to_string(index=False))
    print(f"\nTechniques: {sorted(df['technique'].unique(), key=_natural_sort_key)}")
    print(f"Clipping: {sorted(df['clipping'].unique())}")
    print(f"Total experiments: {len(df)}")

    # Add test metrics if requested
    if include_test:
        logger.info("Loading test metrics from experiment directories...")
        df = add_test_metrics(df)
        test_available = df["test_mse"].notna().sum()
        logger.info(f"Loaded test metrics for {test_available}/{len(df)} runs")

    # Add trainable params
    df = add_trainable_params(df)

    # === Tables ===
    logger.info("Creating grouped results table...")
    grouped_df = create_grouped_results_table(df)

    highlight_cols = ["RMSE", "MAE"]

    grouped_latex = create_latex_table(
        grouped_df,
        caption="Results of GNO RANS-AWF transfer learning based on the best weights of the L1 model from the TurbOPark training. Results are grouped by fine-tuning strategy, showing only the best configuration per category. $N$ is the number of models in each category. Best model options lists the LoRA $r$ and $\\alpha$ where applicable, whether weight anchoring was used, and the gradient clipping setting.",
        label="tab:phase2_global_results",
        highlight_columns=highlight_cols,
        highlight_direction="min",
        float_format="%.6f",
    )
    save_table(grouped_latex, OUTPUTS_DIR / f"{OUTPUT_PREFIX}_results.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(grouped_df, OUTPUTS_DIR / f"{OUTPUT_PREFIX}_results.csv")

    # Flat ranking
    logger.info("Creating flat ranking table...")
    flat_df = create_flat_ranking(df)
    save_dataframe_csv(flat_df, OUTPUTS_DIR / f"{OUTPUT_PREFIX}_flat_ranking.csv")

    # Trainable params table
    if df["total_params"].notna().any():
        logger.info("Creating trainable params table...")
        params_df = create_trainable_params_table(df)
        params_latex = create_latex_table(
            params_df,
            caption="Trainable parameter counts per fine-tuning technique.",
            label="tab:phase2_global_params",
            float_format="%.1f",
        )
        save_table(
            params_latex,
            OUTPUTS_DIR / f"{OUTPUT_PREFIX}_trainable_params.tex",
            warning_url=GITLAB_URL,
        )
        save_dataframe_csv(params_df, OUTPUTS_DIR / f"{OUTPUT_PREFIX}_trainable_params.csv")
    else:
        logger.info("No parameter count data available, skipping trainable params table")

    # === Figures ===
    logger.info("Creating technique comparison figure...")
    plot_technique_comparison(df, FIGURES_DIR / f"{OUTPUT_PREFIX}_technique_comparison.pdf")

    logger.info("Creating validation history figure...")
    plot_validation_history(
        df, FIGURES_DIR / f"{OUTPUT_PREFIX}_validation_history.pdf", top_n=top_n
    )

    logger.info("Creating clipping effect figure...")
    plot_clipping_effect(df, FIGURES_DIR / f"{OUTPUT_PREFIX}_clipping_effect.pdf")

    logger.info("Creating technique group summary figure...")
    plot_technique_group_summary(df, FIGURES_DIR / f"{OUTPUT_PREFIX}_technique_groups.pdf")

    # Generalization gap (only with test metrics)
    if include_test and df["test_mse"].notna().any():
        logger.info("Creating generalization gap figure...")
        plot_generalization_gap(
            df, FIGURES_DIR / f"{OUTPUT_PREFIX}_generalization_gap.pdf", top_n=top_n
        )

    logger.info("=" * 60)
    logger.info("Phase 2 Global analysis complete!")
    logger.info(f"Outputs saved to: {OUTPUTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Global Fine-tuning Results Analysis")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refetch from WandB (ignore cache)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test metrics from JSON files (requires Sophia mount)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top techniques to show in training curves (default: 10)",
    )

    args = parser.parse_args()
    main(refresh=args.refresh, include_test=args.include_test, top_n=args.top_n)
