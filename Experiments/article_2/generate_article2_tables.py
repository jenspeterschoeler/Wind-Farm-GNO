"""
Generate Article 2 publication-ready LaTeX tables.

Article 2 focuses on transfer learning for wind farm flow prediction:
- Phase 1: Architecture search on TurbOPark (pretraining)
- Phase 2: Transfer learning technique comparison on AWF

Generates 6 tables:
1. Architecture + Phase 1 Results (combined, 7 rows sorted by MSE, with training footnote)
2. Training Hyperparameters (Pretraining vs Fine-tuning)
3. Transfer Learning Techniques
4. Phase 2 Results - Transfer Learning Comparison (Multi-column, 5 techniques)
5. Phase 2 Grouped Summary - All 25 techniques grouped by regularization strategy (6 rows)
6. Phase 2 Full Results - Appendix table with all 25 techniques (25 rows)

Usage:
    cd Experiments/article_2
    python generate_article2_tables.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase2_global_results import (  # noqa: E402
    TECHNIQUE_GROUPS,
    TECHNIQUE_TO_GROUP,
)

from utils.reporting.latex_tables import (  # noqa: E402
    create_latex_table,
    highlight_best,
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
OUTPUTS_DIR.mkdir(exist_ok=True)

GITLAB_URL = "https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/article_2/generate_article2_tables.py"

# Scale stats paths
PHASE1_SCALE_STATS = REPO_ROOT / "assets" / "best_model_L2_global" / "scale_stats.json"
PHASE2_SCALE_STATS = REPO_ROOT / "assets" / "best_model_L1_L3_phase2" / "scale_stats.json"

# Data source paths
PHASE1_RESULTS_CSV = OUTPUTS_DIR / "phase1_global_2500layouts_results.csv"
PHASE1_CONFIGS_CSV = OUTPUTS_DIR / "phase1_global_2500layouts_model_configs.csv"
PHASE1_TEST_METRICS_CSV = OUTPUTS_DIR / "phase1_test_metrics.csv"
PHASE2_RESULTS_CSV = OUTPUTS_DIR / "phase2_global_results.csv"
PHASE2_FLAT_CSV = OUTPUTS_DIR / "phase2_global_flat_ranking.csv"
PHASE2_RECALL_CSV = OUTPUTS_DIR / "phase2_recall_metrics.csv"
WANDB_CACHE_CSV = SCRIPT_DIR / "cache" / "GNO_phase2_global_runs.csv"

# Technique descriptions for full 25-row appendix table
TECHNIQUE_DESCRIPTIONS = {
    "SCRATCH": "Random initialization",
    "F1": "Full fine-tuning",
    "F2": "Freeze embedder",
    "F3": "Freeze embedder + processor",
    "F4": "Decoder only trainable",
    "F5": "Full + anchoring ($\\lambda{=}0.01$)",
    "F6": "Freeze emb.\\ + anchoring ($\\lambda{=}0.001$)",
    "L1": "LoRA $r{=}4$",
    "L2": "LoRA $r{=}8$",
    "L3": "LoRA $r{=}16$",
    "L4": "LoRA $r{=}8$ + freeze emb.",
    "L5": "LoRA $r{=}4$ + freeze emb.",
    "L6": "LoRA $r{=}4$ + anchoring",
    "L7": "LoRA $r{=}16$ + freeze emb.",
    "L8": "LoRA $r{=}8$ + anchoring",
    "L9": "LoRA $r{=}16$ + anchoring",
    "L10": "LoRA $r{=}4$ + freeze emb.\\ + anchoring",
    "L11": "LoRA $r{=}8$ + freeze emb.\\ + anchoring",
    "L12": "LoRA $r{=}16$ + freeze emb.\\ + anchoring",
    "L13": "LoRA $r{=}4$ + freeze emb.\\ + proc.",
    "L14": "LoRA $r{=}8$ + freeze emb.\\ + proc.",
    "L15": "LoRA $r{=}16$ + freeze emb.\\ + proc.",
    "L16": "LoRA $r{=}4$ + freeze emb.\\ + proc.\\ + anchoring",
    "L17": "LoRA $r{=}8$ + freeze emb.\\ + proc.\\ + anchoring",
    "L18": "LoRA $r{=}16$ + freeze emb.\\ + proc.\\ + anchoring",
}


def _fmt_time(hours: float) -> str:
    """Format wall time as 'Xh YYm' string."""
    if pd.isna(hours):
        return "--"
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h\\,{m:02d}m"


def _prepare_phase2_best_per_technique(velocity_range: float) -> pd.DataFrame:
    """Load Phase 2 flat ranking, collapse clipping, compute physical units.

    Returns DataFrame with one row per technique (best val_mse), with columns:
    technique, rmse_phys, mae_phys, runtime_seconds, wall_time_hours.
    """
    flat_df = pd.read_csv(PHASE2_FLAT_CSV)
    flat_df = flat_df[flat_df["model"] == "L1"].copy()

    # Collapse clipping: keep best val_mse per technique
    best = flat_df.sort_values("val_mse").groupby("technique").first().reset_index()

    # Compute physical-unit metrics
    best["rmse_phys"] = best["val_rmse"] * velocity_range
    best["mae_phys"] = best["val_mae"] * velocity_range

    # Wall time in hours
    if "runtime_seconds" in best.columns:
        best["wall_time_hours"] = best["runtime_seconds"] / 3600
    else:
        best["wall_time_hours"] = np.nan

    return best


# =============================================================================
# Scale Stats
# =============================================================================
def load_velocity_range(scale_stats_path: Path) -> float:
    """Load velocity range from scale_stats.json for unscaling."""
    with open(scale_stats_path) as f:
        stats = json.load(f)
    vrange = stats["velocity"]["range"]
    if isinstance(vrange, list):
        vrange = vrange[0]
    return float(vrange)


# =============================================================================
# Table 1: Architecture + Phase 1 Results (Combined)
# =============================================================================
def generate_combined_table(velocity_range: float) -> tuple[str, pd.DataFrame]:
    """Generate combined architecture + Phase 1 results table.

    Uses validation metrics from phase1_results CSV (unscaled to physical units).
    Reports only RMSE and MAE.

    Shows all 7 architectures with their best Phase 1 metrics (across 2 learning
    rates), sorted by RMSE. Models whose best result used lr=1e-3 are marked
    with an asterisk.
    """
    logger.info("Generating Combined Architecture + Phase 1 Results Table")

    # Architecture specs
    architectures = {
        "S3": {"Q": 32, "q_int": 80, "L_int": 2, "M_wt": 2, "q_dec": 112, "L_dec": 2},
        "S2": {"Q": 50, "q_int": 125, "L_int": 2, "M_wt": 2, "q_dec": 175, "L_dec": 2},
        "S1": {"Q": 64, "q_int": 160, "L_int": 2, "M_wt": 3, "q_dec": 224, "L_dec": 2},
        "Vj8": {"Q": 100, "q_int": 250, "L_int": 2, "M_wt": 3, "q_dec": 350, "L_dec": 3},
        "L1": {"Q": 128, "q_int": 320, "L_int": 2, "M_wt": 4, "q_dec": 448, "L_dec": 4},
        "L2": {"Q": 160, "q_int": 400, "L_int": 3, "M_wt": 4, "q_dec": 560, "L_dec": 4},
        "L3": {"Q": 180, "q_int": 450, "L_int": 3, "M_wt": 4, "q_dec": 630, "L_dec": 4},
    }

    # Map display names to CSV model names (CSV uses uppercase VJ8)
    csv_name_map = {"Vj8": "VJ8"}

    # Read Phase 1 validation results
    df = pd.read_csv(PHASE1_RESULTS_CSV)

    # Determine RMSE column: compute from Val MSE if Val RMSE not present
    if "Val RMSE" not in df.columns and "Val MSE" in df.columns:
        df["Val RMSE"] = np.sqrt(df["Val MSE"])

    # Build rows: best result per model
    rows = []
    arch_cols = ["Q", "q_int", "L_int", "M_wt", "q_dec", "L_dec"]
    for model in architectures:
        csv_model = csv_name_map.get(model, model)
        arch = architectures[model]

        model_rows = df[df["Model"] == csv_model]
        best_row = model_rows.loc[model_rows["Val RMSE"].idxmin()]
        is_lr001 = "lr001" in str(best_row.get("Reg.", ""))
        rmse = best_row["Val RMSE"] * velocity_range
        mae = best_row["Val MAE"] * velocity_range

        display_name = model + "$^*$" if is_lr001 else model

        rows.append(
            {
                "Model": display_name,
                **{c: arch[c] for c in arch_cols},
                "RMSE": rmse,
                "MAE": mae,
            }
        )

    result_df = pd.DataFrame(rows)
    metric_cols = ["RMSE", "MAE"]
    result_df = result_df.sort_values("RMSE").reset_index(drop=True)

    # Highlight best (minimum) value in each metric column
    formatted = {}
    for col in metric_cols:
        formatted[col] = highlight_best(
            result_df[col], direction="min", format_str="\\textbf{{{:.4f}}}"
        )

    msub = "_{\\mathrm{val}}"

    # Build LaTeX manually (matching existing table style)
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Engineering model architecture search on TurbOPark dataset, "
        "evaluated on validation set. "
        "Best result per architecture across 14 configurations "
        "(7 architectures $\\times$ 2 initial learning rates). "
        "Best value per metric column in bold.}"
    )
    lines.append("\\label{tab:architecture_results}")
    lines.append("\\begin{tabular}{l|cccccc|cc}")
    lines.append("\\hline")
    lines.append(
        "Model & $Q$ & $q_{\\mathrm{int}}$ & $L_{\\mathrm{int}}$ "
        "& $M_{\\mathrm{wt}}$ & $q_{\\mathrm{dec}}$ & $L_{\\mathrm{dec}}$ "
        f"& RMSE${msub}$ & MAE${msub}$ \\\\"
    )
    lines.append(
        " & latent dim. & hidden width & MLP layers "
        "& MP steps & dec.\\ width & dec.\\ layers "
        "& [$\\mathrm{m\\,s^{-1}}$] "
        "& [$\\mathrm{m\\,s^{-1}}$] \\\\"
    )
    lines.append("\\hline")

    for i, row in result_df.iterrows():
        model = row["Model"]
        arch_vals = " & ".join(str(int(row[c])) for c in arch_cols)
        rmse = formatted["RMSE"].iloc[i]
        mae = formatted["MAE"].iloc[i]
        lines.append(f"{model} & {arch_vals} & {rmse} & {mae} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    # Footnote for asterisk
    lines.append("\\par\\vspace{0.5em}")
    lines.append("\\footnotesize")
    lines.append("$^*$Best result obtained with initial learning rate $10^{-3}$.")

    lines.append("\\end{table}")

    latex = "\n".join(lines)
    return latex, result_df


# =============================================================================
# Table 2: Training Hyperparameters
# =============================================================================
def generate_training_hyperparams_table() -> tuple[str, pd.DataFrame]:
    """Generate Table 2: Training hyperparameters (pretraining vs fine-tuning)."""
    logger.info("Generating Table 2: Training Hyperparameters")

    # Define rows as (parameter, phase1_value, phase2_value)
    rows = [
        ("Dataset", "TurbOPark", "AWF"),
        ("Optimizer", "Adam", "Adam"),
        ("Epochs", "3,000", "1,000"),
        (
            "Init.\\ learning rate",
            "$5\\times10^{-3}$",
            "$5\\times10^{-5}$ ($10^{-4}$ for LoRA)",
        ),
        ("LR schedule", "Piecewise const.$^\\dagger$", "Constant"),
        ("Early stop patience", "1,000", "200"),
        ("Dropout rate", "0.1", "---"),
        ("Layer normalization", "Yes", "---"),
        ("Probes per batch $n_p$", "500", "500"),
        ("Graphs per batch $n_G$", "1", "1"),
    ]

    # Build DataFrame for CSV export
    df = pd.DataFrame(rows, columns=["Parameter", "Phase 1 (Pretraining)", "Phase 2 (Fine-tuning)"])

    # Build LaTeX manually for clean formatting
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Training hyperparameters for pretraining (Phase~1) and "
        "fine-tuning (Phase~2). $^\\dagger$Piecewise constant: boundaries at "
        "epochs [200, 350], scale factor 0.1 at each boundary.}"
    )
    lines.append("\\label{tab:training_hyperparams}")
    lines.append("\\begin{tabular}{l|cc}")
    lines.append("\\hline")
    lines.append(
        "Parameter & \\multicolumn{1}{c}{Phase 1 (Pretraining)} "
        "& \\multicolumn{1}{c}{Phase 2 (Fine-tuning)} \\\\"
    )
    lines.append("\\hline")
    for param, p1, p2 in rows:
        lines.append(f"{param} & {p1} & {p2} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    return latex, df


# =============================================================================
# Table 3: Transfer Learning Techniques
# =============================================================================
def generate_techniques_table() -> tuple[str, pd.DataFrame]:
    """Generate Table 3: Transfer learning techniques description."""
    logger.info("Generating Table 3: Transfer Learning Techniques")

    # Try to get trainable % from WandB cache
    trainable_pct = _estimate_trainable_percentages()

    techniques = [
        {
            "ID": "TL1",
            "Technique": "Full fine-tuning",
            "Trainable Components": "All parameters",
            "Key Parameters": "---",
            "Trainable (\\%)": "100",
        },
        {
            "ID": "TL2",
            "Technique": "Decoder-only fine-tuning",
            "Trainable Components": "Decoder MLP",
            "Key Parameters": "---",
            "Trainable (\\%)": trainable_pct.get("F4", "$\\sim$30"),
        },
        {
            "ID": "TL3",
            "Technique": "Full fine-tuning + $L_2$ anchor",
            "Trainable Components": "All parameters",
            "Key Parameters": "$\\lambda = 0.01$",
            "Trainable (\\%)": "100",
        },
        {
            "ID": "TL4",
            "Technique": "Low-Rank Adaptation (LoRA)",
            "Trainable Components": "LoRA adapters on all Dense layers",
            "Key Parameters": "$r=8$, $\\alpha=16$",
            "Trainable (\\%)": trainable_pct.get("L2", "$\\sim$3"),
        },
    ]
    df = pd.DataFrame(techniques)

    latex = create_latex_table(
        df,
        caption=(
            "Transfer learning techniques evaluated in Phase~2. "
            "Trainable percentage is relative to total model parameters."
        ),
        label="tab:techniques",
        column_format="c|llll",
        escape=False,
    )

    return latex, df


def _estimate_trainable_percentages() -> dict[str, str]:
    """Try to estimate trainable % from WandB cache or return approximate values."""
    result = {}

    if not WANDB_CACHE_CSV.exists():
        logger.info("No WandB cache found, using approximate trainable percentages")
        return result

    try:
        cache_df = pd.read_csv(WANDB_CACHE_CSV)
        # Check for param columns
        param_cols = [c for c in cache_df.columns if "param" in c.lower()]
        if not param_cols:
            logger.info("No param columns in WandB cache, using approximate values")
            return result
    except Exception:
        return result

    return result


# =============================================================================
# Table 5: Phase 2 Results (Multi-column)
# =============================================================================
def generate_phase2_results_table(velocity_range: float) -> tuple[str, pd.DataFrame]:
    """Generate Table 5: Phase 2 results (L1 only) with optional recall columns.

    Uses test metrics (from error_metrics_all.json) when available, falling back
    to unscaled validation metrics otherwise. Test metrics are already in physical
    units (post-processing inverse-scales before computing errors).

    Only shows L1 model results (VJ8 removed). Adds TurbOPark recall columns
    from phase2_recall_metrics.csv when available (catastrophic forgetting).

    "From scratch" gets N/A for recall (no pretraining → no recall concept).
    """
    logger.info("Generating Table 5: Phase 2 Results (L1 only + recall)")

    flat_df = pd.read_csv(PHASE2_FLAT_CSV)

    # Filter to L1 only
    flat_df = flat_df[flat_df["model"] == "L1"].copy()

    # Detect whether test metrics are available
    has_test = "test_mse" in flat_df.columns and flat_df["test_mse"].notna().any()
    if has_test:
        logger.info("Using TEST metrics (already in physical units)")
        sort_col = "test_mse"
    else:
        logger.info("Falling back to VALIDATION metrics (unscaling to physical units)")
        sort_col = "val_mse"

    # Collapse clipping: keep best MSE per technique
    best_per_tech = flat_df.sort_values(sort_col).groupby("technique").first().reset_index()

    # Compute physical-unit columns
    if has_test:
        best_per_tech["mse_phys"] = best_per_tech["test_mse"]
        best_per_tech["mae_phys"] = best_per_tech["test_mae"]
        if "test_rmse" in best_per_tech.columns:
            best_per_tech["rmse_phys"] = best_per_tech["test_rmse"]
        else:
            best_per_tech["rmse_phys"] = np.sqrt(best_per_tech["test_mse"])
    else:
        vr2 = velocity_range**2
        best_per_tech["mse_phys"] = best_per_tech["val_mse"] * vr2
        best_per_tech["mae_phys"] = best_per_tech["val_mae"] * velocity_range
        best_per_tech["rmse_phys"] = best_per_tech["val_rmse"] * velocity_range

    # Load recall metrics if available
    has_recall = PHASE2_RECALL_CSV.exists()
    recall_df = None
    if has_recall:
        recall_df = pd.read_csv(PHASE2_RECALL_CSV)
        # Keep only L1, collapse clipping (best recall_mse per technique)
        recall_df = recall_df[recall_df["model"] == "L1"].copy()
        recall_df = recall_df.sort_values("recall_mse").groupby("technique").first().reset_index()
        logger.info(f"Loaded recall metrics for {len(recall_df)} techniques")
    else:
        logger.info("No recall metrics found, omitting recall columns")

    # Technique display names
    trainable_pct = _estimate_trainable_percentages()
    technique_map = {
        "SCRATCH": ("---", "From scratch", "100"),
        "F1": ("TL1", "Full fine-tuning", "100"),
        "L2": ("TL4", "LoRA ($r{=}8$, $\\alpha{=}16$)", trainable_pct.get("L2", "$\\sim$3")),
        "F4": ("TL2", "Decoder-only", trainable_pct.get("F4", "$\\sim$30")),
        "F5": ("TL3", "$L_2$ anchor ($\\lambda{=}0.01$)", "100"),
    }

    # Check if wall time data is available
    has_runtime = "runtime_seconds" in flat_df.columns and flat_df["runtime_seconds"].notna().any()

    # Build all technique records
    all_records = []
    for tech_code in ["SCRATCH", "F1", "L2", "F4", "F5"]:
        tl_id, tl_name, tl_trainable = technique_map[tech_code]
        record = {"ID": tl_id, "Technique": tl_name, "Trainable": tl_trainable}

        row = best_per_tech[best_per_tech["technique"] == tech_code]
        if not row.empty:
            record["AWF_MSE"] = row.iloc[0]["mse_phys"]
            record["AWF_MAE"] = row.iloc[0]["mae_phys"]
            record["AWF_RMSE"] = row.iloc[0]["rmse_phys"]
            if has_runtime and pd.notna(row.iloc[0].get("runtime_seconds")):
                record["wall_time_hours"] = float(row.iloc[0]["runtime_seconds"]) / 3600
            else:
                record["wall_time_hours"] = np.nan
        else:
            record["AWF_MSE"] = np.nan
            record["AWF_MAE"] = np.nan
            record["AWF_RMSE"] = np.nan
            record["wall_time_hours"] = np.nan

        # Recall metrics (N/A for SCRATCH — no pretraining)
        if has_recall and tech_code != "SCRATCH":
            recall_row = recall_df[recall_df["technique"] == tech_code]
            if not recall_row.empty:
                record["Recall_MSE"] = recall_row.iloc[0]["recall_mse"]
                record["Recall_MAE"] = recall_row.iloc[0]["recall_mae"]
                record["Recall_RMSE"] = recall_row.iloc[0]["recall_rmse"]
            else:
                record["Recall_MSE"] = np.nan
                record["Recall_MAE"] = np.nan
                record["Recall_RMSE"] = np.nan
        else:
            record["Recall_MSE"] = np.nan
            record["Recall_MAE"] = np.nan
            record["Recall_RMSE"] = np.nan

        all_records.append(record)

    pivot_df = pd.DataFrame(all_records)
    pivot_df = pivot_df.sort_values("AWF_MSE").reset_index(drop=True)

    show_time = pivot_df["wall_time_hours"].notna().any()
    show_recall = has_recall and pivot_df["Recall_MSE"].notna().any()

    # Highlight best per column
    fmt = "\\textbf{{{:.4f}}}"
    formatted = {}
    for col in ["AWF_MSE", "AWF_MAE", "AWF_RMSE"]:
        formatted[col] = highlight_best(pivot_df[col], direction="min", format_str=fmt)

    if show_recall:
        for col in ["Recall_MSE", "Recall_MAE", "Recall_RMSE"]:
            formatted[col] = highlight_best(pivot_df[col], direction="min", format_str=fmt)

    # Caption and subscripts
    awf_desc = "AWF test set" if has_test else "AWF dataset (validation)"
    sub = "test" if has_test else "val"
    msub = f"$_{{\\mathrm{{{sub}}}}}$"
    time_note = " Time is training wall time." if show_time else ""
    recall_note = (
        " TurbOPark recall measures catastrophic forgetting on the source domain."
        if show_recall
        else ""
    )

    # Build LaTeX
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Phase~2 transfer learning results. "
        f"AWF columns: {awf_desc} (target domain). "
        "``From scratch'' denotes training from random initialization. "
        "Best result per technique (across gradient clipping variants). "
        f"Metrics in physical units. Best value per column in bold."
        f"{time_note}{recall_note}}}"
    )
    lines.append("\\label{tab:phase2_results}")

    # Determine column format
    n_awf_cols = 3
    n_recall_cols = 3 if show_recall else 0
    time_col = "l" if show_time else ""
    recall_fmt = "|ccc" if show_recall else ""
    lines.append(f"\\begin{{tabular}}{{clc{time_col}|ccc{recall_fmt}}}")
    lines.append("\\hline")

    # Multi-column header row
    awf_header = f"\\multicolumn{{{n_awf_cols}}}{{c"
    awf_header += "|" if show_recall else ""
    awf_header += "}{AWF (target)}"
    recall_header = (
        f" & \\multicolumn{{{n_recall_cols}}}{{c}}{{TurbOPark (recall)}}" if show_recall else ""
    )
    time_header = " & " if show_time else ""
    lines.append(f" & & &{time_header} {awf_header}{recall_header} \\\\")

    # Metric name row
    awf_metrics = f"MSE{msub} & MAE{msub} & RMSE{msub}"
    recall_metrics = f" & MSE{msub} & MAE{msub} & RMSE{msub}" if show_recall else ""
    time_col_header = " & Time" if show_time else ""
    lines.append(
        f" & Technique & Train.\\ (\\%){time_col_header} & {awf_metrics}{recall_metrics} \\\\"
    )

    # Units row
    awf_units = "[$\\mathrm{m^2 s^{-2}}$] & [$\\mathrm{m\\,s^{-1}}$] " "& [$\\mathrm{m\\,s^{-1}}$]"
    recall_units = (
        " & [$\\mathrm{m^2 s^{-2}}$] & [$\\mathrm{m\\,s^{-1}}$] " "& [$\\mathrm{m\\,s^{-1}}$]"
        if show_recall
        else ""
    )
    time_units = " & " if show_time else ""
    lines.append(f" & &{time_units} & {awf_units}{recall_units} \\\\")
    lines.append("\\hline")

    # Data rows
    for i, row in pivot_df.iterrows():
        tl_id = row["ID"]
        technique = row["Technique"]
        trainable = row["Trainable"]
        is_scratch = tl_id == "---"

        # AWF metrics
        awf_vals = " & ".join(
            [
                formatted["AWF_MSE"].iloc[i],
                formatted["AWF_MAE"].iloc[i],
                formatted["AWF_RMSE"].iloc[i],
            ]
        )

        # Recall metrics
        if show_recall:
            if is_scratch or pd.isna(row["Recall_MSE"]):
                recall_vals = " & N/A & N/A & N/A"
            else:
                recall_vals = " & " + " & ".join(
                    [
                        formatted["Recall_MSE"].iloc[i],
                        formatted["Recall_MAE"].iloc[i],
                        formatted["Recall_RMSE"].iloc[i],
                    ]
                )
        else:
            recall_vals = ""

        if show_time:
            time_str = _fmt_time(row["wall_time_hours"])
            lines.append(
                f"{tl_id} & {technique} & {trainable} & {time_str} "
                f"& {awf_vals}{recall_vals} \\\\"
            )
        else:
            lines.append(f"{tl_id} & {technique} & {trainable} & {awf_vals}{recall_vals} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    return latex, pivot_df


# =============================================================================
# Table 5: Phase 2 Grouped Summary (All 25 Techniques)
# =============================================================================
def generate_phase2_grouped_table(velocity_range: float) -> tuple[str, pd.DataFrame]:
    """Generate Phase 2 grouped summary table (6 rows, one per regularization group).

    Columns: Group, N, Best, RMSE_val [m/s], MAE_val [m/s], RMSE range.
    """
    logger.info("Generating Phase 2 Grouped Summary Table (6 groups)")

    best = _prepare_phase2_best_per_technique(velocity_range)

    # Map technique -> group
    best["group"] = best["technique"].map(TECHNIQUE_TO_GROUP)

    # Build one row per group, preserving TECHNIQUE_GROUPS key order
    rows = []
    for group_name, tech_list in TECHNIQUE_GROUPS.items():
        grp = best[best["technique"].isin(tech_list)]
        if grp.empty:
            continue
        best_row = grp.loc[grp["rmse_phys"].idxmin()]
        rows.append(
            {
                "Group": group_name,
                "N": len(tech_list),
                "Best": best_row["technique"],
                "RMSE_val": best_row["rmse_phys"],
                "MAE_val": best_row["mae_phys"],
                "RMSE_min": grp["rmse_phys"].min(),
                "RMSE_max": grp["rmse_phys"].max(),
            }
        )

    result_df = pd.DataFrame(rows)

    # Highlight best (minimum) RMSE and MAE across all groups
    fmt = "\\textbf{{{:.4f}}}"
    formatted_rmse = highlight_best(result_df["RMSE_val"], direction="min", format_str=fmt)
    formatted_mae = highlight_best(result_df["MAE_val"], direction="min", format_str=fmt)

    # Build LaTeX
    msub = "_{\\mathrm{val}}"
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Phase~2 fine-tuning results grouped by regularization strategy. "
        "$N$: number of technique variants per group. "
        "Best technique and metrics per group (best across gradient clipping variants). "
        "RMSE range shows within-group spread. "
        "Best value per metric column in bold.}"
    )
    lines.append("\\label{tab:phase2_grouped}")
    lines.append("\\begin{tabular}{l|clccl}")
    lines.append("\\hline")
    lines.append(f"Group & $N$ & Best & RMSE${msub}$ & MAE${msub}$ & RMSE range \\\\")
    lines.append(
        " & & & [$\\mathrm{m\\,s^{-1}}$] & [$\\mathrm{m\\,s^{-1}}$] "
        "& [$\\mathrm{m\\,s^{-1}}$] \\\\"
    )
    lines.append("\\hline")

    for i, row in result_df.iterrows():
        group = row["Group"]
        n = int(row["N"])
        best_tech = row["Best"]
        rmse = formatted_rmse.iloc[i]
        mae = formatted_mae.iloc[i]
        if row["RMSE_min"] == row["RMSE_max"]:
            rmse_range = f"{row['RMSE_min']:.4f}"
        else:
            rmse_range = f"{row['RMSE_min']:.4f}--{row['RMSE_max']:.4f}"
        lines.append(f"{group} & {n} & {best_tech} & {rmse} & {mae} & {rmse_range} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    return latex, result_df


# =============================================================================
# Table 6: Phase 2 Full Results (Appendix, All 25 Techniques)
# =============================================================================
def generate_phase2_full_table(velocity_range: float) -> tuple[str, pd.DataFrame]:
    """Generate Phase 2 full appendix table (25 rows, one per technique).

    Columns: Group (multirow), Code, Technique, Time, RMSE_val, MAE_val.
    Sorted by group order, then RMSE within each group. Hline between groups.
    """
    logger.info("Generating Phase 2 Full Appendix Table (25 techniques)")

    best = _prepare_phase2_best_per_technique(velocity_range)

    # Map technique -> group
    best["group"] = best["technique"].map(TECHNIQUE_TO_GROUP)

    # Assign group order for sorting
    group_order = {name: i for i, name in enumerate(TECHNIQUE_GROUPS.keys())}
    best["group_order"] = best["group"].map(group_order)
    best = best.sort_values(["group_order", "rmse_phys"]).reset_index(drop=True)

    # Find overall best RMSE and MAE for bold highlighting
    fmt = "\\textbf{{{:.4f}}}"
    formatted_rmse = highlight_best(best["rmse_phys"], direction="min", format_str=fmt)
    formatted_mae = highlight_best(best["mae_phys"], direction="min", format_str=fmt)

    has_time = best["wall_time_hours"].notna().any()

    # Build LaTeX
    msub = "_{\\mathrm{val}}"
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{Phase~2 fine-tuning results for all 25 techniques, "
        "grouped by regularization strategy. "
        "Best result per technique across gradient clipping variants. "
        "Metrics in physical units. Best value per metric column in bold.}"
    )
    lines.append("\\label{tab:phase2_full}")

    time_col_fmt = "r" if has_time else ""
    time_col_hdr = " & Time" if has_time else ""
    lines.append(f"\\begin{{tabular}}{{l|ll{time_col_fmt}cc}}")
    lines.append("\\hline")
    lines.append(f"Group & Code & Technique{time_col_hdr} " f"& RMSE${msub}$ & MAE${msub}$ \\\\")
    time_col_empty = " &" if has_time else ""
    lines.append(
        f" & &{time_col_empty}"
        f" & [$\\mathrm{{m\\,s^{{-1}}}}$] & [$\\mathrm{{m\\,s^{{-1}}}}$] \\\\"
    )
    lines.append("\\hline")

    prev_group = None
    group_counts = best["group"].value_counts()

    for i, row in best.iterrows():
        group = row["group"]
        tech = row["technique"]
        desc = TECHNIQUE_DESCRIPTIONS.get(tech, tech)
        rmse = formatted_rmse.iloc[i]
        mae = formatted_mae.iloc[i]
        time_str = _fmt_time(row["wall_time_hours"]) if has_time else ""

        # Group column: multirow for first row in group, empty for rest
        if group != prev_group:
            if prev_group is not None:
                lines.append("\\hline")
            n_in_group = int(group_counts[group])
            group_cell = f"\\multirow{{{n_in_group}}}{{*}}{{{group}}}"
            prev_group = group
        else:
            group_cell = ""

        time_cell = f" & {time_str}" if has_time else ""
        lines.append(f"{group_cell} & {tech} & {desc}{time_cell} & {rmse} & {mae} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    # Build clean DataFrame for CSV export
    csv_df = best[["group", "technique", "rmse_phys", "mae_phys", "wall_time_hours"]].copy()
    csv_df.columns = ["Group", "Code", "RMSE_val_ms", "MAE_val_ms", "Wall_time_hours"]

    return latex, csv_df


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Generate all Article 2 publication tables."""
    logger.info("=" * 60)
    logger.info("Generating Article 2 Publication Tables")
    logger.info("=" * 60)

    # Load scale stats
    logger.info("Loading scale stats for unscaling...")
    phase1_vrange = load_velocity_range(PHASE1_SCALE_STATS)
    phase2_vrange = load_velocity_range(PHASE2_SCALE_STATS)
    logger.info(f"  Phase 1 (TurbOPark) velocity range: {phase1_vrange:.2f} m/s")
    logger.info(f"  Phase 2 (AWF) velocity range: {phase2_vrange:.2f} m/s")

    # Generate Table 1: Architecture + Phase 1 Results (combined)
    latex1, df1 = generate_combined_table(phase1_vrange)
    save_table(latex1, OUTPUTS_DIR / "article2_architecture_results.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df1, OUTPUTS_DIR / "article2_architecture_results.csv")

    # Generate Table 2: Training Hyperparameters
    latex2, df2 = generate_training_hyperparams_table()
    save_table(latex2, OUTPUTS_DIR / "article2_training_hyperparams.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df2, OUTPUTS_DIR / "article2_training_hyperparams.csv")

    # Generate Table 3: Transfer Learning Techniques
    latex3, df3 = generate_techniques_table()
    save_table(latex3, OUTPUTS_DIR / "article2_techniques.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df3, OUTPUTS_DIR / "article2_techniques.csv")

    # Generate Table 4: Phase 2 Results
    latex4, df4 = generate_phase2_results_table(phase2_vrange)
    save_table(latex4, OUTPUTS_DIR / "article2_phase2_results.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df4, OUTPUTS_DIR / "article2_phase2_results.csv")

    # Generate Table 5: Phase 2 Grouped Summary (all 25 techniques)
    latex5, df5 = generate_phase2_grouped_table(phase2_vrange)
    save_table(latex5, OUTPUTS_DIR / "article2_phase2_grouped.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df5, OUTPUTS_DIR / "article2_phase2_grouped.csv")

    # Generate Table 6: Phase 2 Full Results (Appendix, all 25 techniques)
    latex6, df6 = generate_phase2_full_table(phase2_vrange)
    save_table(latex6, OUTPUTS_DIR / "article2_phase2_full.tex", warning_url=GITLAB_URL)
    save_dataframe_csv(df6, OUTPUTS_DIR / "article2_phase2_full.csv")

    # Summary
    logger.info("=" * 60)
    logger.info("All 6 tables generated successfully!")
    logger.info(f"Output directory: {OUTPUTS_DIR}")
    logger.info("Files created:")
    for name in [
        "article2_architecture_results",
        "article2_training_hyperparams",
        "article2_techniques",
        "article2_phase2_results",
        "article2_phase2_grouped",
        "article2_phase2_full",
    ]:
        logger.info(f"  {name}.tex")
        logger.info(f"  {name}.csv")
    logger.info("=" * 60)

    # Print physical unit sanity check
    print("\n" + "=" * 60)
    print("SANITY CHECK: Physical unit values")
    print("=" * 60)
    print(f"\nArchitecture + Phase 1 (velocity_range = {phase1_vrange:.2f} m/s):")
    print(df1.to_string(index=False))
    print(f"\nPhase 2 (velocity_range = {phase2_vrange:.2f} m/s):")
    print(df4.to_string(index=False))
    print(f"\nPhase 2 Grouped (velocity_range = {phase2_vrange:.2f} m/s):")
    print(df5.to_string(index=False))
    print(f"\nPhase 2 Full (velocity_range = {phase2_vrange:.2f} m/s):")
    print(df6.to_string(index=False))


if __name__ == "__main__":
    main()
