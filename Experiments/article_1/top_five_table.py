"""Generate top five models table from grid search results.

This script can be run either on Sophia HPC or locally with Sophia mounted.
The resultant table is saved to assets/top_five_models.tex
"""

import json
import os
import sys
import warnings
from pathlib import Path

# Add project root to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import pandas as pd  # noqa: E402
from hydra import compose, initialize  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from utils.misc import add_to_hydra_cfg  # noqa: E402

warnings.simplefilter(action="ignore", category=FutureWarning)

# Path prefixes for Sophia (on cluster) vs local mount
SOPHIA_PREFIX = "/work/users/jpsch/SPO_sophia_dir/"
LOCAL_MOUNT_PREFIX = "/home/jpsch/Documents/Sophia_work/SPO_sophia_dir/"


def get_available_path(sophia_path: str) -> str:
    """Convert Sophia path to local mount path if running locally."""
    # First try the original Sophia path
    if os.path.exists(sophia_path):
        return sophia_path
    # Try local mount path
    local_path = sophia_path.replace(SOPHIA_PREFIX, LOCAL_MOUNT_PREFIX)
    if os.path.exists(local_path):
        return local_path
    raise FileNotFoundError(
        f"Path not found on Sophia or local mount:\n"
        f"  Sophia: {sophia_path}\n"
        f"  Local:  {local_path}"
    )


top_five_configs_sophia = [
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-20/08-29-21/1/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/2/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-04/15-16-02/5/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/4/.hydra",
]

# Convert to available paths (Sophia or local mount)
top_five_configs = [get_available_path(p) for p in top_five_configs_sophia]

assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


# load grid as text file
grid_search_results_path = os.path.join("assets", "grid_search_results.tex")
# load as lines
with open(grid_search_results_path) as f:
    grid_search_lines = f.readlines()

# find line idx for first data row (after second \hline)
hline_count = 0
data_row_idx = None
for i, line in enumerate(grid_search_lines):
    if line.startswith("\\hline"):
        hline_count += 1
        if hline_count == 2:
            data_row_idx = i + 1
            break


top_models = grid_search_lines[data_row_idx : data_row_idx + 5]  # display top five rows


pd_metrics = pd.DataFrame()
for _i, (config_path, model_line) in enumerate(zip(top_five_configs, top_models)):
    model_desc = model_line.split("&")[0:3]
    grid_search_no = model_desc[0]
    model_ID = model_desc[1]
    optimizer_ID = model_desc[2]

    config_name = "config"
    config_path_abs = os.path.abspath(config_path)
    output_dir = os.path.dirname(config_path_abs)

    # Hydra requires relative path from the calling module's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path_rel = os.path.relpath(config_path_abs, script_dir)

    with initialize(version_base="1.3", config_path=config_path_rel):
        cfg = compose(config_name=config_name)

    cfg = add_to_hydra_cfg(cfg, "wandb", DictConfig({"use": False}))
    # Model save path is in the output directory
    model_save_path = os.path.join(output_dir, "model")
    cfg = add_to_hydra_cfg(cfg, "model_save_path", model_save_path)

    error_metrics_path = os.path.join(cfg.model_save_path, "error_metrics.json")
    with open(error_metrics_path) as f:
        error_metrics = json.load(f)
    df = pd.DataFrame(error_metrics)
    df["Grid search"] = grid_search_no.strip()
    df["Model ID"] = model_ID.strip()
    df["Opt. ID"] = optimizer_ID.strip()
    pd_metrics = pd.concat([pd_metrics, df], ignore_index=True)

# rename columns with units (test metrics have physical units)
# MSE: m²s⁻², MAE/RMSE: ms⁻¹, MANE: dimensionless (normalized)
pd_metrics = pd_metrics.rename(
    columns={
        "test/best/mse": r"MSE [$\mathrm{m^2s^{-2}}$]",
        "test/best/mae": r"MAE [$\mathrm{ms^{-1}}$]",
        "test/best/rmse": r"RMSE [$\mathrm{ms^{-1}}$]",
        "test/best/mape": r"MANE [\%]",
    }
)


# order the columns
column_order = [
    "Grid search",
    "Model ID",
    "Opt. ID",
    r"MSE [$\mathrm{m^2s^{-2}}$]",
    r"MAE [$\mathrm{ms^{-1}}$]",
    r"RMSE [$\mathrm{ms^{-1}}$]",
    r"MANE [\%]",
]

pd_metrics = pd_metrics[column_order]

# Load no-wake baseline metrics (if available) - stored separately for multicolumn row
baseline_path = os.path.join(assets_path, "no_wake_baseline", "error_metrics.json")
baseline_metrics = None
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline_metrics = json.load(f)
else:
    print(f"Warning: Baseline metrics not found at {baseline_path}")
    print("Run compute_no_wake_baseline.py on Sophia first.")

round_decimals = 3


mse_col = r"MSE [$\mathrm{m^2s^{-2}}$]"
mae_col = r"MAE [$\mathrm{ms^{-1}}$]"
rmse_col = r"RMSE [$\mathrm{ms^{-1}}$]"
mane_col = r"MANE [\%]"

best_mse = pd_metrics[mse_col].min()
pd_metrics[mse_col] = pd_metrics[mse_col].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mse else x
)

best_mae = pd_metrics[mae_col].min()
pd_metrics[mae_col] = pd_metrics[mae_col].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mae else x
)

best_rmse = pd_metrics[rmse_col].min()
pd_metrics[rmse_col] = pd_metrics[rmse_col].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_rmse else x
)

best_mane = pd_metrics[mane_col].min()
pd_metrics[mane_col] = pd_metrics[mane_col].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mane else x
)


# find the places with bold and round them separately
for col in [mse_col, mae_col, rmse_col, mane_col]:
    col_interest = pd_metrics[col]
    for i in range(len(col_interest)):
        val = col_interest.iloc[i]
        if isinstance(val, str) and "\\textbf{" in val:
            # extract the number
            num_str = val.replace("\\textbf{", "").replace("}", "")
            num = float(num_str)
            # round the number
            rounded_num = round(num, round_decimals)
            # put it back in bold
            pd_metrics.at[i, col] = f"\\textbf{{{rounded_num}}}"
        else:
            pd_metrics.at[i, col] = round(val, round_decimals)


# export to latex
latex_string_results = pd_metrics.to_latex(
    index=False,
    caption=r"Test set error metrics for the five best performing models based on the grid search and a no-wake baseline, with the best metrics marked in bold. The naive baseline predicts that the flow everywhere is equal to the free stream velocity (i.e., $ u' = U$).",
    label="tab:top_five_models",
    float_format="%.3f",
    column_format=r"p{.7cm}p{.6cm}p{.5cm}p{1.3cm}p{1.1cm}p{1.1cm}p{.8cm}",
    escape=False,
    position="htb",
)

latex_string_results = latex_string_results.replace(r"\toprule", r"\hline")
latex_string_results = latex_string_results.replace(r"\midrule", r"\hline")
latex_string_results = latex_string_results.replace(r"\bottomrule", r"\hline")

# Add baseline row with multicolumn spanning first 3 columns
if baseline_metrics is not None:

    def get_scalar(val):
        return val[0] if isinstance(val, list) else val

    baseline_mse = get_scalar(baseline_metrics["test/best/mse"])
    baseline_mae = get_scalar(baseline_metrics["test/best/mae"])
    baseline_rmse = get_scalar(baseline_metrics["test/best/rmse"])
    baseline_mape = get_scalar(baseline_metrics["test/best/mape"])

    baseline_row_latex = (
        r"\hline" + "\n"
        r"\multicolumn{3}{p{2.5cm}}{Naive free stream baseline} & "
        f"{baseline_mse:.4g} & {baseline_mae:.4g} & {baseline_rmse:.4g} & {baseline_mape:.4g} \\\\\n"
    )
    # Insert before the final \hline
    # Find the last \hline and insert before it
    last_hline_idx = latex_string_results.rfind(r"\hline")
    latex_string_results = (
        latex_string_results[:last_hline_idx]
        + baseline_row_latex
        + latex_string_results[last_hline_idx:]
    )

# add warning line
latex_string_results = (
    r"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/top_five_table.py?ref_type=heads"
    + "\n"
    + latex_string_results
)

top_five_models_path = os.path.join(".", "assets", "top_five_models.tex")

# write to file
with open(os.path.abspath(top_five_models_path), "w") as f:
    f.write(latex_string_results)
