"""This file cannot be run by users, however the resultant table is saved to assets/top_five_models.tex"""

import json
import os
import warnings

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig

from utils.misc import add_to_hydra_cfg

warnings.simplefilter(action="ignore", category=FutureWarning)


top_five_configs = [
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-20/08-29-21/1/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/2/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-04/15-16-02/5/.hydra",
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/4/.hydra",
]

assets_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets")
)


# load grid as text file
grid_search_results_path = os.path.join("assets", "grid_search_results.tex")
# load as lines
with open(grid_search_results_path, "r") as f:
    grid_search_lines = f.readlines()

# find line idx for header row
header_row_idx = None
for i, line in enumerate(grid_search_lines):
    if line.startswith("\\hline") and i + 1 < len(grid_search_lines):
        header_row_idx = i + 1
        break
header_row_idx = header_row_idx + 2


top_models = grid_search_lines[
    header_row_idx : header_row_idx + 5
]  # display top five rows


pd_metrics = pd.DataFrame()
for i, (config_path, model_line) in enumerate(zip(top_five_configs, top_models)):

    model_desc = model_line.split("&")[0:3]
    grid_search_no = model_desc[0]
    model_ID = model_desc[1]
    optimizer_ID = model_desc[2]

    config_name = "config"
    output_dir = os.path.dirname(os.path.abspath(config_path))

    with initialize(version_base="1.3", config_path=os.path.relpath(config_path)):
        cfg = compose(config_name=config_name)

    cfg = add_to_hydra_cfg(cfg, "wandb", DictConfig({"use": False}))
    cfg = add_to_hydra_cfg(cfg, "model_save_path", os.path.join(output_dir, "model"))

    error_metrics_path = os.path.join(cfg.model_save_path, "error_metrics.json")
    with open(error_metrics_path, "r") as f:
        error_metrics = json.load(f)
    df = pd.DataFrame(error_metrics)
    df["Grid search"] = grid_search_no.strip()
    df["Model ID"] = model_ID.strip()
    df["Optimizer ID"] = optimizer_ID.strip()
    pd_metrics = pd.concat([pd_metrics, df], ignore_index=True)

# rename columns
pd_metrics = pd_metrics.rename(
    columns={
        "test/best/mse": "MSE",
        "test/best/mae": "MAE",
        "test/best/rmse": "RMSE",
        "test/best/mape": "MAPE",
    }
)


# order the columns
column_order = [
    "Grid search",
    "Model ID",
    "Optimizer ID",
    "MSE",
    "MAE",
    "RMSE",
    "MAPE",
]

pd_metrics = pd_metrics[column_order]
round_decimals = 3


best_mse = pd_metrics["MSE"].min()
pd_metrics["MSE"] = pd_metrics["MSE"].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mse else x
)

best_mae = pd_metrics["MAE"].min()
pd_metrics["MAE"] = pd_metrics["MAE"].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mae else x
)

best_rmse = pd_metrics["RMSE"].min()
pd_metrics["RMSE"] = pd_metrics["RMSE"].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_rmse else x
)

best_mape = pd_metrics["MAPE"].min()
pd_metrics["MAPE"] = pd_metrics["MAPE"].apply(
    lambda x: f"\\textbf{{{x}}}" if x == best_mape else x
)


# find the places with bold and round them separately
for col in ["MSE", "MAE", "RMSE", "MAPE"]:
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
    caption="Test set error metrics for the five best performing models based on the grid search, with the best metrics marked in bold.",
    label="tab:top_five_models",
    float_format="%.3f",
    column_format=r"p{.8cm}p{.8cm}p{1.1cm}p{.7cm}p{.7cm}p{.7cm}p{.7cm}",
    escape=False,
)

latex_string_results = latex_string_results.replace(r"\toprule", r"\hline")
latex_string_results = latex_string_results.replace(r"\midrule", r"\hline")
latex_string_results = latex_string_results.replace(r"\bottomrule", r"\hline")


# add warning line
latex_string_results = (
    r"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/top_five_table.py?ref_type=heads "
    + "\n"
    + latex_string_results
)

top_five_models_path = os.path.join(".", "assets", "top_five_models.tex")

# write to file
with open(os.path.abspath(top_five_models_path), "w") as f:
    f.write(latex_string_results)
