"""
Wind turbine error analysis script for Article 1.

This script contains wind turbine error analysis and visualization. The
publication figure (wt_spatial_errors_all_layouts) has been moved to
plot_publication_figures.py which uses cached data.

Kept in this script:
- Error metrics bar chart (RMSE, MAE, MAPE, etc.)
- Error distribution scatter plots
- Prediction accuracy plots
- Error histograms

For publication figures, see:
    - plot_publication_figures.py (generates from cached data)
    - generate_plot_data.py (generates cache files on Sophia)
"""

import json
import os
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from omegaconf import DictConfig
from py_wake.examples.data.dtu10mw import DTU10MW

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

# Import shared functions from master script
from plot_publication_figures import (
    get_max_plot_distance,
    get_model_paths,
    select_representative_layouts,
    setup_plot_iterator,
)

from utils.data_tools import (
    retrieve_dataset_stats,
    setup_unscaler,
)
from utils.plotting import matplotlib_set_rcparams
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

matplotlib_set_rcparams("paper")

# Same model path as in the original script
# main_path = os.path.abspath(
#     "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
# )  # sophia
main_path = "./assets/best_model_Vj8"  # local, run from repo root
# main_path = "../../assets/best_model_Vj8"  # local, run from this folder (for ipython)


if "assets" in main_path:
    running_locally = True
else:
    running_locally = False


if running_locally:
    test_data_path = os.path.abspath(
        "./data/zenodo_graphs/test_pre_processed"
    )  # Download dataset and place in data directory: https://doi.org/10.5281/zenodo.17671257
else:
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )

# Load paths and configs using shared function
paths = get_model_paths(main_path)
cfg_path = paths["cfg_path"]
model_cfg_path = paths["model_cfg_path"]
params_paths = paths["params_path"]

if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"

fig_folder_path = os.path.join(main_path, "model/figures_wt_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)

# Load model config
with open(model_cfg_path) as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)

# Set test data path
if not running_locally:
    test_data_path = restored_cfg_model.data.test_path  # Sophia server
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )

dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

# Find representative layouts using shared function
layout_type_idxs = select_representative_layouts(dataset)
plot_distance, max_y_range = get_max_plot_distance(dataset, layout_type_idxs)

# Setup plot iterator using shared function
plot_graphs, test_dataset = setup_plot_iterator(
    restored_cfg_model, test_data_path, dataset, layout_type_idxs
)

# Load model
restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
    params_paths, model_cfg_path, dataset
)

stats, scale_stats = retrieve_dataset_stats(dataset)
unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
_inverse_scale_target = unscaler.inverse_scale_output

wt = DTU10MW()
D = wt.diameter()

pred_fn = jax.jit(model.apply)


# Define function to analyze wind turbine errors
def analyze_wt_errors(graphs, probe_graphs, targets, wt_mask, probe_mask, model_params, unscaler):
    """
    Analyze wind turbine prediction errors.
    Returns dictionary with error metrics for wind turbines.
    """
    prediction = pred_fn(
        model_params,
        graphs,
        probe_graphs,
        jnp.atleast_2d(wt_mask),
        jnp.atleast_2d(probe_mask),
    ).squeeze()

    wt_idx = np.where(wt_mask != 0)[0]

    # Get predictions and targets for wind turbines
    wt_predictions = _inverse_scale_target(prediction[wt_idx]).squeeze()
    wt_targets = _inverse_scale_target(targets[wt_idx]).squeeze()
    wt_errors = wt_targets - wt_predictions

    # Calculate error metrics
    mse = np.mean(wt_errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(wt_errors))
    mape = np.mean(np.abs(wt_errors / wt_targets)) * 100  # percentage

    # For reference, get free stream velocity from the graphs global attributes
    U_free = unscaler.inverse_scale_graph(graphs).globals[0][0]

    # Calculate error as percentage of free stream velocity
    rel_error_U = (np.abs(wt_errors) / U_free) * 100
    mean_rel_error_U = np.mean(rel_error_U)
    max_rel_error_U = np.max(rel_error_U)

    return {
        "wt_predictions": wt_predictions,
        "wt_targets": wt_targets,
        "wt_errors": wt_errors,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mean_rel_error_U": mean_rel_error_U,
        "max_rel_error_U": max_rel_error_U,
        "U_free": U_free,
    }


# Analyze errors for each layout type
layout_errors = {}
for layout_type, val in plot_graphs.items():
    graphs, probe_graphs, node_array_tuple, layout_type_str, wt_spacing = val
    targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

    error_dict = analyze_wt_errors(
        graphs, probe_graphs, targets, wt_mask, probe_mask, restored_params, unscaler
    )

    layout_errors[layout_type] = {
        "error_metrics": error_dict,
        "graphs": graphs,
        "probe_graphs": probe_graphs,
        "node_positions": node_positions,
        "wt_mask": wt_mask,
        "probe_mask": probe_mask,
        "wt_spacing": wt_spacing,
        "n_wts": int(np.sum(wt_mask)),
    }

# %% Plot wind turbine error metrics for each layout type (KEPT - different visualization)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

metric_names = ["mse", "rmse", "mae", "mape", "mean_rel_error_U", "max_rel_error_U"]

# Create bar chart of error metrics
metrics_for_plot = ["rmse", "mae", "mape", "mean_rel_error_U"]
labels = ["RMSE (m/s)", "MAE (m/s)", "MAPE (%)", "Mean Rel. Error (% of U)"]

x = np.arange(len(metrics_for_plot))
width = 0.2

for i, (layout_type, data) in enumerate(layout_errors.items()):
    metrics = [data["error_metrics"][m] for m in metrics_for_plot]
    axes[0].bar(x + i * width, metrics, width, label=layout_type)

axes[0].set_ylabel("Error Value")
axes[0].set_title("Wind Turbine Error Metrics by Layout Type")
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(labels)
axes[0].legend()

# Create scatter plot of error distributions
for i, (layout_type, data) in enumerate(layout_errors.items()):
    axes[1].scatter(
        np.ones(len(data["error_metrics"]["wt_errors"])) * (i + 1),
        data["error_metrics"]["wt_errors"],
        alpha=0.5,
        label=layout_type,
    )

axes[1].set_ylabel("Error (m/s)")
axes[1].set_title("Distribution of Wind Turbine Errors")
axes[1].set_xticks(range(1, len(layout_errors) + 1))
axes[1].set_xticklabels(list(layout_errors.keys()))
axes[1].axhline(y=0, color="k", linestyle="-", alpha=0.3)

# Create scatter plot of actual vs predicted values
for _i, (layout_type, data) in enumerate(layout_errors.items()):
    axes[2].scatter(
        data["error_metrics"]["wt_targets"],
        data["error_metrics"]["wt_predictions"],
        alpha=0.5,
        label=f"{layout_type} (n={data['n_wts']})",
    )

# Add diagonal line
min_val = min(
    [
        min(
            data["error_metrics"]["wt_targets"].min(),
            data["error_metrics"]["wt_predictions"].min(),
        )
        for data in layout_errors.values()
    ]
)
max_val = max(
    [
        max(
            data["error_metrics"]["wt_targets"].max(),
            data["error_metrics"]["wt_predictions"].max(),
        )
        for data in layout_errors.values()
    ]
)

axes[2].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
axes[2].set_xlabel("Target Velocity (m/s)")
axes[2].set_ylabel("Predicted Velocity (m/s)")
axes[2].set_title("Wind Turbine Prediction Accuracy")
axes[2].legend()

# Create error histogram
all_rel_errors = []
for layout_type, data in layout_errors.items():
    rel_errors = (data["error_metrics"]["wt_errors"] / data["error_metrics"]["U_free"]) * 100
    all_rel_errors.extend(rel_errors)
    axes[3].hist(rel_errors, bins=20, alpha=0.5, label=layout_type)

axes[3].set_xlabel("Relative Error (% of U)")
axes[3].set_ylabel("Count")
axes[3].set_title("Distribution of Relative Errors")
axes[3].legend()

plt.tight_layout()
plt.savefig(
    os.path.join(fig_folder_path, f"wt_error_metrics_{model_type_str}.pdf"),
    dpi=300,
    bbox_inches="tight",
)

# %%
