import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from py_wake.examples.data.dtu10mw import DTU10MW
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from utils.data_tools import (
    retrieve_dataset_stats,
    setup_test_val_iterator,
    setup_unscaler,
)
from utils.plotting import matplotlib_set_rcparams, plot_probe_graph_fn
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

# Load paths and configs
cfg_path = os.path.join(main_path, ".hydra/config.yaml")
model_cfg_path = os.path.abspath(os.path.join(main_path, "model_config.json"))
params_paths = os.path.join(main_path, "best_params.msgpack")

if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"

fig_folder_path = os.path.join(main_path, "model/figures_wt_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)

# Load model config
with open(model_cfg_path, "r") as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)

# Set test data path
if not running_locally:
    test_data_path = restored_cfg_model.data.test_path  # Sophia server
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )

dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

# Find representative layouts like in the original script
layout_type_idxs = {
    "cluster": None,
    "single string": None,
    "multiple string": None,
    "parallel string": None,
}

cluster_yrange_offset_old = 1e9
single_string_yrange_offset_old = 1e9
multiple_string_yrange_offset_old = 1e9
parrallel_string_yrange_offset_old = 1e9
y_chosen = False
idx_start = 0

for idx, data in tqdm(enumerate(dataset)):
    layout_type = data.layout_type
    x_range = data.pos[:, 0].max() - data.pos[:, 0].min()
    y_range = data.pos[:, 1].max() - data.pos[:, 1].min()

    if (
        idx > idx_start
        and layout_type == "cluster"
        and not y_chosen
        and np.round(data.wt_spacing, 0) == 5.0
        and data.n_wt >= 50
    ):
        y_range_chosen = y_range
        y_chosen = True

    if y_chosen:
        if layout_type == "cluster":
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < cluster_yrange_offset_old:
                layout_type_idxs["cluster"] = idx
                cluster_yrange_offset_old = y_range_offset
        elif layout_type == "single string":
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < single_string_yrange_offset_old:
                layout_type_idxs["single string"] = idx
                single_string_yrange_offset_old = y_range_offset
        elif layout_type == "multiple string":
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < multiple_string_yrange_offset_old:
                layout_type_idxs["multiple string"] = idx
                multiple_string_yrange_offset_old = y_range_offset
        elif layout_type == "parallel string":
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < parrallel_string_yrange_offset_old:
                layout_type_idxs["parallel string"] = idx
                parrallel_string_yrange_offset_old = y_range_offset

# Calculate max distance for plotting
max_distance = 0
max_y_range = 0
for idx in layout_type_idxs.values():
    data = dataset[idx]
    x_range = data.pos[:, 0].max() - data.pos[:, 0].min()
    y_range = data.pos[:, 1].max() - data.pos[:, 1].min()
    distance = max(x_range, y_range)
    if distance > max_distance:
        max_distance = distance
    if y_range > max_y_range:
        max_y_range = y_range
plot_distance = max_distance * 1.5

# Setup iterator for test data
get_plot_data_iterator, test_dataset, _, _ = setup_test_val_iterator(
    restored_cfg_model,
    type_str="test",
    return_idxs=True,
    return_positions=True,
    path=test_data_path,
    cache=False,
    return_layout_info=True,
    dataset=dataset,
)

iterator = get_plot_data_iterator()

plot_graphs = deepcopy(layout_type_idxs)

for i, data_in in tqdm(enumerate(iterator)):
    if i in layout_type_idxs.values():
        # find which layout type
        for key, val in layout_type_idxs.items():
            if val == i:
                layout_type = key
        plot_graphs[layout_type] = data_in

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
def analyze_wt_errors(
    graphs, probe_graphs, targets, wt_mask, probe_mask, model_params, unscaler
):
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
    probe_idx = np.where(probe_mask != 0)[0]

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

# %%  Plot wind turbine error metrics for each layout type
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
for i, (layout_type, data) in enumerate(layout_errors.items()):
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
    rel_errors = (
        data["error_metrics"]["wt_errors"] / data["error_metrics"]["U_free"]
    ) * 100
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


# %% Pre-calc

# Define wind speeds as in plot_predictions_article.py
U_free_values = [6, 12, 18]  # m/s
min_error = 1e9
max_error = -1e9
agg_min_error = 1e9
agg_max_error = -1e9
# Create a data structure to store all predictions and errors
all_prediction_data = {}
aggregated_predictions = {}
for i, (layout_type, data) in enumerate(layout_errors.items()):
    # Get unscaled data for plotting
    unscaled_node_positions = unscaler.inverse_scale_trunk_input(data["node_positions"])

    # Find wind turbine indices and positions
    wt_idx = np.where(data["wt_mask"] != 0)[0]
    wt_positions = unscaler.inverse_scale_trunk_input(data["node_positions"])[wt_idx]

    # Get graph data needed for prediction
    graphs = data["graphs"]
    probe_graphs = data["probe_graphs"]
    wt_mask = data["wt_mask"]
    probe_mask = data["probe_mask"]

    # Original free stream velocity
    original_U_free = unscaler.inverse_scale_graph(graphs).globals[0][0]

    # Initialize data structure for this layout
    all_prediction_data[layout_type] = {"wt_positions": wt_positions, "wind_speeds": {}}

    # Run prediction with the original wind speed - we'll do this once and scale for each wind speed
    prediction = pred_fn(
        restored_params,
        graphs,
        probe_graphs,
        jnp.atleast_2d(wt_mask),
        jnp.atleast_2d(probe_mask),
    ).squeeze()

    wt_rel_errors_list = []
    # Process each wind speed
    for j, U_flow in enumerate(U_free_values):
        # Extract wind turbine predictions and scale them to the new wind speed
        wt_predictions = _inverse_scale_target(prediction[wt_idx]).squeeze()
        wt_predictions = wt_predictions * (U_flow / original_U_free)

        # Scale the targets for the new wind speed
        wt_targets = data["error_metrics"]["wt_targets"] * (U_flow / original_U_free)

        # Calculate errors
        wt_errors = wt_targets - wt_predictions
        wt_rel_errors = (wt_errors / U_flow) * 100  # Percent of inflow speed
        wt_rel_errors_list.append(wt_rel_errors)

        # Update min/max errors for colorbar scaling
        if wt_rel_errors.min() < min_error:
            min_error = wt_rel_errors.min()
        if wt_rel_errors.max() > max_error:
            max_error = wt_rel_errors.max()

        # Store all the data for this wind speed
        all_prediction_data[layout_type]["wind_speeds"][U_flow] = {
            "wt_predictions": wt_predictions,
            "wt_targets": wt_targets,
            "wt_errors": wt_errors,
            "wt_rel_errors": wt_rel_errors,
        }
    aggregated_predictions[layout_type] = {
        "mean": np.mean(wt_rel_errors_list, axis=0),
        "std": np.std(wt_rel_errors_list, axis=0),
        "abs_max": np.max(np.abs(wt_rel_errors_list), axis=0),
    }
    # Update aggregated min/max errors for colorbar scaling
    if agg_min_error > aggregated_predictions[layout_type]["mean"].min():
        agg_min_error = aggregated_predictions[layout_type]["mean"].min()
    if agg_max_error < aggregated_predictions[layout_type]["mean"].max():
        agg_max_error = aggregated_predictions[layout_type]["mean"].max()


# create a dict of extremas for different aggregated metrics
aggregated_extremas = {
    "mean": (agg_min_error, agg_max_error),
    "std": (
        min([aggregated_predictions[lt]["std"].min() for lt in aggregated_predictions]),
        max([aggregated_predictions[lt]["std"].max() for lt in aggregated_predictions]),
    ),
    "abs_max": (
        min(
            [
                aggregated_predictions[lt]["abs_max"].min()
                for lt in aggregated_predictions
            ]
        ),
        max(
            [
                aggregated_predictions[lt]["abs_max"].max()
                for lt in aggregated_predictions
            ]
        ),
    ),
}

# %% Create a single figure with 2x2 grid for all layout error plots
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()
metric_to_plot = "abs_max"
# metric_to_plot = "mean"

# Create error plots for each layout type
for i, (layout_type, data) in enumerate(layout_errors.items()):
    ax = axes[i]

    # Get the pre-computed positions
    wt_positions = all_prediction_data[layout_type]["wt_positions"]

    wt_rel_errors_agg = aggregated_predictions[layout_type]
    wt_rel_errors_agg_metric = wt_rel_errors_agg[metric_to_plot]

    extrema = np.max(np.abs(aggregated_extremas[metric_to_plot]))
    if metric_to_plot == "abs_max":
        vmin = 0
        vmax = extrema
        colormap = "gist_heat_r"
        cbar_legend = (
            r"$\max| (\boldsymbol{u}-\hat{\boldsymbol{u}})/\boldsymbol{U} |$ [$\%$]"
        )
    else:
        vmin = -extrema
        vmax = extrema
        colormap = "seismic"
        cbar_legend = (
            r"$\overline{u_\mathrm{err}/U}$"
            if metric_to_plot == "mean"
            else r"$\sigma(u_\mathrm{err}/U)$"
        )

    sc = ax.scatter(
        wt_positions[:, 0] / D,  # X positions normalized by diameter
        wt_positions[:, 1] / D,  # Y positions normalized by diameter
        c=wt_rel_errors_agg_metric,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        s=20,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.5,  # Increased line width for more pronounced border
        marker="o",
        # marker=(3, 2, 0),  # Custom plus marker: (num_points, style, angle)
        # Style 1 is a plus with wider horizontal lines
    )
    # add a scatter with just position as circles
    # ax.scatter(
    #     wt_positions[:, 0] / D,  # X positions normalized by diameter
    #     wt_positions[:, 1] / D,  # Y positions normalized by diameter
    #     facecolors="none",
    #     edgecolors="k",
    #     s=3,
    #     linewidths=0.5,
    #     alpha=1,
    #     marker="o",
    # )

    ax.set_xlabel("$x/D$ [-]")
    ax.set_ylabel("$y/D$ [-]")
    # ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal")

    if layout_type == "cluster":
        # pos = (0.05, 0.95)
        ax.set_xlim(-50, 50)
        ypos = 1.1
    elif layout_type == "single string":
        # pos = (0.7, 0.95)
        ax.set_xlim(-120, 150)
        ypos = 1.15
    elif layout_type == "multiple string":
        # pos = (0.05, 0.95)
        ax.set_xlim(-70, 60)
        ypos = 1.1
    elif layout_type == "parallel string":
        # pos = (0.05, 0.95)
        ax.set_xlim(-40, 40)
        ypos = 1.1
        # roatre labels for better fit
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    # add (a), (b), etc. to each subplot
    ax.text(
        0.0,
        ypos,
        f"({chr(97 + i)}) {layout_type}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="white", alpha=1),
    )

# Add a single colorbar for all subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(cbar_legend)


# Add a legend for the marker style
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker=(3, 2, 0),
        color="w",
        label="Wind Turbine",
        markerfacecolor="gray",
        markeredgecolor="k",
        markersize=10,
        markeredgewidth=1.5,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Turbine Position",
        markerfacecolor="none",
        markeredgecolor="k",
        markersize=6,
        markeredgewidth=0.5,
    ),
]


# Adjust layout
plt.tight_layout(
    rect=[0, 0.06, 0.9, 1]
)  # Adjust to leave space for colorbar and legend
plt.savefig(
    os.path.join(
        fig_folder_path, f"wt_spatial_errors_all_layouts_{model_type_str}.pdf"
    ),
    dpi=300,
    bbox_inches="tight",
)

# %%
