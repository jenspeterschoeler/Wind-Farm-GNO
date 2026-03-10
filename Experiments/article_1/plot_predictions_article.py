# %%
"""
Exploratory plotting script for Article 1.

This script contains exploratory visualizations and analysis plots that require
the full dataset and model. Publication figures have been moved to
plot_publication_figures.py which uses cached data.

Exploratory plots included:
- Single-layout cross-stream profiles (presentation style)
- Contour plots of flow field predictions
- Static and animated error visualizations
- Zoom-in velocity comparison plots

For publication figures, see:
    - plot_publication_figures.py (generates from cached data)
    - generate_plot_data.py (generates cache files on Sophia)
"""

import json
import os
import sys
from pathlib import Path

import jax
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jraph._src import utils as jraph_utils
from omegaconf import DictConfig
from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

# Import shared functions from master script
from plot_publication_figures import (
    apply_mask,
    apply_normalizations,
    get_max_plot_distance,
    get_model_paths,
    select_representative_layouts,
    setup_plot_iterator,
)

from utils.data_tools import (
    retrieve_dataset_stats,
    setup_unscaler,
)
from utils.plotting import (
    matplotlib_set_rcparams,
    plot_crossstream_predictions,
    plot_probe_graph_fn,
)
from utils.run_pywake import construct_on_the_fly_probe_graph
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

matplotlib_set_rcparams("presentation")

# %% Paths and configurations
# main_path = os.path.abspath(
#     "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
# )  # sophia
main_path = "./assets/best_model_Vj8"  # local, run from repo root
# main_path = "../../assets/best_model_Vj8"  # local, run from this folder (for ipython)


if "best_model_Vj8" in main_path:
    running_locally = True
    test_data_path = "./data/zenodo_graphs/test_pre_processed"

    # load dummy input
    import importlib.util

    dummy_file = os.path.join(main_path, "dummy_input.py")
    spec = importlib.util.spec_from_file_location("dummy_input", dummy_file)
    dummy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dummy_module)
    graphs = dummy_module.graphs
    probe_graphs = dummy_module.probe_graphs
    node_array_tuple = dummy_module.node_array_tuple
else:
    running_locally = False

paths = get_model_paths(main_path)
cfg_path = paths["cfg_path"]
model_cfg_path = paths["model_cfg_path"]
params_paths = paths["params_path"]

if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"
fig_folder_path = os.path.join(main_path, "model/figures_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)

### Load
with open(model_cfg_path) as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)


if running_locally:
    test_data_path = os.path.abspath(
        "./data/zenodo_graphs/test_pre_processed"
    )  # Download dataset and place in data directory: https://doi.org/10.5281/zenodo.17671257
else:
    test_data_path = restored_cfg_model.data.test_path  # Sophia server
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )

dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

if running_locally:
    restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
        params_paths,
        model_cfg_path,
        dataset=None,
        inputs=(graphs, probe_graphs, node_array_tuple),
    )
else:
    restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
        params_paths, model_cfg_path, dataset
    )


# %% Select graphs for each layout type using shared function
layout_type_idxs = select_representative_layouts(dataset)
plot_distance, max_y_range = get_max_plot_distance(dataset, layout_type_idxs)

# Setup iterator using shared function
plot_graphs, test_dataset = setup_plot_iterator(
    restored_cfg_model, test_data_path, dataset, layout_type_idxs
)

stats, scale_stats = retrieve_dataset_stats(dataset)
unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
_inverse_scale_target = unscaler.inverse_scale_output

wt = DTU10MW()
D = wt.diameter()
grid_density = 3  # grid density is created to match original dataset
unscaled_rel_plot_distance = plot_distance * scale_stats["distance"]["range"] / D


pred_fn = jax.jit(model.apply)

# %% Pre-calculations

x_downstream = [25, 50, 100]
U_free = [6, 12, 18]

TI_flow = 0.05
y_plot_range = plot_distance * 1.0 * scale_stats["distance"]["range"]
y = np.linspace(-y_plot_range / 2, y_plot_range / 2, int(y_plot_range * grid_density))

# Get the colormap
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = colors[:4]

plot_graphs_predictions_list = []


for (key, val), test_idx in tqdm(zip(plot_graphs.items(), layout_type_idxs.values())):
    graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
    targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

    unpadded_graph = jraph_utils.unpad_with_graphs(graphs)
    unpadded_probe_graph = jraph_utils.unpad_with_graphs(probe_graphs)

    unscaled_graphs = unscaler.inverse_scale_graph(graphs)
    unscaled_probe_graphs = unscaler.inverse_scale_graph(probe_graphs)

    xmin = node_positions[:, 0].min() * scale_stats["distance"]["range"] / D
    xmax = node_positions[:, 0].max() * scale_stats["distance"]["range"] / D
    x_range = [xmin - 10, xmax + 100]  # ranges are created to match original dataset
    x_steps = int((x_range[1] - x_range[0]) * grid_density)
    x = np.linspace(x_range[0], x_range[1], x_steps) * D

    # generate graphs for each x position
    wt_positions = test_dataset[test_idx].pos.numpy() * scale_stats["distance"]["range"]
    predictions_dist = []

    graph_proccessed = False
    for x_sel in x_downstream:
        predictions_flow = []

        for U_flow in U_free:
            grid = HorizontalGrid(
                x=[x_sel * D + wt_positions[:, 0].max()],
                y=y,
                h=wt.hub_height(),
            )
            xx, yy = np.meshgrid(grid.x, grid.y)

            jraph_graph_gen, jraph_probe_graphs_gen, node_array_tuple_gen = (
                construct_on_the_fly_probe_graph(
                    positions=wt_positions,
                    U=[U_flow],
                    TI=[TI_flow],
                    grid=grid,  # Assuming grid is not used in this context
                    scale_stats=scale_stats,
                    return_positions=True,
                )
            )
            (
                targets_gen,
                wt_mask_gen,
                probe_mask_gen,
                node_positions_gen,
            ) = node_array_tuple_gen

            prediction = pred_fn(
                restored_params,
                jraph_graph_gen,
                jraph_probe_graphs_gen,
                jnp.atleast_2d(wt_mask_gen).T,
                jnp.atleast_2d(probe_mask_gen).T,
            ).squeeze()

            unscaled_probe_predictions = _inverse_scale_target(
                apply_mask(prediction, probe_mask_gen)
            )
            unscaled_probe_targets = _inverse_scale_target(
                apply_mask(targets_gen.squeeze(), probe_mask_gen)
            )

            if not graph_proccessed:
                unscaled_node_positions = unscaler.inverse_scale_trunk_input(node_positions_gen) / D
                unscaled_graphs_gen = unscaler.inverse_scale_graph(jraph_graph_gen)
                unscaled_probe_graphs_gen = unscaler.inverse_scale_graph(jraph_probe_graphs_gen)
                unscaled_graphs_gen = unscaled_graphs_gen._replace(
                    edges=unscaled_graphs_gen.edges / D
                )
                unscaled_probe_graphs_gen = unscaled_probe_graphs_gen._replace(
                    edges=unscaled_probe_graphs_gen.edges / D
                )
                graph_proccessed = True

            pred_dict = {
                "layout": key,
                "wt_spacing": wt_spacing,
                "x_downstream": x_sel,
                "U_free": U_flow,
                "unscaled_predictions": unscaled_probe_predictions,
                "unscaled_targets": unscaled_probe_targets,
                "rel_error": (unscaled_probe_predictions - unscaled_probe_targets) / U_flow * 100,
                "y/D": y / wt.diameter(),
                "unscaled_node_positions": unscaled_node_positions,
                "unscaled_graphs": unscaled_graphs_gen,
                "unscaled_probe_graphs": unscaled_probe_graphs_gen,
            }
            predictions_flow.append(pred_dict)
        predictions_dist.append(predictions_flow)
    plot_graphs_predictions_list.append(predictions_dist)

# %% settings and normalization terms
plot_velocity_deficit = True
plot_error = False
normalize_by_U = True
assert (
    plot_velocity_deficit + plot_error <= 1
), "Only one of the two plot options can be true at the same time"


# pre-run to find the velocity range for the plots
for plot_graph_predictions in plot_graphs_predictions_list:
    for dist_predictions in plot_graph_predictions:
        for U_flow, _color, predictions_U_dict in zip(U_free, colors, dist_predictions):
            unscaled_probe_predictions = predictions_U_dict["unscaled_predictions"]
            unscaled_probe_targets = predictions_U_dict["unscaled_targets"]
            y = predictions_U_dict["y/D"]

            unscaled_probe_predictions, unscaled_probe_targets, _ = apply_normalizations(
                np.array(unscaled_probe_predictions),
                np.array(unscaled_probe_targets),
                U_flow,
                plot_velocity_deficit=plot_velocity_deficit,
                normalize_by_U=normalize_by_U,
            )

            current_min = min(unscaled_probe_predictions.min(), unscaled_probe_targets.min())
            current_max = max(unscaled_probe_predictions.max(), unscaled_probe_targets.max())
            if "global_min" not in locals():
                global_min = current_min
                global_max = current_max
            else:
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max

pad_size = (global_max - global_min) * 0.1  # 10% padding
velocity_range = [global_min - pad_size, global_max + pad_size]


# %% separate plots for presentation in slides (EXPLORATORY)
matplotlib_set_rcparams("presentation")


for plot_graph_predictions, (key, val), test_idx in zip(
    plot_graphs_predictions_list, plot_graphs.items(), layout_type_idxs.values()
):
    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(1, len(x_downstream) + 1, wspace=0.3, hspace=0.3)

    graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
    targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple
    print(key)
    print(f"layout type: {layout_type[0]}")
    print(f"wt spacing: {wt_spacing} D")
    print(f"num wts: {jnp.sum(wt_mask)}")
    print(f"num probes: {jnp.sum(probe_mask)}")

    predictions_U_dict = plot_graph_predictions[0][0]

    unscaled_node_positions = predictions_U_dict["unscaled_node_positions"]
    unscaled_graphs_gen = predictions_U_dict["unscaled_graphs"]
    unscaled_probe_graphs_gen = predictions_U_dict["unscaled_probe_graphs"]

    ax = fig.add_subplot(gs[0, 0])
    plot_probe_graph_fn(
        unscaled_graphs_gen,
        unscaled_probe_graphs_gen,
        unscaled_node_positions,
        include_probe_edges=False,
        include_probe_nodes=False,
        ax=ax,
    )

    ax.set_xlabel(r"$x/D$ [-]")
    ax.set_ylabel(r"$y/D$ [-]")
    ax.set_xlim(-unscaled_rel_plot_distance / 2, unscaled_rel_plot_distance / 2)
    ax.set_ylim(-unscaled_rel_plot_distance / 2, unscaled_rel_plot_distance / 2)
    ax.tick_params(axis="x", rotation=30)

    ax.legend_.remove()

    # add a text box outside graph with ascending letters
    letter = chr(97 + list(layout_type_idxs.keys()).index(key))
    row_text = (
        r"$n_\mathrm{wt}=$"
        + f"$ {int(np.sum(wt_mask))}$, "
        + r"$s_\mathrm{wt}$="
        + f"$ {wt_spacing.numpy()[0].round(2)}$"
        r"$D$"
    )
    ax.text(-150, 125, row_text)

    for ds_row, (x_sel, dist_predictions) in enumerate(zip(x_downstream, plot_graph_predictions)):
        ax = fig.add_subplot(gs[0, ds_row + 1])

        for U_flow, color, predictions_U_dict in zip(U_free, colors, dist_predictions):
            unscaled_probe_predictions = predictions_U_dict["unscaled_predictions"]
            unscaled_probe_targets = predictions_U_dict["unscaled_targets"]
            y = predictions_U_dict["y/D"]

            unscaled_probe_predictions, unscaled_probe_targets, label = apply_normalizations(
                np.array(unscaled_probe_predictions),
                np.array(unscaled_probe_targets),
                U_flow,
                plot_velocity_deficit=plot_velocity_deficit,
                normalize_by_U=normalize_by_U,
            )

            ax.plot(
                unscaled_probe_predictions,
                y,
                ls="-",
                color=color,
                linewidth=1.5,
                alpha=0.75,
            )
            if not plot_error:
                ax.plot(
                    unscaled_probe_targets,
                    y,
                    ls="--",
                    linewidth=1.5,
                    dashes=(2, 2),
                    color=color,
                    alpha=0.75,
                )

        ax.set_xlim(velocity_range)
        ax.set_ylim(-y_plot_range / (2 * D), y_plot_range / (2 * D))
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel(f"{label}")

        ax.text(
            -0.095,
            135,
            f"$\\widebar{{x}}={x_sel}D$",
        )

    # Local legend for colors and line styles
    color_legend_labels = [f"{U} m/s" for U in U_free]
    lines = [
        plt.Line2D([0], [0], color="black", ls="-", alpha=0.75, lw=2),
        plt.Line2D([0], [0], color="black", ls="--", alpha=0.75, lw=2),
    ]
    legend_labels = ["Predictions", "Targets"]
    for c, lab in zip(colors, color_legend_labels):
        lines.append(plt.Line2D([0], [0], color=c, ls="-", alpha=0.75, lw=2))
        legend_labels.append(lab)

    fig.legend(
        lines,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.85, 1.1),
        fontsize=16,
        frameon=True,
        ncol=5,
    )

    # plt.suptitle(f"{key} layout, $s_{{wt}}={wt_spacing.numpy()[0].round(2)}D$")
    plt.savefig(
        os.path.join(
            fig_folder_path,
            f"crossstream_profiles_{model_type_str}_case{test_idx}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )


# %% EXPLORATORY PLOTS - Graph visualization and basic predictions

for (key, val), _test_idx in zip(plot_graphs.items(), layout_type_idxs.values()):
    graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
    targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple
    print(key)
    print(f"layout type: {layout_type[0]}")
    print(f"wt spacing: {wt_spacing} D")
    print(f"num wts: {jnp.sum(wt_mask)}")
    print(f"num probes: {jnp.sum(probe_mask)}")
    unpadded_graph = jraph_utils.unpad_with_graphs(graphs)
    unpadded_probe_graph = jraph_utils.unpad_with_graphs(probe_graphs)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_probe_graph_fn(
        unpadded_graph,
        unpadded_probe_graph,
        node_positions,
        include_probe_edges=False,
        include_probe_nodes=False,
        ax=ax,
    )
    # plt.suptitle(f"{key} layout")
    plt.xlim(-plot_distance / 2, plot_distance / 2)
    plt.ylim(-plot_distance / 2, plot_distance / 2)
    # plt.axis("equal")
    plt.show()


prediction_restored_params = model.apply(
    restored_params,
    graphs,
    probe_graphs,
    wt_mask,
    probe_mask,
)


trunk_idxs = (
    trunk_idxs[: np.sum(probe_mask, dtype=int)].astype(int).squeeze()
)  # removes padding, is a little complex due to the sizes shifting because of the combination of probes and wts
trunk_xy = dataset[0]["trunk_inputs"].numpy()
trunk_local_xy = trunk_xy[trunk_idxs]
unscaled_trunk_xy = unscaler.inverse_scale_trunk_input(trunk_local_xy)

probe_idx = np.where(probe_mask != 0)[0]
wt_idx = np.where(wt_mask != 0)[0]

probe_predictions = _inverse_scale_target(prediction_restored_params[probe_idx]).squeeze()
probe_targets = _inverse_scale_target(targets[probe_idx]).squeeze()
probe_errors = probe_targets - probe_predictions

wt_predictions = _inverse_scale_target(prediction_restored_params[wt_idx]).squeeze()
wt_targets = _inverse_scale_target(targets[wt_idx]).squeeze()
wt_erros = wt_targets - wt_predictions

persistence_pred = _inverse_scale_target(
    probe_graphs.nodes[:, 0]
).squeeze()  # only works when the only target is U
persitence_probe_pred = persistence_pred[probe_idx].squeeze()
persitence_wt_pred = persistence_pred[wt_idx].squeeze()

persistence_probe_error = persitence_probe_pred - probe_targets
persistence_wt_error = persitence_wt_pred - wt_targets


unscaled_graphs = unscaler.inverse_scale_graph(graphs)
unscaled_probe_graphs = unscaler.inverse_scale_graph(probe_graphs)


unscaled_node_positions = unscaler.inverse_scale_trunk_input(node_positions)

unscaled_trunk_xy = unscaler.inverse_scale_trunk_input(trunk_local_xy)

trunk_local_x = unscaled_trunk_xy[:, 0]
trunk_local_y = unscaled_trunk_xy[:, 1]


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_probe_graph_fn(
    unscaled_graphs,
    unscaled_probe_graphs,
    unscaled_node_positions,
    include_probe_edges=False,
    ax=axes[0],
)
plot_crossstream_predictions(probe_predictions, probe_targets, trunk_local_y, ax=axes[1])
# plt.suptitle(
#     f"Probe predictions {model_type_str}, flowcase [U and TI]{np.round(unscaled_graphs.globals[0,:].squeeze(),2)}",
# )
plt.savefig(
    os.path.join(fig_folder_path, f"probe_predictions_{model_type_str}_case{test_idx}.png"),
    dpi=300,
)
# %%

# xmin = unscaled_graphs.edges[:, 1].min() /wt.diameter()
xmin = node_positions[:, 0].min() * scale_stats["distance"]["range"] / D
xmax = node_positions[:, 0].max() * scale_stats["distance"]["range"] / D
x_range = [xmin - 10, xmax + 100]  # ranges are created to match original dataset
x_steps = int((x_range[1] - x_range[0]) * grid_density)
x = np.linspace(x_range[0], x_range[1], x_steps) * D
ymax = np.abs(unscaled_graphs.edges[:, 2]).max() / D
y_range = [-ymax - 5, ymax + 5]
y_steps = int(
    (y_range[1] - y_range[0]) * grid_density
)  # ranges are created to match original dataset

y = np.linspace(-ymax - 5, ymax + 5, y_steps) * wt.diameter()

# generate graphs for each x position
wt_positions = test_dataset[test_idx].pos.numpy() * scale_stats["distance"]["range"]


predictions = []
targets = []
xx_list = []
yy_list = []

for i, x_sel in enumerate(x):
    grid = HorizontalGrid(
        x=[x_sel],
        y=y,
        h=wt.hub_height(),
    )
    xx, yy = np.meshgrid(grid.x, grid.y)

    jraph_graph_gen, jraph_probe_graphs_gen, node_array_tuple_gen = (
        construct_on_the_fly_probe_graph(
            positions=wt_positions,
            U=[U_flow],
            TI=[TI_flow],
            grid=grid,  # Assuming grid is not used in this context
            scale_stats=scale_stats,
            return_positions=True,
        )
    )
    (
        targets_gen,
        wt_mask_gen,
        probe_mask_gen,
        node_positions_gen,
    ) = node_array_tuple_gen

    # gen_graphs = jraph_graph_gen
    # gen_probe_graphs = jraph_probe_graphs_gen
    gen_targets = targets_gen
    gen_wt_mask = wt_mask_gen
    gen_probe_mask = probe_mask_gen
    # Add empty dimension
    gen_wt_mask = jnp.atleast_2d(gen_wt_mask).T  # Ensure wt_mask is 2D
    gen_probe_mask = jnp.atleast_2d(gen_probe_mask).T  # Ensure probe_mask is 2D
    gen_node_positions = node_positions_gen
    gen_node_array_tuple = (gen_targets, gen_wt_mask, gen_probe_mask)

    if i == 0:
        restored_params_gen, _, model_gen, dropout_active_gen = load_portable_model(
            params_paths,
            model_cfg_path,
            dataset=None,
            inputs=(jraph_graph_gen, jraph_probe_graphs_gen, gen_node_array_tuple),
        )

    predictions_gen = model_gen.apply(
        restored_params_gen,
        jraph_graph_gen,
        jraph_probe_graphs_gen,
        gen_wt_mask,
        gen_probe_mask,
        train=False,
    ).squeeze()

    gen_targets = gen_targets.squeeze()
    gen_wt_mask = gen_wt_mask.squeeze()
    gen_probe_mask = gen_probe_mask.squeeze()

    probe_idx = np.where(gen_probe_mask != 0)[0]
    probe_predictions_gen = jnp.where(gen_probe_mask != 0, predictions_gen, jnp.nan)
    probe_predictions_gen = probe_predictions_gen[~jnp.isnan(probe_predictions_gen)]
    probe_predictions_gen = probe_predictions_gen.reshape(xx.shape)
    probe_predictions_gen = probe_predictions_gen * np.array(scale_stats["velocity"]["range"])
    probe_targets_gen = jnp.where(gen_probe_mask != 0, gen_targets, jnp.nan)
    probe_targets_gen = probe_targets_gen[~jnp.isnan(probe_targets_gen)]
    probe_targets_gen = probe_targets_gen.reshape(xx.shape)
    probe_targets_gen = probe_targets_gen * np.array(scale_stats["velocity"]["range"])

    predictions.append(probe_predictions_gen)
    targets.append(probe_targets_gen)
    xx_list.append(xx)
    yy_list.append(yy)

predictions = np.concatenate(predictions, axis=1)
targets = np.concatenate(targets, axis=1)
xx = np.concatenate(xx_list, axis=1)
yy = np.concatenate(yy_list, axis=1)


# %%

grid_full = HorizontalGrid(
    x=x,
    y=y,
    h=wt.hub_height(),
)

jraph_graph_gen_full, jraph_probe_graphs_gen_full, node_array_tuple_gen_full = (
    construct_on_the_fly_probe_graph(
        positions=wt_positions,
        U=[U_flow],
        TI=[TI_flow],
        grid=grid_full,  # Assuming grid is not used in this context
        scale_stats=scale_stats,
        return_positions=True,
    )
)

(
    targets_gen_full,
    wt_mask_gen_full,
    probe_mask_gen_full,
    node_positions_gen_full,
) = node_array_tuple_gen_full


plt.fig = plt.figure(figsize=(3, 3))
ax = plt.gca()
plot_probe_graph_fn(
    jraph_graph_gen_full,
    jraph_probe_graphs_gen_full,
    node_positions_gen_full,
    include_probe_nodes=False,
    include_probe_edges=False,
    ax=ax,
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend(
    loc="lower right",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


min_cbar = min(
    probe_predictions_gen.min(),
    probe_targets_gen.min(),
)

max_cbar = max(
    np.max(probe_predictions_gen),
    np.max(probe_targets_gen),
)

levels = np.round(np.linspace(min_cbar, max_cbar, 500), 2)

xxD = xx / wt.diameter()
yyD = yy / wt.diameter()


x_start_idx = 650

plt.figure(figsize=(5, 5))
ax = plt.gca()
plt.contourf(
    xxD[:, x_start_idx:], yyD[:, x_start_idx:], targets[:, x_start_idx:], cmap="viridis"
)  # , levels=levels)
plt.colorbar(label="Probe Targets")
# ax.set_xlim(min(x) / wt.diameter(), 80)  # Normalize x limits by diameter
# ax.set_ylim(-70, 90)
plt.axis("equal")
plt.show()


plt.figure(figsize=(5, 5))
ax = plt.gca()
plt.contourf(
    xxD[:, x_start_idx:],
    yyD[:, x_start_idx:],
    predictions[:, x_start_idx:],
    cmap="viridis",
)  #     levels=levels)
plt.colorbar(label="Probe Predictions")
# ax.set_xlim(min(x) / wt.diameter(), 80)  # Normalize x limits by diameter
# ax.set_ylim(-70, 90)
plt.axis("equal")
plt.show()


plt.figure(figsize=(10, 5))
ax = plt.gca()
plt.contourf(
    xxD[:, x_start_idx:],
    yyD[:, x_start_idx:],
    (np.abs(targets - predictions) / U_flow * 100)[:, x_start_idx:],
    cmap="viridis",
    levels=30,
)
plt.axis("equal")
plt.colorbar(label="Error [% of U]")
plt.show()

# %%
### Version with an added axes zoom-in box to the right with a range of 1 m/s

for x_sel in np.linspace(0, predictions.shape[1] - 1, 10):
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])  # 3:1 ratio
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_err_zoom = fig.add_subplot(gs[2])

    ax_main.plot(predictions[:, int(x_sel)], y / wt.diameter(), label="pred", alpha=0.5)
    ax_main.plot(
        targets[:, int(x_sel)],
        y / wt.diameter(),
        label="targ",
        alpha=0.5,
        linestyle="--",
    )
    ax_main.set_xlabel("Velocity [m/s]")
    ax_main.set_ylabel("y [D]")
    ax_main.set_title(
        f"Cross-stream profiles at x = {np.round(np.unique(xx[:, int(x_sel)]).squeeze()/wt.diameter(),1)} D"
    )
    ax_main.legend(loc="upper left")
    ax_main.set_xlim(-1, U_flow * 1.1)

    # Zoom-in box
    zoom_center = U_flow  # Center of the zoom-in box
    zoom_range = 1.5  # Range of the zoom-in box (±0.5 m/s)
    ax_zoom.set_xlim(zoom_center - zoom_range / 2, zoom_center + zoom_range / 2)
    # ax_zoom.set_ylim(-5, 5)  # Adjust y-limits for better visibility
    ax_zoom.plot(predictions[:, int(x_sel)], y / wt.diameter(), label="pred", alpha=0.5)
    ax_zoom.plot(
        targets[:, int(x_sel)],
        y / wt.diameter(),
        label="targ",
        alpha=0.75,
        linestyle="--",
    )
    ax_zoom.set_title("Zoom-in")
    ax_zoom.set_xlabel("Velocity [m/s]")
    ax_zoom.set_ylabel("y [D]")
    # Add grid for better readability
    ax_zoom.grid(True)
    ax_zoom.tick_params(axis="x", rotation=65)

    # Error plot

    ax_err_zoom.plot(
        (predictions[:, int(x_sel)] - targets[:, int(x_sel)]) / U_flow * 100,
        y / wt.diameter(),
        label="error",
        color="orange",
        alpha=0.75,
    )
    # ax_err_zoom.set_xlim(-5, 5)
    # rotate x ticks 30 deg
    ax_err_zoom.tick_params(axis="x", rotation=65)
    ax_err_zoom.set_title("Error [% of U]")
    ax_err_zoom.set_xlabel("Error [% of U]")
    ax_err_zoom.set_ylabel("y [D]")
    ax_err_zoom.grid(True)

    plt.tight_layout()
    plt.show()

# %%


unscaled_gen_graphs = unscaler.inverse_scale_graph(jraph_graph_gen)
unscaled_gen_graphs = unscaled_gen_graphs._replace(edges=unscaled_gen_graphs.edges / wt.diameter())
unscaled_gen_probe_graphs = unscaler.inverse_scale_graph(jraph_probe_graphs_gen)
unscaled_gen_probe_graphs = unscaled_gen_probe_graphs._replace(
    edges=unscaled_gen_probe_graphs.edges / wt.diameter()
)
unscaled_gen_node_positions = unscaler.inverse_scale_trunk_input(node_positions_gen) / wt.diameter()

rel_pred_error_U = ((targets - predictions) / U_flow) * 100
rel_pred_error = ((probe_targets_gen - probe_predictions_gen) / probe_targets_gen) * 100
rel_pred_abs_mean_crossstream = np.mean(np.abs(rel_pred_error), axis=0)
rel_pred_abs_max_crossstream_U = np.max(np.abs(rel_pred_error_U), axis=0)
rel_pred_abs_mean_crossstream_U = np.mean(np.abs(rel_pred_error_U), axis=0)


rel_x = x / wt.diameter()  # Normalize x position by diameter
rel_y = y / wt.diameter()  # Normalize y position by diameter

# %% Static Plotting

case = 1

if case == 1:
    height_ratios = [3, 2, 1]  # Row 1 is 3x height of Row 2
    figsize = (8, 12)
    ax1_legend_ncols = 1
    ax2_xlim_additions = [-2.5, 0.25]
    ax3_xlim = [-10, 10]

elif case == 2:
    height_ratios = [1, 2, 1]
    figsize = (8, 9)
    ax1_legend_ncols = 3
    ax2_xlim_additions = [-5.5, 0.5]
    ax3_xlim = [-2.5, 2.5]


y_idx = 122


def static_plot(y_idx):
    probe_pred_sel = probe_predictions_gen[:, y_idx]
    probe_target_sel = probe_targets_gen[:, y_idx]

    x_pos = x[y_idx] / wt.diameter()  # Normalize x position by diameter
    # Create figure
    fig = plt.figure(figsize=figsize)

    # Set up GridSpec: 2 rows, 3 columns
    gs = gridspec.GridSpec(3, 2, height_ratios=height_ratios)  # Row 1 is 2x height of Row 2

    # First row: three square plots
    ax1 = fig.add_subplot(gs[0, :])

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax4 = fig.add_subplot(gs[2, :])
    # ax1.set_aspect("equal")  # Make the plot square
    # ax2.set_aspect("equal")  # Make the plot square
    # ax3.set_aspect("equal")  # Make the plot square

    plot_probe_graph_fn(
        unscaled_gen_graphs,
        unscaled_gen_probe_graphs,
        unscaled_gen_node_positions,
        include_probe_edges=False,
        include_probe_nodes=False,
        ax=ax1,
    )
    ax1.vlines(
        x_pos,
        min(rel_y),
        max(rel_y),
        color="red",
        linestyle="-",
        label="Probe positions",
    )
    ax1.legend(loc="upper left", ncols=ax1_legend_ncols)
    ax1.set_xlim(
        min(x) / wt.diameter() - 20, max(x) / wt.diameter() + 10
    )  # Normalize x limits by diameter

    ax1.set_xlabel(r"$x [D]$")
    ax1.set_ylabel(r"$y [D]$")

    plot_crossstream_predictions(probe_pred_sel, probe_target_sel, rel_y, marker=False, ax=ax2)
    # ax2.set_xlim()
    # ax2.set_xlim(max(probe_target_sel)-1, max(probe_target_sel))

    ax2.set_xlabel(r"$u$  [m/s]")
    ax2.set_ylabel(r"$y [D]$")
    # reorient x ticks 30 deg
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(loc="upper left")
    ax2.set_xlim(
        max(probe_targets_gen.flatten()) + ax2_xlim_additions[0],
        max(probe_targets_gen.flatten()) + ax2_xlim_additions[1],
    )

    ax3.plot(
        ((probe_pred_sel - probe_target_sel) / U_flow) * 100,
        rel_y,
        label="Probe  Error (%)",
        color="orange",
    )
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax3_xlim)
    ax3.set_yticks([])
    ax3.set_xlabel(r"$\dfrac{u - \hat{u}}{U}$")

    # Second row: one plot spanning all columns
    # ax4 = fig.add_subplot(gs[1, :])
    # ax4.plot(
    #     rel_x[: y_idx + 1],
    #     rel_pred_abs_mean_crossstream[: y_idx + 1],
    #     label=r"$\dfrac{1}{N} \sum_{i=1}^N \left(|\dfrac{u - \hat{u}}{u}|\right)$",
    #     color="orange",
    # )
    # ax4.scatter(
    #     rel_x[y_idx],
    #     rel_pred_abs_mean_crossstream[y_idx],
    #     color="orange",
    #     s=10,
    # )
    ax4.plot(
        rel_x[: y_idx + 1],
        rel_pred_abs_mean_crossstream_U[: y_idx + 1],
        label=r"$\dfrac{1}{N} \sum_{i=1}^N \left(|\dfrac{u - \hat{u}}{U}|\right)$",
        color="orange",
    )
    ax4.scatter(
        rel_x[y_idx],
        rel_pred_abs_mean_crossstream_U[y_idx],
        color="orange",
        s=10,
    )

    ax4.set_xlim(min(rel_x) - 2, max(rel_x) + 2)  # Normalize x limits by diameter
    ax4.set_xlabel(r"$x [D]$")
    ax4.set_ylabel("Mean error [%]")
    ax4.legend(ncols=2, loc="upper right")
    ax4.set_ylim(min(rel_pred_abs_mean_crossstream) - 1, max(rel_pred_abs_mean_crossstream) + 1)

    plt.tight_layout()
    return fig, ax1, ax2, ax3, ax4


fig, ax1, ax2, ax3, ax4 = static_plot(y_idx)  # Call the function to plot for the initial y_idx
plt.savefig(
    os.path.join(fig_folder_path, f"probe_predictions_static_{model_type_str}_{test_idx}.png"),
    dpi=300,
)
# %% Animation

# Set up the figure and axes once
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 2, height_ratios=height_ratios)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, :])


def animate(y_idx):
    # Clear axes, not the whole figure
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    # Use the same plotting logic as static_plot, but pass in the axes
    probe_pred_sel = probe_predictions_gen[:, y_idx]
    probe_target_sel = probe_targets_gen[:, y_idx]
    x_pos = x[y_idx] / wt.diameter()
    plot_probe_graph_fn(
        unscaled_gen_graphs,
        unscaled_gen_probe_graphs,
        unscaled_gen_node_positions,
        include_probe_edges=False,
        include_probe_nodes=False,
        ax=ax1,
    )
    ax1.vlines(
        x_pos,
        min(rel_y),
        max(rel_y),
        color="red",
        linestyle="-",
        label="Probe positions",
    )
    ax1.legend(loc="upper left", ncols=ax1_legend_ncols)
    ax1.set_xlim(min(x) / wt.diameter() - 20, max(x) / wt.diameter() + 10)
    ax1.set_xlabel(r"$x [D]$")
    ax1.set_ylabel(r"$y [D]$")

    plot_crossstream_predictions(probe_pred_sel, probe_target_sel, rel_y, marker=False, ax=ax2)
    ax2.set_xlabel(r"$u$  [m/s]")
    ax2.set_ylabel(r"$y [D]$")
    ax2.set_xlim(
        max(probe_targets_gen.flatten()) + ax2_xlim_additions[0],
        max(probe_targets_gen.flatten()) + ax2_xlim_additions[1],
    )
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(loc="upper left")

    ax3.plot(
        ((probe_pred_sel - probe_target_sel) / U_flow) * 100,
        rel_y,
        label="Probe  Error (%)",
        color="orange",
    )
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax3_xlim)
    ax3.set_yticks([])
    ax3.set_xlabel(r"$\dfrac{u - \hat{u}}{U}$ [%]")

    ax4.plot(
        rel_x[: y_idx + 1],
        rel_pred_abs_max_crossstream_U[: y_idx + 1],
        label=r"$\max \left(|\dfrac{u - \hat{u}}{U}|\right)$",
        color="purple",
    )
    ax4.scatter(
        rel_x[y_idx],
        rel_pred_abs_max_crossstream_U[y_idx],
        color="purple",
        s=10,
    )
    ax4.plot(
        rel_x[: y_idx + 1],
        rel_pred_abs_mean_crossstream_U[: y_idx + 1],
        label=r"$\dfrac{1}{N} \sum_{i=1}^N \left(|\dfrac{u - \hat{u}}{U}|\right)$",
        color="orange",
    )
    ax4.scatter(
        rel_x[y_idx],
        rel_pred_abs_mean_crossstream_U[y_idx],
        color="orange",
        s=10,
    )
    ax4.set_xlim(min(rel_x) - 2, max(rel_x) + 2)
    ax4.set_xlabel(r"$x [D]$")
    ax4.set_ylabel("Mean error [%]")
    ax4.legend(ncols=2, loc="upper right")
    ax4.set_ylim(min(rel_pred_abs_mean_crossstream) - 1, max(rel_pred_abs_max_crossstream_U) + 1)
    fig.tight_layout()


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(rel_x),
    interval=200,
)


def update_func(_i, _n):
    return progress_bar.update(1)


# FMMpegWriter = animation.FFMpegWriter(
#     fps=20,
#     codec="libx264",  # H.264 codec
#     extra_args=["-pix_fmt", "yuv420p"],  # ensures compatibility with most players
# )

with tqdm(total=len(rel_x), desc="Saving gif") as progress_bar:
    ani.save(
        os.path.join(
            fig_folder_path,
            f"probe_predictions_animation_{model_type_str}_{test_idx}.gif",
        ),
        # writer=FMMpegWriter,
        writer=animation.PillowWriter(fps=20),
        progress_callback=update_func,
        dpi=300,
    )

# %%
