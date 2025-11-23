import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from jax import numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from py_wake import HorizontalGrid
from py_wake.deficit_models import NiayifarGaussianDeficit, SelfSimilarityDeficit2020
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site import XRSite
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.wind_farm_models import All2AllIterative
from tqdm import tqdm

from utils.data_tools import retrieve_dataset_stats, setup_unscaler
from utils.plotting import matplotlib_set_rcparams
from utils.run_pywake import construct_on_the_fly_probe_graph
from utils.torch_loader import JraphDataLoader, Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

matplotlib_set_rcparams("paper")

base_dir = os.path.abspath(".")  # run with command line from repository base dir
# base_dir = os.path.abspath("../..") # run from Experiments/articles_plotting/ with iPython


# load layouts
resources_dir = os.path.join(
    base_dir, "Experiments", "articles_plotting", "IEA740_resources"
)

regular_grid_layout = yaml.safe_load(
    open(os.path.join(resources_dir, "ROWP_Regular.yaml"), "r")
)["layouts"]["initial_layout"]["coordinates"]

irregular_grid_layout = yaml.safe_load(
    open(os.path.join(resources_dir, "ROWP_Irregular.yaml"), "r")
)["layouts"]["initial_layout"]["coordinates"]

wind_resource = yaml.safe_load(
    open(os.path.join(resources_dir, "Wind_Resource.yaml"), "r")
)


ws_sw = 1  # Wind speed step width in [m/s] for wind rose discretization
wd_sw = 1  # Wind direction step width in [deg] for wind rose discretization
plot_power = (
    "on"  # 'on' or 'off', for plant power vs. wind speed and wind direction plots
)


# Extract site data
A = wind_resource["wind_resource"]["weibull_a"]
k = wind_resource["wind_resource"]["weibull_k"]
freq = wind_resource["wind_resource"]["sector_probability"]
wd = wind_resource["wind_resource"]["wind_direction"]
ws = wind_resource["wind_resource"]["wind_speed"]
TI = wind_resource["wind_resource"]["turbulence_intensity"]["data"]

site = XRSite(
    ds=xr.Dataset(
        data_vars={
            "Sector_frequency": ("wd", freq["data"]),
            "Weibull_A": ("wd", A["data"]),
            "Weibull_k": ("wd", k["data"]),
            "TI": (
                wind_resource["wind_resource"]["turbulence_intensity"]["dims"][0],
                TI,
            ),
        },
        coords={"wd": wd, "ws": ws},
    )
)
site.interp_method = "linear"

wt = DTU10MW()

# Windrose discretization to evaluate in pywake
ws_py = np.arange(4, 25 + ws_sw, ws_sw)
wd_py = np.arange(0, 360, wd_sw)
TI = np.interp(ws_py, ws, TI)


wf_model = All2AllIterative(
    site,
    wt,
    wake_deficitModel=NiayifarGaussianDeficit(),
    blockage_deficitModel=SelfSimilarityDeficit2020(),
    superpositionModel=LinearSum(),
    turbulenceModel=CrespoHernandez(),
)


# %% Initalize model
main_path = os.path.abspath(
    "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
)  # sophia
# main_path = "/home/jpsch/Documents/Sophia_work/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"  # local

if "Documents" in main_path:
    running_locally = True
    test_data_path = "/home/jpsch/Documents/Sophia_work/SPO_sophia_dir/data/large_graphs_nodes_2/test_pre_processed"  # run locally
    # test_data_path2 = "/home/jpsch/Documents/Sophia_work/SPO_sophia_dir/data/large_graphs_nodes_2/test_pre_processed"  # run locally
else:
    running_locally = False


cfg_path = os.path.join(main_path, ".hydra/config.yaml")
model_cfg_path = os.path.abspath(os.path.join(main_path, "model_config.json"))
params_paths = os.path.join(main_path, "best_params.msgpack")
if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"
fig_folder_path = os.path.join(main_path, "model/figures_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)


### Load
with open(model_cfg_path, "r") as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)


if not running_locally:
    test_data_path = restored_cfg_model.data.test_path  # Sophia server
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )
dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)


stats, scale_stats = retrieve_dataset_stats(dataset)
unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
_inverse_scale_target = unscaler.inverse_scale_output

restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
    params_paths, model_cfg_path, dataset
)


def model_prediction_fn(
    input_graphs,
    input_probe_graphs,
    input_wt_mask,
    input_probe_mask,
) -> jnp.ndarray:
    """This function assumes the graphs are padded"""

    prediction = model.apply(
        restored_params,
        input_graphs,
        input_probe_graphs,
        input_wt_mask,
        input_probe_mask,
    )

    return prediction


pred_fn = jax.jit(model_prediction_fn)


def setup_farm_layout(layout_type):
    if layout_type == "regular":
        x = regular_grid_layout["x"]
        y = regular_grid_layout["y"]
    elif layout_type == "irregular":
        x = irregular_grid_layout["x"]
        y = irregular_grid_layout["y"]
    else:
        raise ValueError("layout_type must be 'regular' or 'irregular'")

    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    graph_x = x - np.min(x) - x_range / 2
    graph_y = y - np.min(y) - y_range / 2
    return graph_x, graph_y


def rotate_graph_layout(graph_x, graph_y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotated_x = graph_x * cos_angle - graph_y * sin_angle
    rotated_y = graph_x * sin_angle + graph_y * cos_angle

    return rotated_x, rotated_y


def calc_turbine_power_for_wind_conditions(cur_ws, cur_TI, wd_py, graph_x, graph_y):
    power_list = []
    power_targets_list = []
    for wd_i in tqdm(wd_py, desc="Constructing probe graphs"):
        rotated_x, rotated_y = rotate_graph_layout(graph_x, graph_y, -wd_i)

        grid = HorizontalGrid(
            x=[0],
            y=[0],
            h=wt.hub_height(),
        )

        jraph_graph_gen, jraph_probe_graphs_gen, node_array_tuple_gen = (
            construct_on_the_fly_probe_graph(
                positions=jnp.array(list(zip(rotated_x, rotated_y))),
                U=[cur_ws],
                TI=[cur_TI],
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

        predictions = pred_fn(
            jraph_graph_gen,
            jraph_probe_graphs_gen,
            jnp.atleast_2d(wt_mask_gen).T,
            jnp.atleast_2d(probe_mask_gen).T,
        ).squeeze()
        # use mask to extract only valid predictions
        wt_indexes = jnp.where(wt_mask_gen == 1)[0]
        wt_predictions = _inverse_scale_target(predictions[wt_indexes])

        wt_targets = _inverse_scale_target(targets_gen[wt_indexes])

        power = wt.power(wt_predictions)
        power_targets_list.append(wt.power(wt_targets.squeeze()))
        power_list.append(power)
        farm_power = np.sum(np.array(power_list), axis=1)
        farm_power_targets = np.sum(np.array(power_targets_list), axis=1)

    return power_list, power_targets_list, farm_power, farm_power_targets


# Plot power vs. wind speed and wind direction

wind = [14, 12, 10, 8, 6]  # wind speeds to evaluate in pywake
# retrieve colors from RSParams chosen colormap as set with matplotlib_set_rcparams
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(wind))]


# %% Pre-calc

# layout selection
Res_dict = {}
Res_targets_dict = {}
for layout_type in ["regular", "irregular"]:

    # initially the farm is rotated to match 270 deg wind direction the GNO uses
    graph_x, graph_y = setup_farm_layout(layout_type)
    graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)

    Res = []  # solution matrix
    Res_targets = []  # target solution matrix
    for i in range(len(wind)):
        cur_ws = [wind[i]]
        cur_TI = [0.05]
        _, _, farm_power, farm_power_targets = calc_turbine_power_for_wind_conditions(
            cur_ws, cur_TI, wd_py, graph_x, graph_y
        )
        Res.append(farm_power / 740e6)
        Res_targets.append(farm_power_targets / 740e6)
    Res_dict[layout_type] = Res
    Res_targets_dict[layout_type] = Res_targets

# %% Plot content
fig, axes = plt.subplots(
    1,
    2,
    figsize=(4.5, 3),
    subplot_kw={"projection": "polar"},
)
for i_ax, (ax, Res, Res_targets) in enumerate(
    zip(
        axes,
        [Res_dict["regular"], Res_dict["irregular"]],
        [Res_targets_dict["regular"], Res_targets_dict["irregular"]],
    )
):
    ax.set_theta_direction(-1)

    for i in range(len(wind)):
        cur_ws = [wind[i]]
        cur_TI = [0.05]
        ax.plot(
            np.deg2rad(np.arange(0, 360, 1)),
            Res[i],
            label=str(cur_ws[0]) + " m/s",
            color=colors[i],
            linestyle="solid",
        )
        ax.plot(
            np.deg2rad(np.arange(0, 360, 1)),
            Res_targets[i],
            color=colors[i],
            linestyle="dotted",
        )

    # decorate plot
    ax.set_theta_zero_location("N")
    ax.set_ylim([0, 1.01])
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(np.linspace(0, 2 * np.pi * 7 / 8, 8))
    ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""])
# Create a legend above subplot 0 (regular layout), with black dashed lines for GNO and solid lines for PyWake (targets) and then solid colored lines for different wind speeds
lines = []
labels = []

(line_gno,) = axes[0].plot(
    [],
    [],
    color="black",
    label="GNO",
    linestyle="solid",
)
lines.append(line_gno)
labels.append("GNO")
# add PyWake legend entry
(line_pywake,) = axes[0].plot(
    [],
    [],
    color="black",
    label="PyWake",
    linestyle="dotted",
)
lines.append(line_pywake)
labels.append("PyWake")

# # add an empty line for spacing
# line_space, = axes[0].plot(
#     [],
#     [],
#     color="white",
#     label="",
#     linestyle="solid",
# )
# lines.append(line_space)
# labels.append("")
wind_lines = []
wind_labels = []
for i in range(len(wind)):
    (line,) = axes[0].plot(
        [],
        [],
        color=colors[i],
        label=str(wind[i]) + " m/s",
        linestyle="solid",
    )
    wind_lines.append(line)
    wind_labels.append(str(wind[i]) + " m/s")
wind_lines = wind_lines[::-1]
wind_labels = wind_labels[::-1]
lines.extend(wind_lines)
labels.extend(wind_labels)

# add GNO legend entry

fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=4,
    fontsize=10,
    frameon=True,
)
# add subplot letter (a) and (b)
subplot_labels = ["(a) Regular", "(b) Irregular"]
for ax, label in zip(axes, subplot_labels):
    ax.text(
        0.0,
        1.1,
        label,
        transform=ax.transAxes,
        # fontsize=12,
        # fontweight="bold",
        ha="center",
    )
fig.tight_layout()
plt.savefig(
    os.path.join(fig_folder_path, "IEA740_power_rose.pdf"),
    bbox_inches="tight",
)


# %% Plot layouts

fig, axes = plt.subplots(
    1,
    2,
    figsize=(5.5, 3),
    sharey=True,
)

subplot_labels = ["(a) Regular", "(b) Irregular"]
for i_ax, (ax, layout_type, label) in enumerate(
    zip(axes, ["regular", "irregular"], subplot_labels)
):
    graph_x, graph_y = setup_farm_layout(layout_type)
    ax.scatter(graph_x / wt.diameter(), graph_y / wt.diameter(), color="blue", s=10)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$x/D$ [-]")
    if i_ax == 0:
        ax.set_ylabel(r"$y/D$ [-]")
    ax.grid(True)

    # add subplot letter (a) and (b)
    ax.text(
        0.32 + i_ax * 0.04,
        0.9,
        label,
        transform=ax.transAxes,
        # fontsize=12,
        # fontweight="bold",
        ha="center",
    )

# fig.tight_layout()
plt.savefig(
    os.path.join(fig_folder_path, "IEA740_layouts.pdf"),
    bbox_inches="tight",
)


# %%
