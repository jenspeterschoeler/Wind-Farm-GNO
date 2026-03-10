import json
import os
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import xarray as xr
import yaml
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib_map_utils import NorthArrow
from omegaconf import DictConfig
import cmcrameri.cm as cmc
from matplotlib.colors import Normalize


def add_north_arrow(ax, location="upper left", scale=0.25):
    """Add a north arrow using matplotlib-map-utils.

    Args:
        ax: matplotlib axes
        location: location string (e.g., "upper left", "upper right")
        scale: size scaling factor
    """
    # Get current rcParams font family to match plot style
    font_family = plt.rcParams.get("font.family", ["serif"])[0]

    na = NorthArrow(
        location=location,
        scale=scale,
        rotation={"degrees": 0},
        shadow=False,  # Remove shadow
        label={
            "fontfamily": font_family,
            "fontsize": 10,
            "fontweight": "bold",
        },
    )
    ax.add_artist(na)


class CenteredPlateauNorm(Normalize):
    """Normalization with a plateau (white band) around the center value.

    Maps values to [0, 1] such that a range around vcenter all map to 0.5 (white).
    """

    def __init__(self, vmin, vmax, vcenter=1.0, plateau_half_width=0.01):
        self.vcenter = vcenter
        self.plateau_half_width = plateau_half_width
        super().__init__(vmin, vmax)

    def __call__(self, value, clip=None):
        value = np.asarray(value)
        vmin, vmax = self.vmin, self.vmax
        vcenter = self.vcenter
        hw = self.plateau_half_width

        result = np.zeros_like(value, dtype=float)

        # Below plateau: map [vmin, vcenter-hw] -> [0, 0.5]
        below = value < (vcenter - hw)
        if np.any(below):
            result[below] = 0.5 * (value[below] - vmin) / (vcenter - hw - vmin)

        # In plateau: map to 0.5 (white)
        in_plateau = (value >= vcenter - hw) & (value <= vcenter + hw)
        result[in_plateau] = 0.5

        # Above plateau: map [vcenter+hw, vmax] -> [0.5, 1.0]
        above = value > (vcenter + hw)
        if np.any(above):
            result[above] = 0.5 + 0.5 * (value[above] - (vcenter + hw)) / (vmax - (vcenter + hw))

        return np.ma.masked_array(result)


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

from utils.data_tools import setup_unscaler
from utils.plotting import matplotlib_set_rcparams
from utils.run_pywake import construct_on_the_fly_probe_graph
from utils.weight_converter import load_portable_model

matplotlib_set_rcparams("paper")

base_dir = os.path.abspath(".")  # run with command line from repository base dir
# base_dir = os.path.abspath("../..") # run from Experiments/articles_plotting/ with iPython


# load layouts
resources_dir = os.path.join(base_dir, "Experiments", "article_1", "IEA740_resources")

regular_grid_layout = yaml.safe_load(open(os.path.join(resources_dir, "ROWP_Regular.yaml"), "r"))[
    "layouts"
]["initial_layout"]["coordinates"]

irregular_grid_layout = yaml.safe_load(
    open(os.path.join(resources_dir, "ROWP_Irregular.yaml"), "r")
)["layouts"]["initial_layout"]["coordinates"]

wind_resource = yaml.safe_load(open(os.path.join(resources_dir, "Wind_Resource.yaml"), "r"))


ws_sw = 1  # Wind speed step width in [m/s] for wind rose discretization
wd_sw = 1  # Wind direction step width in [deg] for wind rose discretization
plot_power = "on"  # 'on' or 'off', for plant power vs. wind speed and wind direction plots


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
# Use local model from assets - no dataset required
main_path = os.path.join(base_dir, "assets", "best_model_Vj8")

model_cfg_path = os.path.join(main_path, "model_config.json")
params_paths = os.path.join(main_path, "best_params.msgpack")
scale_stats_path = os.path.join(main_path, "scale_stats.json")
model_type_str = "best_model"
fig_folder_path = os.path.join(main_path, "model/figures_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)


### Load model config
with open(model_cfg_path) as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)

### Load scale_stats from JSON (no dataset needed)
with open(scale_stats_path) as f:
    scale_stats = json.load(f)

unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
_inverse_scale_target = unscaler.inverse_scale_output


### Initialize model using a dummy probe graph from IEA740 layout
# We generate a single probe graph to get the correct input shapes for model init
print("Initializing model with dummy probe graph...")
_init_x = regular_grid_layout["x"]
_init_y = regular_grid_layout["y"]
_init_x_centered = np.array(_init_x) - np.mean(_init_x)
_init_y_centered = np.array(_init_y) - np.mean(_init_y)
_init_positions = jnp.array(list(zip(_init_x_centered, _init_y_centered)))

_init_grid = HorizontalGrid(x=[0], y=[0], h=wt.hub_height())
_init_jraph_graph, _init_probe_graph, _init_node_tuple = construct_on_the_fly_probe_graph(
    positions=_init_positions,
    U=[10.0],
    TI=[0.05],
    grid=_init_grid,
    scale_stats=scale_stats,
    return_positions=False,
)
# Reshape masks for broadcasting: (n_nodes,) -> (n_nodes, 1)
_init_targets, _init_wt_mask, _init_probe_mask = _init_node_tuple
_init_wt_mask = jnp.atleast_2d(_init_wt_mask).T
_init_probe_mask = jnp.atleast_2d(_init_probe_mask).T
_init_node_tuple_reshaped = (_init_targets, _init_wt_mask, _init_probe_mask)

restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
    params_paths,
    model_cfg_path,
    dataset=None,
    inputs=(_init_jraph_graph, _init_probe_graph, _init_node_tuple_reshaped),
)
print("Model initialized successfully.")


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


# ============================================================================
# Flow Field Visualization Helper Functions
# ============================================================================


def generate_flow_field_grid(wt_positions, grid_density=3, x_extent=None, y_extent=None):
    """Generate a HorizontalGrid for flow field visualization.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        grid_density: Number of grid points per rotor diameter
        x_extent: Tuple (x_min, x_max) in meters. Defaults to 10D upstream to 100D downstream
        y_extent: Tuple (y_min, y_max) in meters. Defaults to 5D beyond farm boundaries

    Returns:
        grid: HorizontalGrid object
        xx: 2D meshgrid of x coordinates
        yy: 2D meshgrid of y coordinates
    """
    D = wt.diameter()

    # Get farm extent
    x_min_wt = np.min(wt_positions[:, 0])
    x_max_wt = np.max(wt_positions[:, 0])
    y_min_wt = np.min(wt_positions[:, 1])
    y_max_wt = np.max(wt_positions[:, 1])

    # Default extent: 10D upstream, 100D downstream, 5D lateral buffer
    # (matches TurbOPark generation settings)
    if x_extent is None:
        x_min = x_min_wt - 10 * D
        x_max = x_max_wt + 100 * D
    else:
        x_min = x_extent[0]
        x_max = x_extent[1]

    if y_extent is None:
        y_min = y_min_wt - 5 * D
        y_max = y_max_wt + 5 * D
    else:
        y_min = y_extent[0]
        y_max = y_extent[1]

    # Create grid
    n_x = int((x_max - x_min) / D * grid_density)
    n_y = int((y_max - y_min) / D * grid_density)

    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)

    grid = HorizontalGrid(
        x=x,
        y=y,
        h=wt.hub_height(),
    )

    xx, yy = np.meshgrid(x, y)
    return grid, xx, yy, x, y


def compute_unified_grid_extent(grid_density=3):
    """Compute the unified grid extent that covers both regular and irregular layouts.

    Returns:
        x_extent: Tuple (x_min, x_max) in meters
        y_extent: Tuple (y_min, y_max) in meters
    """
    D = wt.diameter()

    # Get positions for both layouts
    all_x = []
    all_y = []
    for layout_type in ["regular", "irregular"]:
        graph_x, graph_y = setup_farm_layout(layout_type)
        graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)
        all_x.extend(graph_x)
        all_y.extend(graph_y)

    x_min_wt = np.min(all_x)
    x_max_wt = np.max(all_x)
    y_min_wt = np.min(all_y)
    y_max_wt = np.max(all_y)

    # Extent: 10D upstream, 100D downstream, 5D lateral buffer
    # (matches TurbOPark generation settings)
    x_extent = (x_min_wt - 10 * D, x_max_wt + 100 * D)
    y_extent = (y_min_wt - 5 * D, y_max_wt + 5 * D)

    return x_extent, y_extent


def get_gno_flow_field(wt_positions, x_vals, y_vals, U_flow, TI_flow):
    """Get GNO flow field predictions using column-by-column iteration.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        x_vals: 1D array of x coordinates
        y_vals: 1D array of y coordinates
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]

    Returns:
        predictions: 2D array (n_y, n_x) of velocity predictions in m/s
    """
    predictions_list = []

    for x_sel in tqdm(x_vals, desc="GNO flow field columns", leave=False):
        grid = HorizontalGrid(
            x=[x_sel],
            y=y_vals,
            h=wt.hub_height(),
        )

        jraph_graph_gen, jraph_probe_graphs_gen, node_array_tuple_gen = (
            construct_on_the_fly_probe_graph(
                positions=jnp.array(wt_positions),
                U=[U_flow],
                TI=[TI_flow],
                grid=grid,
                scale_stats=scale_stats,
                return_positions=True,
            )
        )
        (
            _,  # targets_gen (not needed for flow field)
            wt_mask_gen,
            probe_mask_gen,
            _,
        ) = node_array_tuple_gen

        predictions = pred_fn(
            jraph_graph_gen,
            jraph_probe_graphs_gen,
            jnp.atleast_2d(wt_mask_gen).T,
            jnp.atleast_2d(probe_mask_gen).T,
        ).squeeze()

        # Extract probe predictions only
        probe_predictions = jnp.where(probe_mask_gen != 0, predictions, jnp.nan)
        probe_predictions = probe_predictions[~jnp.isnan(probe_predictions)]

        # Inverse scale to physical units
        probe_predictions = _inverse_scale_target(probe_predictions)

        predictions_list.append(probe_predictions.reshape(-1, 1))

    return np.concatenate(predictions_list, axis=1)


def get_pywake_flow_field(wt_positions, grid, U_flow, TI_flow):
    """Get PyWake flow field simulation.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        grid: HorizontalGrid object
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]

    Returns:
        flow_field: 2D array (n_y, n_x) of velocity values in m/s
    """
    x_wt = wt_positions[:, 0]
    y_wt = wt_positions[:, 1]

    # Run PyWake simulation (wind from west = 270 deg)
    farm_sim = wf_model(x_wt, y_wt, wd=270, ws=U_flow, TI=TI_flow)
    flow_map = farm_sim.flow_map(grid=grid, wd=270, ws=U_flow)

    # Extract effective wind speed
    ws_eff = flow_map.WS_eff.values.squeeze()

    return ws_eff


def get_flow_field_cache_path(U_flow, TI_flow, layout_type, grid_density, cache_dir):
    """Get the cache file path for flow field data."""
    return os.path.join(
        cache_dir,
        f"flow_field_U{U_flow}_TI{TI_flow}_{layout_type}_grid{grid_density}.pkl",
    )


def get_flow_field_cache_path_unified(U_flow, TI_flow, layout_type, grid_density, cache_dir):
    """Get the cache file path for unified flow field data."""
    return os.path.join(
        cache_dir,
        f"flow_field_unified_U{U_flow}_TI{TI_flow}_{layout_type}_grid{grid_density}.pkl",
    )


def compute_and_cache_flow_field(
    U_flow,
    TI_flow,
    layout_type,
    grid_density,
    cache_dir,
    force_recompute=False,
    x_extent=None,
    y_extent=None,
):
    """Compute flow field data for a single layout, with caching.

    Args:
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        layout_type: 'regular' or 'irregular'
        grid_density: Number of grid points per rotor diameter
        cache_dir: Directory for cache files
        force_recompute: If True, ignore cache and recompute
        x_extent: Optional tuple (x_min, x_max) in meters for unified grid
        y_extent: Optional tuple (y_min, y_max) in meters for unified grid

    Returns:
        dict with flow field data
    """
    # Use unified cache path if extents provided
    if x_extent is not None or y_extent is not None:
        cache_path = get_flow_field_cache_path_unified(
            U_flow, TI_flow, layout_type, grid_density, cache_dir
        )
    else:
        cache_path = get_flow_field_cache_path(
            U_flow, TI_flow, layout_type, grid_density, cache_dir
        )

    # Try to load from cache
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached data: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"\nComputing {layout_type} layout at U={U_flow} m/s...")

    # Setup farm layout (centered, rotated for 270 deg wind)
    graph_x, graph_y = setup_farm_layout(layout_type)
    graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)
    wt_positions = np.column_stack([graph_x, graph_y])

    # Generate grid (using unified extent if provided)
    grid, xx, yy, x_vals, y_vals = generate_flow_field_grid(
        wt_positions, grid_density=grid_density, x_extent=x_extent, y_extent=y_extent
    )

    # Get GNO predictions
    gno_field = get_gno_flow_field(wt_positions, x_vals, y_vals, U_flow, TI_flow)

    # Get PyWake ground truth
    pywake_field = get_pywake_flow_field(wt_positions, grid, U_flow, TI_flow)

    # Compute velocity deficit: (U - u) / U
    gno_deficit = (U_flow - gno_field) / U_flow
    pywake_deficit = (U_flow - pywake_field) / U_flow

    # Compute absolute error: |u - u'| in m/s
    error = np.abs(gno_field - pywake_field)

    data = {
        "wt_positions": wt_positions,
        "xx": xx,
        "yy": yy,
        "gno_field": gno_field,
        "pywake_field": pywake_field,
        "gno_deficit": gno_deficit,
        "pywake_deficit": pywake_deficit,
        "error": error,
        "U_flow": U_flow,
        "TI_flow": TI_flow,
        "layout_type": layout_type,
        "grid_density": grid_density,
    }

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cached: {cache_path}")

    return data


def plot_flow_field_comparison(
    U_flow, TI_flow, grid_density=3, save_path=None, cache_dir=None, force_recompute=False
):
    """Create 2x3 flow field comparison figure for both layouts.

    Args:
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        grid_density: Number of grid points per rotor diameter
        save_path: Optional path to save figure
        cache_dir: Directory for caching flow field data (enables caching if provided)
        force_recompute: If True, ignore cache and recompute

    Returns:
        fig: matplotlib figure object
    """
    D = wt.diameter()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )

    # Compute unified grid extent covering both layouts
    x_extent, y_extent = compute_unified_grid_extent(grid_density)

    # Storage for shared colorbar ranges
    u_min, u_max = np.inf, -np.inf
    error_max = 0

    layout_data = {}

    # Process both layouts (with caching if cache_dir provided)
    for layout_type in ["regular", "irregular"]:
        if cache_dir:
            data = compute_and_cache_flow_field(
                U_flow,
                TI_flow,
                layout_type,
                grid_density,
                cache_dir,
                force_recompute,
                x_extent=x_extent,
                y_extent=y_extent,
            )
        else:
            # Original computation without caching
            print(f"\nProcessing {layout_type} layout at U={U_flow} m/s...")

            graph_x, graph_y = setup_farm_layout(layout_type)
            graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)
            wt_positions = np.column_stack([graph_x, graph_y])

            grid, xx, yy, x_vals, y_vals = generate_flow_field_grid(
                wt_positions,
                grid_density=grid_density,
                x_extent=x_extent,
                y_extent=y_extent,
            )

            gno_field = get_gno_flow_field(wt_positions, x_vals, y_vals, U_flow, TI_flow)
            pywake_field = get_pywake_flow_field(wt_positions, grid, U_flow, TI_flow)

            gno_deficit = (U_flow - gno_field) / U_flow
            pywake_deficit = (U_flow - pywake_field) / U_flow
            error = np.abs(gno_field - pywake_field)

            data = {
                "wt_positions": wt_positions,
                "xx": xx,
                "yy": yy,
                "gno_field": gno_field,
                "pywake_field": pywake_field,
                "gno_deficit": gno_deficit,
                "pywake_deficit": pywake_deficit,
                "error": error,
            }

        # Update colorbar ranges for effective wind speed
        u_min = min(u_min, np.nanmin(data["gno_field"]), np.nanmin(data["pywake_field"]))
        u_max = max(u_max, np.nanmax(data["gno_field"]), np.nanmax(data["pywake_field"]))
        # Compute error on-the-fly from raw fields (allows changing error metric without recomputing)
        error = np.abs(data["gno_field"] - data["pywake_field"])
        error_max = max(error_max, np.nanpercentile(error, 99))

        layout_data[layout_type] = data

    # Ensure valid range
    u_max = max(u_max, u_min + 0.01)  # Avoid zero range

    # Create contour levels for effective wind speed
    u_levels = np.linspace(u_min, u_max, 50)
    error_levels = np.linspace(0, error_max, 50)

    # Plot both layouts
    row_labels = ["Regular", "Irregular"]
    col_labels = ["GNO", "PyWake", r"$|u - \widehat{u}|$"]
    subplot_letters = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    contour_handles = {}

    for i_row, layout_type in enumerate(["regular", "irregular"]):
        data = layout_data[layout_type]
        wt_pos_D = data["wt_positions"] / D
        xxD = data["xx"] / D
        yyD = data["yy"] / D

        # GNO effective wind speed
        ax = axes[i_row, 0]
        cf_gno = ax.contourf(
            xxD,
            yyD,
            data["gno_field"],
            levels=u_levels,
            cmap="viridis",
            extend="both",
        )
        ax.scatter(
            wt_pos_D[:, 0],
            wt_pos_D[:, 1],
            marker="2",
            s=50,
            color="white",
            zorder=10,
        )
        contour_handles["u"] = cf_gno

        # PyWake effective wind speed
        ax = axes[i_row, 1]
        ax.contourf(
            xxD,
            yyD,
            data["pywake_field"],
            levels=u_levels,
            cmap="viridis",
            extend="both",
        )
        ax.scatter(
            wt_pos_D[:, 0],
            wt_pos_D[:, 1],
            marker="2",
            s=50,
            color="white",
            zorder=10,
        )

        # Error (computed on-the-fly from raw fields)
        error = np.abs(data["gno_field"] - data["pywake_field"])
        ax = axes[i_row, 2]
        cf_err = ax.contourf(
            xxD,
            yyD,
            error,
            levels=error_levels,
            cmap="Reds",
            extend="max",
        )
        ax.scatter(
            wt_pos_D[:, 0],
            wt_pos_D[:, 1],
            marker="2",
            s=50,
            color="white",
            zorder=10,
        )
        contour_handles["error"] = cf_err

    # Add labels and formatting
    for i_row in range(2):
        for i_col in range(3):
            ax = axes[i_row, i_col]
            ax.set_aspect("equal", "box")

            # Add subplot letter
            ax.text(
                0.02,
                0.98,
                subplot_letters[i_row][i_col],
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
            )

            # Column titles (top row only)
            if i_row == 0:
                ax.set_title(col_labels[i_col])

            # Row labels (left column only)
            if i_col == 0:
                ax.set_ylabel(f"{row_labels[i_row]}\n" + r"$y/D$ [-]")
            else:
                ax.set_ylabel("")

            # X labels (bottom row only)
            if i_row == 1:
                ax.set_xlabel(r"$x/D$ [-]")

    # Add colorbars with improved tick formatting and direction arrows
    fig.subplots_adjust(bottom=0.18, wspace=0.05, hspace=0.1)

    # Effective wind speed colorbar (spans columns 0-1)
    cbar_ax1 = fig.add_axes([0.125, 0.06, 0.5, 0.02])
    cbar1 = fig.colorbar(
        contour_handles["u"],
        cax=cbar_ax1,
        orientation="horizontal",
    )
    cbar1.set_label(r"$u$ [$\mathrm{m\,s^{-1}}$]")
    # Set limited number of ticks for readability
    u_ticks = np.linspace(u_min, u_max, 5)
    cbar1.set_ticks(u_ticks)
    cbar1.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Error colorbar (column 2)
    cbar_ax2 = fig.add_axes([0.68, 0.06, 0.22, 0.02])
    cbar2 = fig.colorbar(
        contour_handles["error"],
        cax=cbar_ax2,
        orientation="horizontal",
    )
    cbar2.set_label(r"$|u - \widehat{u}|$ [$\mathrm{m\,s^{-1}}$]")
    # Set limited number of ticks for readability
    error_ticks = np.linspace(0, error_max, 4)
    cbar2.set_ticks(error_ticks)
    cbar2.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Suptitle with wind condition
    fig.suptitle(
        rf"$U = {U_flow}$ $\mathrm{{m\,s^{{-1}}}}$, TI $= {TI_flow}$",
        y=0.98,
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_flow_field_stacked(
    wind_speeds,
    TI_flow,
    grid_density=3,
    save_path=None,
    cache_dir=None,
    force_recompute=False,
):
    """Create stacked 6-row flow field figure: 3 regular + 3 irregular layouts.

    Args:
        wind_speeds: List of wind speeds [m/s] (e.g., [6, 12, 18])
        TI_flow: Turbulence intensity [-]
        grid_density: Number of grid points per rotor diameter
        save_path: Optional path to save figure
        cache_dir: Directory for caching flow field data
        force_recompute: If True, ignore cache and recompute

    Returns:
        fig: matplotlib figure object
    """
    D = wt.diameter()
    n_speeds = len(wind_speeds)

    # 6 rows: 3 regular (top) + 3 irregular (bottom), 3 columns: GNO, PyWake, Error
    fig, axes = plt.subplots(
        2 * n_speeds,
        3,
        figsize=(10, 3 * n_speeds),
        sharex=True,
        sharey=True,
    )

    # Compute unified grid extent covering both layouts
    x_extent, y_extent = compute_unified_grid_extent(grid_density)

    # Collect all data first to determine global colorbar ranges
    # Normalize by freestream velocity for better comparability across wind speeds
    all_data = {}
    u_norm_min_global, u_norm_max_global = np.inf, -np.inf
    error_pct_max_global = 0

    for U_flow in wind_speeds:
        for layout_type in ["regular", "irregular"]:
            if cache_dir:
                data = compute_and_cache_flow_field(
                    U_flow,
                    TI_flow,
                    layout_type,
                    grid_density,
                    cache_dir,
                    force_recompute,
                    x_extent=x_extent,
                    y_extent=y_extent,
                )
            else:
                print(f"\nProcessing {layout_type} layout at U={U_flow} m/s...")
                graph_x, graph_y = setup_farm_layout(layout_type)
                graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)
                wt_positions = np.column_stack([graph_x, graph_y])

                grid, xx, yy, x_vals, y_vals = generate_flow_field_grid(
                    wt_positions,
                    grid_density=grid_density,
                    x_extent=x_extent,
                    y_extent=y_extent,
                )

                gno_field = get_gno_flow_field(wt_positions, x_vals, y_vals, U_flow, TI_flow)
                pywake_field = get_pywake_flow_field(wt_positions, grid, U_flow, TI_flow)

                data = {
                    "wt_positions": wt_positions,
                    "xx": xx,
                    "yy": yy,
                    "gno_field": gno_field,
                    "pywake_field": pywake_field,
                    "U_flow": U_flow,
                }

            all_data[(U_flow, layout_type)] = data

            # Get freestream velocity for normalization
            U = data.get("U_flow", U_flow)

            # Compute normalized fields (u/U)
            gno_norm = data["gno_field"] / U
            pywake_norm = data["pywake_field"] / U

            # Update global ranges for normalized velocity
            u_norm_min_global = min(u_norm_min_global, np.nanmin(gno_norm), np.nanmin(pywake_norm))
            u_norm_max_global = max(u_norm_max_global, np.nanmax(gno_norm), np.nanmax(pywake_norm))

            # Compute error in percentage: |(u - u')/U| * 100 (normalized by freestream)
            error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100
            error_pct_max_global = max(error_pct_max_global, np.nanpercentile(error_pct, 99))

    # Ensure valid range
    u_norm_max_global = max(u_norm_max_global, u_norm_min_global + 0.01)

    # Create contour levels for normalized velocity and percentage error
    # Levels span the data range
    u_norm_levels = np.linspace(u_norm_min_global, u_norm_max_global, 50)
    error_pct_levels = np.linspace(0, error_pct_max_global, 50)

    # Create normalization centered at 1 (unwaked flow) for the VIK colormap
    # Values below 1 show wake deficit, values above 1 show speedup
    # Use plateau to make values near 1.0 (0.99-1.01) appear white
    u_norm_centered = CenteredPlateauNorm(
        vmin=u_norm_min_global, vmax=u_norm_max_global, vcenter=1.0, plateau_half_width=0.01
    )

    contour_handles = {}

    # Plot: first regular layouts (rows 0 to n_speeds-1), then irregular (rows n_speeds to 2*n_speeds-1)
    subplot_letters = [
        ["(a)", "(b)", "(c)"],
        ["(d)", "(e)", "(f)"],
        ["(g)", "(h)", "(i)"],
        ["(j)", "(k)", "(l)"],
        ["(m)", "(n)", "(o)"],
        ["(p)", "(q)", "(r)"],
    ]

    for layout_idx, layout_type in enumerate(["regular", "irregular"]):
        for speed_idx, U_flow in enumerate(wind_speeds):
            i_row = layout_idx * n_speeds + speed_idx
            data = all_data[(U_flow, layout_type)]

            xxD = data["xx"] / D
            yyD = data["yy"] / D

            # Get freestream velocity for normalization
            U = data.get("U_flow", U_flow)

            # Normalized GNO flow field (u/U)
            ax = axes[i_row, 0]
            cf_gno = ax.contourf(
                xxD,
                yyD,
                data["gno_field"] / U,
                levels=u_norm_levels,
                cmap=cmc.vik_r,
                norm=u_norm_centered,
                extend="both",
            )
            contour_handles["u"] = cf_gno

            # Normalized PyWake flow field (u/U)
            ax = axes[i_row, 1]
            ax.contourf(
                xxD,
                yyD,
                data["pywake_field"] / U,
                levels=u_norm_levels,
                cmap=cmc.vik_r,
                norm=u_norm_centered,
                extend="both",
            )

            # Error in percentage: |(u - u')/U| * 100 (normalized by freestream)
            error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100

            ax = axes[i_row, 2]
            cf_err = ax.contourf(
                xxD,
                yyD,
                error_pct,
                levels=error_pct_levels,
                cmap="Reds",
                extend="max",
            )
            contour_handles["error"] = cf_err

    # Add labels and formatting
    col_labels = ["GNO", "PyWake", "Error"]

    for i_row in range(2 * n_speeds):
        layout_idx = i_row // n_speeds  # 0 for regular, 1 for irregular
        speed_idx = i_row % n_speeds
        U_flow = wind_speeds[speed_idx]

        for i_col in range(3):
            ax = axes[i_row, i_col]
            ax.set_aspect("equal", "box")

            # Add subplot letter with inflow velocity for first column
            if i_col == 0:
                label_text = (
                    subplot_letters[i_row][i_col]
                    + r" - $"
                    + f"{U_flow}"
                    + r"$ $\mathrm{m\,s^{-1}}$"
                )
            else:
                label_text = subplot_letters[i_row][i_col]

            ax.text(
                0.02,
                0.98,
                label_text,
                transform=ax.transAxes,
                fontsize=9,
                color="black",
                va="top",
                ha="left",
            )

            # Column titles (top row only)
            if i_row == 0:
                ax.set_title(col_labels[i_col], fontweight="bold")

            # Row labels (left column only) - just y/D
            if i_col == 0:
                ax.set_ylabel(r"$y/D$ [-]")
            else:
                ax.set_ylabel("")

            # X labels (bottom row only)
            if i_row == 2 * n_speeds - 1:
                ax.set_xlabel(r"$x/D$ [-]")

    # Add colorbars (moved down to avoid overlap with xlabels)
    fig.subplots_adjust(left=0.08, bottom=0.10, top=0.95, wspace=0.05, hspace=0.15)

    # Add layout supertitles on the y-axis (Regular / Irregular)
    # Position them on the left side, centered vertically for each layout section
    fig.text(
        0.005,
        0.73,
        "Regular",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )
    fig.text(
        0.005,
        0.32,
        "Irregular",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        rotation=90,
        transform=fig.transFigure,
    )

    # Add north arrow in top left of figure (outside subplots)
    # After the 270° rotation: +X = North, so arrow should point right
    # Create a small inset axes for the north arrow
    na_ax = fig.add_axes([0.01, 0.93, 0.04, 0.04])  # [left, bottom, width, height]
    na_ax.set_xlim(0, 1)
    na_ax.set_ylim(0, 1)
    na_ax.axis("off")

    font_family = plt.rcParams.get("font.family", ["serif"])[0]
    na = NorthArrow(
        location="center",
        scale=0.4,
        rotation={"degrees": -90},  # Point right (North after 270° rotation)
        shadow=False,
        label={
            "fontfamily": font_family,
            "fontsize": 8,
            "fontweight": "bold",
        },
    )
    na_ax.add_artist(na)

    # Normalized velocity colorbar (spans columns 0-1)
    cbar_ax1 = fig.add_axes([0.125, 0.02, 0.5, 0.015])
    cbar1 = fig.colorbar(
        contour_handles["u"],
        cax=cbar_ax1,
        orientation="horizontal",
    )
    cbar1.set_label(r"$u/U$ [-]")
    # Default ticks from min to max
    u_norm_ticks = np.linspace(u_norm_min_global, u_norm_max_global, 5)
    cbar1.set_ticks(u_norm_ticks)
    cbar1.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    # Add 1.0 tick on top of colorbar as a reference for unwaked flow
    cbar1.ax.axvline(x=1.0, color="black", linewidth=1.0, linestyle="-")
    cbar1.ax.text(
        0.985,
        1.3,
        "1.0",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=cbar1.ax.get_xaxis_transform(),
    )

    # Error colorbar in percentage (column 2)
    cbar_ax2 = fig.add_axes([0.68, 0.02, 0.22, 0.015])
    cbar2 = fig.colorbar(
        contour_handles["error"],
        cax=cbar_ax2,
        orientation="horizontal",
    )
    cbar2.set_label(r"$|(u - \widehat{u})/U|$ [$\%$]")
    error_pct_ticks = np.linspace(0, error_pct_max_global, 4)
    cbar2.set_ticks(error_pct_ticks)
    cbar2.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

    return fig


def calc_turbine_power_for_wind_conditions(cur_ws, cur_TI, wd_py, graph_x, graph_y):
    power_list = []
    power_targets_list = []
    for wd_i in tqdm(wd_py, desc="Constructing probe graphs"):
        # Rotate farm so that met wind direction wd_i appears as wind from -x (GNO convention)
        # Combined with 90° pre-rotation, total rotation = 90° + wd_i
        rotated_x, rotated_y = rotate_graph_layout(graph_x, graph_y, wd_i)

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
            _,  # node_positions_gen (not needed here)
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


# %% Output directories
# Define output directories early so all plots can use the same location
flow_field_output_dir = os.path.join(base_dir, "assets", "IEA740_flow_fields")
flow_field_cache_dir = os.path.join(flow_field_output_dir, "cache")
os.makedirs(flow_field_output_dir, exist_ok=True)
os.makedirs(flow_field_cache_dir, exist_ok=True)

# %% Pre-calc (with caching)

windrose_cache_dir = flow_field_cache_dir  # Use same cache directory
windrose_cache_path = os.path.join(windrose_cache_dir, "windrose_power_data.pkl")

if os.path.exists(windrose_cache_path):
    print(f"Loading cached wind rose data: {windrose_cache_path}")
    with open(windrose_cache_path, "rb") as f:
        cache_data = pickle.load(f)
    Res_dict = cache_data["Res_dict"]
    Res_targets_dict = cache_data["Res_targets_dict"]
else:
    print("Computing wind rose data (this may take a while)...")
    # layout selection
    Res_dict = {}
    Res_targets_dict = {}
    for layout_type in ["regular", "irregular"]:
        # Pre-rotate farm by 90° so that combined with per-direction rotation of +wd_i,
        # met wind direction wd_i maps correctly to the GNO's -x wind direction
        graph_x, graph_y = setup_farm_layout(layout_type)
        graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 90)

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

    # Save to cache
    with open(windrose_cache_path, "wb") as f:
        pickle.dump({"Res_dict": Res_dict, "Res_targets_dict": Res_targets_dict}, f)
    print(f"Cached wind rose data: {windrose_cache_path}")

# %% Calculate AEP
# Calculate Annual Energy Production for both layouts using GNO and PyWake
# AEP = sum over wd of: P(wd) * sum over ws of: P(ws|wd) * Power(wd, ws) * 8760 hours

from scipy.stats import weibull_min

print("\nCalculating AEP...")

# Get Weibull parameters per sector
A_vals = np.array(A["data"])  # Weibull A parameter per sector
k_vals = np.array(k["data"])  # Weibull k parameter per sector
freq_vals = np.array(freq["data"])  # Sector probability

# Wind speeds used in the wind rose calculation
wind_speeds_aep = np.array(wind)  # [14, 12, 10, 8, 6] m/s

aep_results = []

for layout_type in ["regular", "irregular"]:
    # Get power data: Res_dict[layout_type][i] is power/740MW at wind[i] for all directions
    # Shape: (n_wind_speeds, n_directions)
    gno_power_normalized = np.array(Res_dict[layout_type])  # GNO predictions
    pywake_power_normalized = np.array(Res_targets_dict[layout_type])  # PyWake targets

    # Convert back to actual power in MW
    gno_power_mw = gno_power_normalized * 740  # MW
    pywake_power_mw = pywake_power_normalized * 740  # MW

    # Calculate AEP by integrating over wind speed probability
    # For each direction, weight power by Weibull PDF and sector probability
    gno_aep_gwh = 0.0
    pywake_aep_gwh = 0.0

    for i_wd, wd_val in enumerate(wd_py):
        # Get Weibull parameters for this sector (interpolate if needed)
        # wd array from wind resource is the sector centers
        sector_idx = int(wd_val / (360 / len(A_vals))) % len(A_vals)
        A_sector = A_vals[sector_idx]
        k_sector = k_vals[sector_idx]
        freq_sector = freq_vals[sector_idx] / len(wd_py) * len(A_vals)  # Normalize for 1-deg resolution

        # Calculate Weibull PDF at each wind speed
        # scipy weibull_min: pdf(x, c, scale) where c=k and scale=A
        ws_pdf = weibull_min.pdf(wind_speeds_aep, k_sector, scale=A_sector)

        # Normalize PDF to sum to 1 over the discrete wind speeds (trapezoidal approximation)
        ws_weights = ws_pdf / np.sum(ws_pdf) if np.sum(ws_pdf) > 0 else ws_pdf

        # Weighted sum of power over wind speeds
        gno_power_wd = np.sum(gno_power_mw[:, i_wd] * ws_weights)
        pywake_power_wd = np.sum(pywake_power_mw[:, i_wd] * ws_weights)

        # Add to AEP (weighted by sector probability)
        gno_aep_gwh += gno_power_wd * freq_sector * 8760 / 1000  # GWh
        pywake_aep_gwh += pywake_power_wd * freq_sector * 8760 / 1000  # GWh

    aep_results.append({
        "layout": layout_type,
        "GNO_AEP_GWh": gno_aep_gwh,
        "PyWake_AEP_GWh": pywake_aep_gwh,
        "AEP_error_percent": (gno_aep_gwh - pywake_aep_gwh) / pywake_aep_gwh * 100,
    })

    print(f"{layout_type.capitalize()} layout:")
    print(f"  GNO AEP:            {gno_aep_gwh:.2f} GWh")
    print(f"  PyWake AEP:         {pywake_aep_gwh:.2f} GWh")
    print(f"  GNO vs PyWake error: {aep_results[-1]['AEP_error_percent']:.2f}%")

# Save AEP results to CSV
import pandas as pd

aep_df = pd.DataFrame(aep_results)
aep_csv_path = os.path.join(flow_field_output_dir, "IEA740_AEP.csv")
aep_df.to_csv(aep_csv_path, index=False, float_format="%.2f")
print(f"\nAEP results saved to: {aep_csv_path}")

# %% Plot content
# Two-row layout for better readability in printed A4 version
# Each polar plot gets full column width
fig, axes = plt.subplots(
    2,
    1,
    figsize=(3.0, 3.9),  # Reduced by 25% area (scaled by sqrt(0.75))
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
            linewidth=1.5,  # Increased for A4 readability
        )
        ax.plot(
            np.deg2rad(np.arange(0, 360, 1)),
            Res_targets[i],
            color=colors[i],
            linestyle="dotted",
            linewidth=1.5,  # Increased for A4 readability
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
    linewidth=1.5,
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
    linewidth=1.5,
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
        linewidth=1.5,
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
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    fontsize=7,  # Smaller font to match reduced figure size
    frameon=True,
    handlelength=1.2,
    columnspacing=0.8,
    handletextpad=0.4,
    borderpad=0.3,
)
# Apply tight_layout first, then add labels using figure coordinates
fig.tight_layout(rect=[0.26, 0, 1, 0.92])  # Adjust margins for legend and labels

# add subplot letter (a) and (b) with AEP - positioned at left edge using figure coords
subplot_labels = ["(a) Regular", "(b) Irregular"]
layout_names = ["regular", "irregular"]
# Y positions in figure coordinates (top plot ~0.85, bottom plot ~0.42)
y_positions_label = [0.88, 0.45]
y_positions_aep = [0.82, 0.39]

for label, layout_name, y_label, y_aep in zip(
    subplot_labels, layout_names, y_positions_label, y_positions_aep
):
    aep_data = next(r for r in aep_results if r["layout"] == layout_name)
    # Subplot label at left edge of figure
    fig.text(
        0.12,
        y_label,
        label,
        fontsize=9,
        ha="left",
        va="center",
    )
    # AEP values beneath the label
    aep_text = (
        f"AEP [GWh]\n"
        f"GNO: {aep_data['GNO_AEP_GWh']:.0f}\n"
        f"PyWake: {aep_data['PyWake_AEP_GWh']:.0f}"
    )
    fig.text(
        0.12,
        y_aep,
        aep_text,
        fontsize=7,
        ha="left",
        va="top",
        linespacing=1.2,
    )
# Save to same directory as flow field plots
plt.savefig(
    os.path.join(flow_field_output_dir, "IEA740_power_rose.pdf"),
    bbox_inches="tight",
    dpi=300,
)


# %% Plot layouts
# Layout is shown in original UTM orientation where +y is North

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

    # Add north arrow in top-left corner using matplotlib-map-utils
    add_north_arrow(ax, location="upper left", scale=0.25)

    # add subplot letter (a) and (b) - positioned at top center
    ax.text(
        0.5,
        1.02,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )

# fig.tight_layout()
# Save to same directory as flow field plots
plt.savefig(
    os.path.join(flow_field_output_dir, "IEA740_layouts.pdf"),
    bbox_inches="tight",
    dpi=300,
)


# %% Flow Field Visualization
# ============================================================================
# Generate 2D flow field comparison figures for GNO vs PyWake
# One figure per wind speed showing both layouts
# ============================================================================

wind_speeds_flow = [6, 12, 18]  # m/s
TI_flow = 0.05

print("\n" + "=" * 60)
print("Flow Field Visualization")
print(f"Cache directory: {flow_field_cache_dir}")
print("=" * 60)

for U_flow in wind_speeds_flow:
    print(f"\nGenerating flow field figure for U = {U_flow} m/s...")

    fig = plot_flow_field_comparison(
        U_flow=U_flow,
        TI_flow=TI_flow,
        grid_density=3,
        save_path=os.path.join(flow_field_output_dir, f"IEA740_flow_field_U{U_flow}.pdf"),
        cache_dir=flow_field_cache_dir,
        force_recompute=False,  # Set to True to regenerate cached data
    )

    # Also save PNG version
    fig.savefig(
        os.path.join(flow_field_output_dir, f"IEA740_flow_field_U{U_flow}.png"),
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)

print(f"\nFlow field figures saved to: {flow_field_output_dir}")
print(f"Cached data saved to: {flow_field_cache_dir}")
print("=" * 60)

# Generate stacked figure with all wind speeds and layouts
print("\nGenerating stacked flow field figure...")
fig_stacked = plot_flow_field_stacked(
    wind_speeds=wind_speeds_flow,
    TI_flow=TI_flow,
    grid_density=3,
    save_path=os.path.join(flow_field_output_dir, "IEA740_flow_field_stacked.pdf"),
    cache_dir=flow_field_cache_dir,
    force_recompute=False,
)
fig_stacked.savefig(
    os.path.join(flow_field_output_dir, "IEA740_flow_field_stacked.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close(fig_stacked)
print("Stacked figure saved.")
print("=" * 60)


# %%
