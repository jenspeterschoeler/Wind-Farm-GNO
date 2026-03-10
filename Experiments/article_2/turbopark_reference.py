"""
TurbOPark Reference Flow Field Comparison for Article 2.

This script generates flow field comparison figures between the GNO model (S2 from Phase 1)
and PyWake TurbOPark ground truth for the IEA740 reference layouts.

Key differences from Article 1 (IEA740_reference.py):
- PyWake Model: TurboGaussianDeficit + SquaredSum (vs NiayifarGaussian + LinearSum)
- No blockage model
- No turbulence model
- PropagateDownwind (vs All2AllIterative)
- Model: S2 from Phase 1 architecture search (trained on TurbOPark dataset)

Usage:
    pixi run python Experiments/article_2/turbopark_reference.py

    # Force recompute cached flow fields
    pixi run python Experiments/article_2/turbopark_reference.py --force

    # Use Vj8 model for testing (NOT recommended for production)
    pixi run python Experiments/article_2/turbopark_reference.py --model Vj8

Model Loading:
    - S2: Loaded from Phase 1 selection JSON checkpoint path. On first run, exports to
          assets/best_model_S2/ for faster subsequent loads.
    - Vj8: Pre-exported model in assets/best_model_Vj8/ (for testing only)

Outputs:
    Experiments/article_2/figures/
    ├── turbopark_flow_field_U6.pdf
    ├── turbopark_flow_field_U12.pdf
    ├── turbopark_flow_field_U18.pdf
    └── turbopark_flow_field_stacked.pdf
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cmcrameri.cm as cmc
import jax
import numpy as np
import yaml
from jax import numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from py_wake import HorizontalGrid
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site._site import UniformSite
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models import PropagateDownwind
from tqdm import tqdm

# Project root setup
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.data_tools import setup_unscaler  # noqa: E402
from utils.plotting import matplotlib_set_rcparams  # noqa: E402
from utils.run_pywake import construct_on_the_fly_probe_graph  # noqa: E402
from utils.weight_converter import load_portable_model  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

matplotlib_set_rcparams("paper")

BASE_DIR = REPO_ROOT
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
CACHE_DIR = SCRIPT_DIR / "cache" / "turbopark_flow_fields"

# Create directories
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "S2": {
        "path": BASE_DIR / "assets" / "best_model_S2",
        "selection_json": SCRIPT_DIR / "outputs" / "phase1_2500layouts_selection.json",
        "description": "Phase 1 best model (TurbOPark trained) - PRIMARY",
    },
    "Vj8": {
        "path": BASE_DIR / "assets" / "best_model_Vj8",
        "selection_json": None,
        "description": "TESTING ONLY - trained on IEA740 dataset, NOT TurbOPark!",
    },
}
DEFAULT_MODEL = "S2"

# IEA740 layouts from Article 1
IEA740_RESOURCES_DIR = BASE_DIR / "Experiments" / "article_1" / "IEA740_resources"

# =============================================================================
# Load IEA740 Layouts
# =============================================================================

with open(IEA740_RESOURCES_DIR / "ROWP_Regular.yaml") as f:
    regular_grid_layout = yaml.safe_load(f)["layouts"]["initial_layout"]["coordinates"]

with open(IEA740_RESOURCES_DIR / "ROWP_Irregular.yaml") as f:
    irregular_grid_layout = yaml.safe_load(f)["layouts"]["initial_layout"]["coordinates"]

# =============================================================================
# PyWake Setup - TurbOPark Configuration
# =============================================================================

wt = DTU10MW()
site = UniformSite()

# TurbOPark configuration:
# - TurboGaussianDeficit: Gaussian wake deficit model used in TurbOPark
# - SquaredSum: Quadratic superposition (vs LinearSum in Article 1)
# - PropagateDownwind: Simple downwind propagation (vs All2AllIterative)
# - No blockage model
# - No turbulence model
wf_model = PropagateDownwind(
    site,
    wt,
    wake_deficitModel=TurboGaussianDeficit(),
    superpositionModel=SquaredSum(),
)


# =============================================================================
# Model Loading Functions
# =============================================================================


def _find_best_mse_checkpoint(experiment_dir: Path) -> Path | None:
    """Find the best_mse checkpoint directory including step subdirectory."""
    checkpoint_parent = experiment_dir / "model" / "checkpoints_best_mse"

    if not checkpoint_parent.exists():
        return None

    # Find step-numbered subdirectories
    step_dirs = []
    for item in checkpoint_parent.iterdir():
        if item.is_dir():
            try:
                int(item.name)
                step_dirs.append(item)
            except ValueError:
                continue

    if not step_dirs:
        return None

    # Return highest step number (most recent)
    step_dirs.sort(key=lambda x: int(x.name), reverse=True)
    return step_dirs[0]


def _export_model_to_portable(checkpoint_path: Path, output_dir: Path, cfg_model) -> Path:
    """
    Export model checkpoint to portable msgpack format.

    Args:
        checkpoint_path: Path to Orbax checkpoint (step directory)
        output_dir: Where to save exported files
        cfg_model: Model configuration DictConfig

    Returns:
        Path to output directory
    """
    import flax.serialization
    from omegaconf import OmegaConf

    from utils.model_tools import load_model

    print(f"Exporting model from checkpoint: {checkpoint_path}")

    # Load model from checkpoint (only need params and cfg_model)
    _, params, _, cfg_model = load_model(checkpoint_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params to msgpack
    params_path = output_dir / "best_params.msgpack"
    bytes_data = flax.serialization.to_bytes(params)
    params_path.write_bytes(bytes_data)
    print(f"  Saved: {params_path}")

    # Save model config
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg_model), f, indent=4)
    print(f"  Saved: {config_path}")

    # Extract and save scale_stats from model config
    cfg_dict = OmegaConf.to_container(cfg_model)
    if "data" in cfg_dict and "scale_stats" in cfg_dict["data"]:
        scale_stats_path = output_dir / "scale_stats.json"
        with open(scale_stats_path, "w") as f:
            json.dump(cfg_dict["data"]["scale_stats"], f, indent=4)
        print(f"  Saved: {scale_stats_path}")

    return output_dir


def _ensure_s2_model_exported() -> Path:
    """
    Ensure S2 model is exported to portable format.

    Loads from selection JSON checkpoint and exports to assets/best_model_S2/.
    If already exported, returns existing path.

    Returns:
        Path to exported model directory
    """
    config = MODEL_CONFIGS["S2"]
    output_path = config["path"]
    selection_json = config["selection_json"]

    # Check if already exported
    if (output_path / "best_params.msgpack").exists():
        print(f"S2 model already exported at: {output_path}")
        return output_path

    # Load selection JSON
    if not selection_json.exists():
        raise FileNotFoundError(
            f"Selection JSON not found: {selection_json}\n"
            "Run 'python Experiments/article_2/phase1_architecture.py' first."
        )

    with open(selection_json) as f:
        selection = json.load(f)

    sophia_path = Path(selection["sophia_remote_path"])

    print(f"Loading S2 model from: {sophia_path}")
    print(f"  Model ID: {selection['model_id']}")
    print(f"  Val MSE: {selection['val_mse']:.8f}")

    if not sophia_path.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {sophia_path}\n"
            "Ensure the path is accessible."
        )

    # Find best_mse checkpoint
    checkpoint_dir = _find_best_mse_checkpoint(sophia_path)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"No best_mse checkpoint found in: {sophia_path}")

    print(f"  Checkpoint: {checkpoint_dir}")

    # Export to portable format
    _export_model_to_portable(checkpoint_dir, output_path, None)

    return output_path


def load_gno_model(model_name: str = DEFAULT_MODEL):
    """
    Load GNO model for flow field predictions.

    For S2: Loads from checkpoint (via selection JSON) and exports to assets/best_model_S2/
            on first run for faster subsequent loads.
    For Vj8: Loads from pre-exported assets/best_model_Vj8/.

    Args:
        model_name: Model identifier ("S2" or "Vj8")

    Returns:
        Tuple of (pred_fn, inverse_scale_target, scale_stats)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    # For S2, ensure model is exported
    if model_name == "S2":
        model_path = _ensure_s2_model_exported()
    else:
        model_path = config["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Description: {config['description']}")

    # Load paths
    model_cfg_path = model_path / "model_config.json"
    params_path = model_path / "best_params.msgpack"
    scale_stats_path = model_path / "scale_stats.json"

    # Load model config
    with open(model_cfg_path) as f:
        nested_dict = json.load(f)
        restored_cfg_model = DictConfig(nested_dict)

    # Load scale_stats
    with open(scale_stats_path) as f:
        scale_stats = json.load(f)

    # Setup unscaler
    unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
    inverse_scale_target = unscaler.inverse_scale_output

    # Initialize model with dummy probe graph
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
    _init_targets, _init_wt_mask, _init_probe_mask = _init_node_tuple
    _init_wt_mask = jnp.atleast_2d(_init_wt_mask).T
    _init_probe_mask = jnp.atleast_2d(_init_probe_mask).T
    _init_node_tuple_reshaped = (_init_targets, _init_wt_mask, _init_probe_mask)

    restored_params, restored_cfg_model, model, _ = load_portable_model(
        str(params_path),
        str(model_cfg_path),
        dataset=None,
        inputs=(_init_jraph_graph, _init_probe_graph, _init_node_tuple_reshaped),
    )
    print("Model initialized successfully.")

    # Create JIT-compiled prediction function
    def model_prediction_fn(
        input_graphs,
        input_probe_graphs,
        input_wt_mask,
        input_probe_mask,
    ) -> jnp.ndarray:
        """Prediction function - assumes graphs are padded."""
        return model.apply(
            restored_params,
            input_graphs,
            input_probe_graphs,
            input_wt_mask,
            input_probe_mask,
        )

    pred_fn = jax.jit(model_prediction_fn)

    return pred_fn, inverse_scale_target, scale_stats


# =============================================================================
# Layout Functions
# =============================================================================


def setup_farm_layout(layout_type: str):
    """
    Get centered farm layout coordinates.

    Args:
        layout_type: "regular" or "irregular"

    Returns:
        Tuple of (graph_x, graph_y) centered coordinates in meters
    """
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


def rotate_graph_layout(graph_x, graph_y, angle_deg: float):
    """
    Rotate layout coordinates by given angle.

    Args:
        graph_x, graph_y: Layout coordinates
        angle_deg: Rotation angle in degrees

    Returns:
        Tuple of (rotated_x, rotated_y)
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotated_x = graph_x * cos_angle - graph_y * sin_angle
    rotated_y = graph_x * sin_angle + graph_y * cos_angle

    return rotated_x, rotated_y


# =============================================================================
# Flow Field Generation Functions
# =============================================================================


def generate_flow_field_grid(
    wt_positions: np.ndarray,
    grid_density: int = 3,
    x_extent: tuple | None = None,
    y_extent: tuple | None = None,
):
    """
    Generate a HorizontalGrid for flow field visualization.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        grid_density: Number of grid points per rotor diameter
        x_extent: Tuple (x_min, x_max) in meters. Defaults to 2D upstream to 15D downstream
        y_extent: Tuple (y_min, y_max) in meters. Defaults to 3D beyond farm boundaries

    Returns:
        Tuple of (grid, xx, yy, x_vals, y_vals)
    """
    D = wt.diameter()

    x_min_wt = np.min(wt_positions[:, 0])
    x_max_wt = np.max(wt_positions[:, 0])
    y_min_wt = np.min(wt_positions[:, 1])
    y_max_wt = np.max(wt_positions[:, 1])

    # Match TurbOPark generation settings: 10D upstream, 100D downstream, 5D lateral
    if x_extent is None:
        x_min = x_min_wt - 10 * D
        x_max = x_max_wt + 100 * D
    else:
        x_min, x_max = x_extent

    if y_extent is None:
        y_min = y_min_wt - 5 * D
        y_max = y_max_wt + 5 * D
    else:
        y_min, y_max = y_extent

    n_x = int((x_max - x_min) / D * grid_density)
    n_y = int((y_max - y_min) / D * grid_density)

    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)

    grid = HorizontalGrid(x=x, y=y, h=wt.hub_height())
    xx, yy = np.meshgrid(x, y)

    return grid, xx, yy, x, y


def compute_unified_grid_extent():
    """
    Compute unified grid extent covering both regular and irregular layouts.

    Returns:
        Tuple of (x_extent, y_extent)
    """
    D = wt.diameter()

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

    # Match TurbOPark generation settings: 10D upstream, 100D downstream, 5D lateral
    x_extent = (x_min_wt - 10 * D, x_max_wt + 100 * D)
    y_extent = (y_min_wt - 5 * D, y_max_wt + 5 * D)

    return x_extent, y_extent


def get_gno_flow_field(
    wt_positions: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    U_flow: float,
    TI_flow: float,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
):
    """
    Get GNO flow field predictions using column-by-column iteration.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        x_vals: 1D array of x coordinates
        y_vals: 1D array of y coordinates
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        pred_fn: JIT-compiled prediction function
        inverse_scale_target: Unscaling function
        scale_stats: Scaling statistics dict

    Returns:
        2D array (n_y, n_x) of velocity predictions in m/s
    """
    predictions_list = []

    for x_sel in tqdm(x_vals, desc="GNO flow field columns", leave=False):
        grid = HorizontalGrid(x=[x_sel], y=y_vals, h=wt.hub_height())

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
        _, wt_mask_gen, probe_mask_gen, _ = node_array_tuple_gen

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
        probe_predictions = inverse_scale_target(probe_predictions)

        predictions_list.append(probe_predictions.reshape(-1, 1))

    return np.concatenate(predictions_list, axis=1)


def get_pywake_flow_field(
    wt_positions: np.ndarray,
    grid: HorizontalGrid,
    U_flow: float,
    TI_flow: float,
):
    """
    Get PyWake flow field simulation using TurbOPark configuration.

    Args:
        wt_positions: Array of (x, y) wind turbine positions in meters
        grid: HorizontalGrid object
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]

    Returns:
        2D array (n_y, n_x) of velocity values in m/s
    """
    x_wt = wt_positions[:, 0]
    y_wt = wt_positions[:, 1]

    # Run PyWake simulation (wind from west = 270 deg)
    farm_sim = wf_model(x_wt, y_wt, wd=270, ws=U_flow, TI=TI_flow)
    flow_map = farm_sim.flow_map(grid=grid, wd=270, ws=U_flow)

    return flow_map.WS_eff.values.squeeze()


def get_cache_path(U_flow: float, TI_flow: float, layout_type: str, grid_density: int) -> Path:
    """Get cache file path for flow field data."""
    return (
        CACHE_DIR / f"flow_field_unified_U{U_flow}_TI{TI_flow}_{layout_type}_grid{grid_density}.pkl"
    )


def compute_and_cache_flow_field(
    U_flow: float,
    TI_flow: float,
    layout_type: str,
    grid_density: int,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    x_extent: tuple | None = None,
    y_extent: tuple | None = None,
    force_recompute: bool = False,
) -> dict:
    """
    Compute flow field data for a single layout, with caching.

    Returns:
        Dict with flow field data
    """
    cache_path = get_cache_path(U_flow, TI_flow, layout_type, grid_density)

    if not force_recompute and cache_path.exists():
        print(f"Loading cached data: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"\nComputing {layout_type} layout at U={U_flow} m/s...")

    # Setup farm layout
    graph_x, graph_y = setup_farm_layout(layout_type)
    graph_x, graph_y = rotate_graph_layout(graph_x, graph_y, 270)
    wt_positions = np.column_stack([graph_x, graph_y])

    # Generate grid
    grid, xx, yy, x_vals, y_vals = generate_flow_field_grid(
        wt_positions, grid_density=grid_density, x_extent=x_extent, y_extent=y_extent
    )

    # Get GNO predictions
    gno_field = get_gno_flow_field(
        wt_positions, x_vals, y_vals, U_flow, TI_flow, pred_fn, inverse_scale_target, scale_stats
    )

    # Get PyWake ground truth
    pywake_field = get_pywake_flow_field(wt_positions, grid, U_flow, TI_flow)

    # Compute metrics
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
        "U_flow": U_flow,
        "TI_flow": TI_flow,
        "layout_type": layout_type,
        "grid_density": grid_density,
    }

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cached: {cache_path}")

    return data


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_flow_field_comparison(
    U_flow: float,
    TI_flow: float,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    grid_density: int = 3,
    save_path: Path | None = None,
    force_recompute: bool = False,
):
    """
    Create 2x3 flow field comparison figure for both layouts.

    Publication-level formatting with normalized velocity (u/U) and percentage error.

    Args:
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        pred_fn: JIT-compiled prediction function
        inverse_scale_target: Unscaling function
        scale_stats: Scaling statistics dict
        grid_density: Number of grid points per rotor diameter
        save_path: Optional path to save figure
        force_recompute: If True, ignore cache and recompute

    Returns:
        matplotlib figure object
    """
    D = wt.diameter()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)

    # Compute unified grid extent
    x_extent, y_extent = compute_unified_grid_extent()

    # Storage for colorbar ranges (normalized)
    u_norm_min, u_norm_max = np.inf, -np.inf
    error_pct_max = 0

    layout_data = {}

    # Process both layouts
    for layout_type in ["regular", "irregular"]:
        data = compute_and_cache_flow_field(
            U_flow,
            TI_flow,
            layout_type,
            grid_density,
            pred_fn,
            inverse_scale_target,
            scale_stats,
            x_extent=x_extent,
            y_extent=y_extent,
            force_recompute=force_recompute,
        )

        # Compute normalized values for colorbar range
        U = data["U_flow"]
        u_norm_min = min(
            u_norm_min, np.nanmin(data["gno_field"] / U), np.nanmin(data["pywake_field"] / U)
        )
        u_norm_max = max(
            u_norm_max, np.nanmax(data["gno_field"] / U), np.nanmax(data["pywake_field"] / U)
        )
        error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100
        error_pct_max = max(error_pct_max, np.nanpercentile(error_pct, 99))

        layout_data[layout_type] = data

    u_norm_max = max(u_norm_max, u_norm_min + 0.01)
    u_norm_levels = np.linspace(u_norm_min, u_norm_max, 50)
    error_pct_levels = np.linspace(0, error_pct_max, 50)

    row_labels = ["Regular", "Irregular"]
    col_labels = ["GNO", "TurbOPark", r"$|(u - u')/U|$"]
    subplot_letters = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    contour_handles = {}

    for i_row, layout_type in enumerate(["regular", "irregular"]):
        data = layout_data[layout_type]
        U = data["U_flow"]
        xxD = data["xx"] / D
        yyD = data["yy"] / D

        # Normalized GNO flow field (u/U)
        ax = axes[i_row, 0]
        cf_gno = ax.contourf(
            xxD, yyD, data["gno_field"] / U, levels=u_norm_levels, cmap=cmc.vik_r, extend="both"
        )
        contour_handles["u"] = cf_gno

        # Normalized TurbOPark flow field (u/U)
        ax = axes[i_row, 1]
        ax.contourf(
            xxD, yyD, data["pywake_field"] / U, levels=u_norm_levels, cmap=cmc.vik_r, extend="both"
        )

        # Error in percentage: |(u - u')/U| * 100 (normalized by freestream)
        error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100
        ax = axes[i_row, 2]
        cf_err = ax.contourf(
            xxD, yyD, error_pct, levels=error_pct_levels, cmap="Reds", extend="max"
        )
        contour_handles["error"] = cf_err

    # Labels and formatting
    for i_row in range(2):
        for i_col in range(3):
            ax = axes[i_row, i_col]
            ax.set_aspect("equal", "box")

            # Use white text for flow field columns (0, 1), black for error column (2)
            text_color = "white" if i_col < 2 else "black"

            ax.text(
                0.02,
                0.98,
                subplot_letters[i_row][i_col],
                transform=ax.transAxes,
                fontsize=10,
                color=text_color,
                va="top",
                ha="left",
            )

            if i_row == 0:
                ax.set_title(col_labels[i_col], fontweight="bold")
            if i_col == 0:
                ax.set_ylabel(f"{row_labels[i_row]}\n" + r"$y/D$ [-]")
            if i_row == 1:
                ax.set_xlabel(r"$x/D$ [-]")

    # Colorbars - moved down to avoid overlap
    fig.subplots_adjust(bottom=0.18, wspace=0.05, hspace=0.1)

    cbar_ax1 = fig.add_axes([0.125, 0.055, 0.5, 0.02])
    cbar1 = fig.colorbar(contour_handles["u"], cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label(r"$u/U$ [-]")
    u_ticks = np.linspace(u_norm_min, u_norm_max, 5)
    cbar1.set_ticks(u_ticks)
    cbar1.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    cbar_ax2 = fig.add_axes([0.68, 0.055, 0.22, 0.02])
    cbar2 = fig.colorbar(contour_handles["error"], cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label(r"$|(u - u')/U|$ [$\%$]")
    error_ticks = np.linspace(0, error_pct_max, 4)
    cbar2.set_ticks(error_ticks)
    cbar2.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

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
    wind_speeds: list,
    TI_flow: float,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    grid_density: int = 3,
    save_path: Path | None = None,
    force_recompute: bool = False,
):
    """
    Create stacked 6-row flow field figure: 3 regular + 3 irregular layouts.

    Publication-level formatting with normalized velocity (u/U) and percentage error.

    Args:
        wind_speeds: List of wind speeds [m/s] (e.g., [6, 12, 18])
        TI_flow: Turbulence intensity [-]
        pred_fn: JIT-compiled prediction function
        inverse_scale_target: Unscaling function
        scale_stats: Scaling statistics dict
        grid_density: Number of grid points per rotor diameter
        save_path: Optional path to save figure
        force_recompute: If True, ignore cache and recompute

    Returns:
        matplotlib figure object
    """
    D = wt.diameter()
    n_speeds = len(wind_speeds)

    fig, axes = plt.subplots(2 * n_speeds, 3, figsize=(10, 3 * n_speeds), sharex=True, sharey=True)

    x_extent, y_extent = compute_unified_grid_extent()

    # Collect all data
    all_data = {}
    u_norm_min_global, u_norm_max_global = np.inf, -np.inf
    error_pct_max_global = 0

    for U_flow in wind_speeds:
        for layout_type in ["regular", "irregular"]:
            data = compute_and_cache_flow_field(
                U_flow,
                TI_flow,
                layout_type,
                grid_density,
                pred_fn,
                inverse_scale_target,
                scale_stats,
                x_extent=x_extent,
                y_extent=y_extent,
                force_recompute=force_recompute,
            )

            all_data[(U_flow, layout_type)] = data

            # Compute normalized values for colorbar range
            U = data["U_flow"]
            u_norm_min_global = min(
                u_norm_min_global,
                np.nanmin(data["gno_field"] / U),
                np.nanmin(data["pywake_field"] / U),
            )
            u_norm_max_global = max(
                u_norm_max_global,
                np.nanmax(data["gno_field"] / U),
                np.nanmax(data["pywake_field"] / U),
            )
            # Error normalized by freestream velocity, as percentage
            error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100
            error_pct_max_global = max(error_pct_max_global, np.nanpercentile(error_pct, 99))

    u_norm_max_global = max(u_norm_max_global, u_norm_min_global + 0.01)
    u_norm_levels = np.linspace(u_norm_min_global, u_norm_max_global, 50)
    error_pct_levels = np.linspace(0, error_pct_max_global, 50)

    contour_handles = {}
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

            U = data["U_flow"]
            xxD = data["xx"] / D
            yyD = data["yy"] / D

            # Normalized GNO flow field (u/U)
            ax = axes[i_row, 0]
            cf_gno = ax.contourf(
                xxD,
                yyD,
                data["gno_field"] / U,
                levels=u_norm_levels,
                cmap=cmc.vik_r,
                extend="both",
            )
            contour_handles["u"] = cf_gno

            # Normalized TurbOPark flow field (u/U)
            ax = axes[i_row, 1]
            ax.contourf(
                xxD,
                yyD,
                data["pywake_field"] / U,
                levels=u_norm_levels,
                cmap=cmc.vik_r,
                extend="both",
            )

            # Error in percentage: |(u - u')/U| * 100 (normalized by freestream)
            error_pct = np.abs((data["gno_field"] - data["pywake_field"]) / U) * 100
            ax = axes[i_row, 2]
            cf_err = ax.contourf(
                xxD, yyD, error_pct, levels=error_pct_levels, cmap="Reds", extend="max"
            )
            contour_handles["error"] = cf_err

    # Labels and formatting
    col_labels = ["GNO", "TurbOPark", r"$|(u - u')/U|$"]

    for i_row in range(2 * n_speeds):
        layout_idx = i_row // n_speeds
        speed_idx = i_row % n_speeds
        U_flow = wind_speeds[speed_idx]

        for i_col in range(3):
            ax = axes[i_row, i_col]
            ax.set_aspect("equal", "box")

            # Build label text - add inflow velocity only to first column
            if i_col == 0:
                label_text = f"{subplot_letters[i_row][i_col]} - {U_flow} " + r"$\mathrm{m\,s^{-1}}$"
            else:
                label_text = subplot_letters[i_row][i_col]

            # Use white text for flow field columns (0, 1), black for error column (2)
            text_color = "white" if i_col < 2 else "black"

            ax.text(
                0.02,
                0.98,
                label_text,
                transform=ax.transAxes,
                fontsize=9,
                color=text_color,
                va="top",
                ha="left",
            )

            if i_row == 0:
                ax.set_title(col_labels[i_col], fontweight="bold")
            if i_col == 0:
                ax.set_ylabel(r"$y/D$ [-]")
            if i_row == 2 * n_speeds - 1:
                ax.set_xlabel(r"$x/D$ [-]")

    # Add y-axis supertitles for Regular and Irregular
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

    # Colorbars - moved down to avoid overlap
    fig.subplots_adjust(bottom=0.08, wspace=0.05, hspace=0.1, left=0.07)

    cbar_ax1 = fig.add_axes([0.125, 0.025, 0.5, 0.015])
    cbar1 = fig.colorbar(contour_handles["u"], cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label(r"$u/U$ [-]")
    u_ticks = np.linspace(u_norm_min_global, u_norm_max_global, 5)
    cbar1.set_ticks(u_ticks)
    cbar1.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    cbar_ax2 = fig.add_axes([0.68, 0.025, 0.22, 0.015])
    cbar2 = fig.colorbar(contour_handles["error"], cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label(r"$|(u - u')/U|$ [$\%$]")
    error_ticks = np.linspace(0, error_pct_max_global, 4)
    cbar2.set_ticks(error_ticks)
    cbar2.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Main Execution
# =============================================================================


def main(model_name: str = DEFAULT_MODEL, force_recompute: bool = False):
    """
    Main function to generate all flow field comparison figures.

    Args:
        model_name: Model to use ("S2" or "Vj8")
        force_recompute: If True, ignore cache and recompute all flow fields
    """
    print("=" * 60)
    print("TurbOPark Reference Flow Field Comparison")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("PyWake config: TurboGaussianDeficit + SquaredSum + PropagateDownwind")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    # Load model
    pred_fn, inverse_scale_target, scale_stats = load_gno_model(model_name)

    # Configuration
    wind_speeds_flow = [6, 12, 18]  # m/s
    TI_flow = 0.05
    grid_density = 3  # Match TurbOPark generation settings

    # Generate individual wind speed figures
    for U_flow in wind_speeds_flow:
        print(f"\nGenerating flow field figure for U = {U_flow} m/s...")

        fig = plot_flow_field_comparison(
            U_flow=U_flow,
            TI_flow=TI_flow,
            pred_fn=pred_fn,
            inverse_scale_target=inverse_scale_target,
            scale_stats=scale_stats,
            grid_density=grid_density,
            save_path=FIGURES_DIR / f"turbopark_flow_field_U{U_flow}.pdf",
            force_recompute=force_recompute,
        )

        # Also save PNG
        fig.savefig(
            FIGURES_DIR / f"turbopark_flow_field_U{U_flow}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    # Generate stacked figure
    print("\nGenerating stacked flow field figure...")
    fig_stacked = plot_flow_field_stacked(
        wind_speeds=wind_speeds_flow,
        TI_flow=TI_flow,
        pred_fn=pred_fn,
        inverse_scale_target=inverse_scale_target,
        scale_stats=scale_stats,
        grid_density=grid_density,
        save_path=FIGURES_DIR / "turbopark_flow_field_stacked.pdf",
        force_recompute=force_recompute,
    )
    fig_stacked.savefig(
        FIGURES_DIR / "turbopark_flow_field_stacked.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig_stacked)

    print("\n" + "=" * 60)
    print("Flow field generation complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Cached data saved to: {CACHE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TurbOPark reference flow field comparisons for Article 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute cached flow fields",
    )

    args = parser.parse_args()

    main(model_name=args.model, force_recompute=args.force)
