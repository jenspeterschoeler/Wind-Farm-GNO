"""
Article 2 publication figure generation (local machine).

Loads cached graph data and generates model predictions with publication figures.
Supports two evaluation paths:

1. **L2_global** (Phase 1, TurbOPark): PyWake on-the-fly ground truth
   - Cache from: generate_plot_data.py (run on Sophia)
2. **L1_L3_phase2** (Phase 2, AWF RANS): NetCDF ground truth
   - Cache from: generate_awf_plot_data.py (run on Sophia)

Usage:
    # Phase 1 model (TurbOPark)
    python Experiments/article_2/plot_publication_figures.py --model L2_global

    # Phase 2 model (AWF RANS)
    python Experiments/article_2/plot_publication_figures.py --model L1_L3_phase2

    # Force recompute predictions
    python Experiments/article_2/plot_publication_figures.py --force-predictions

Outputs (organized by run ID parsed from checkpoint path):
    Experiments/article_2/outputs/<run_id>/
    |-- crossstream_profiles_best_model.pdf
    |-- crossstream_profiles_errors.csv
    |-- crossstream_errors_summary.csv
    |-- deficit_threshold_sweep.csv
    |-- deficit_threshold_sweep.pdf
    |-- wt_spatial_errors_all_layouts_best_model.pdf
    |-- wt_spatial_errors_per_windspeed.pdf
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from py_wake.examples.data.dtu10mw import DTU10MW
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from article2_plotting import (  # noqa: E402
    export_crossstream_errors_csv,
    export_deficit_threshold_sweep,
    plot_crossstream_figure,
    plot_wt_spatial_errors_aggregated,
    plot_wt_spatial_errors_per_windspeed,
)
from article2_utils import (  # noqa: E402
    AWF_TURBINE_DIAMETER,
    MODELS,
    aggregate_wt_errors_across_windspeeds,
    apply_mask,
    apply_normalizations,
    compute_crossstream_errors,
    compute_wt_relative_errors,
    construct_awf_probe_graph,
    create_turbopark_wf_model,
    dict_to_graph,
    load_model_for_prediction,
    predict_wt_errors,
    summarize_errors,
)

from utils.data_tools import setup_unscaler  # noqa: E402

SCRIPT_DIR = Path(__file__).parent


def generate_all_crossstream_data(
    raw_data: dict,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    unscaler,
    wf_model,
    x_downstream: list[int],
    U_free: list[float],
    TI_flow: float = 0.05,
    grid_density: int = 3,
) -> dict:
    """Generate crossstream predictions locally from raw graph data.

    This is the Article 2 version: uses TurbOPark wf_model for ground truth.
    """
    from jax import numpy as jnp
    from py_wake import HorizontalGrid

    from utils.run_pywake import construct_on_the_fly_probe_graph

    wt = DTU10MW()
    D = wt.diameter()
    plot_distance = raw_data["metadata"]["plot_distance"]

    # Calculate y probe positions
    y_plot_range = plot_distance * 1.0 * scale_stats["distance"]["range"]
    n_y_points = int((y_plot_range / D) * grid_density)
    y = np.linspace(-y_plot_range / 2, y_plot_range / 2, n_y_points)
    unscaled_rel_plot_distance = plot_distance * scale_stats["distance"]["range"] / D

    result = {
        "metadata": {
            "x_downstream": x_downstream,
            "U_free": U_free,
            "TI_flow": TI_flow,
            "grid_density": grid_density,
            "model_type_str": "best_model",
        },
        "global_limits": {
            "velocity_range": None,
            "y_plot_range_D": y_plot_range / D,
            "unscaled_rel_plot_distance": unscaled_rel_plot_distance,
        },
        "layout_data": {},
    }

    global_min = None
    global_max = None

    for layout_name, layout_data in tqdm(
        raw_data["layout_data"].items(), desc="Generating crossstream predictions"
    ):
        all_positions = layout_data["wt_positions"]
        wt_mask = layout_data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]
        wt_positions = all_positions[wt_idx] * scale_stats["distance"]["range"]
        wt_spacing = layout_data["wt_spacing"]

        layout_result = {
            "node_positions_D": None,
            "wt_mask": wt_mask,
            "wt_spacing": wt_spacing,
            "predictions": {},
        }

        graph_processed = False
        for x_sel in x_downstream:
            layout_result["predictions"][x_sel] = {}

            for U_flow in U_free:
                grid = HorizontalGrid(
                    x=[x_sel * D + wt_positions[:, 0].max()],
                    y=y,
                    h=wt.hub_height(),
                )

                jraph_graph_gen, jraph_probe_gen, node_tuple_gen = construct_on_the_fly_probe_graph(
                    positions=wt_positions,
                    U=[U_flow],
                    TI=[TI_flow],
                    grid=grid,
                    scale_stats=scale_stats,
                    return_positions=True,
                    wf_model=wf_model,
                )
                targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_tuple_gen

                prediction = pred_fn(
                    jraph_graph_gen,
                    jraph_probe_gen,
                    jnp.atleast_2d(wt_mask_gen).T,
                    jnp.atleast_2d(probe_mask_gen).T,
                ).squeeze()

                unscaled_predictions = inverse_scale_target(apply_mask(prediction, probe_mask_gen))
                unscaled_targets = inverse_scale_target(
                    apply_mask(targets_gen.squeeze(), probe_mask_gen)
                )

                # Compute errors
                u = np.array(unscaled_targets)
                u_hat = np.array(unscaled_predictions)
                errors = compute_crossstream_errors(u, u_hat, U_flow)
                y_D = y / D
                error_stats = summarize_errors(errors, y_D)

                # Normalized profiles
                norm_preds, norm_tgts, _ = apply_normalizations(
                    np.array(unscaled_predictions),
                    np.array(unscaled_targets),
                    U_flow,
                    plot_velocity_deficit=True,
                    normalize_by_U=True,
                )

                current_min = min(norm_preds.min(), norm_tgts.min())
                current_max = max(norm_preds.max(), norm_tgts.max())
                if global_min is None or current_min < global_min:
                    global_min = current_min
                if global_max is None or current_max > global_max:
                    global_max = current_max

                if not graph_processed:
                    unscaled_node_positions = (
                        unscaler.inverse_scale_trunk_input(node_positions_gen) / D
                    )
                    unscaled_graphs_gen = unscaler.inverse_scale_graph(jraph_graph_gen)
                    unscaled_probe_graphs_gen = unscaler.inverse_scale_graph(jraph_probe_gen)
                    unscaled_graphs_gen = unscaled_graphs_gen._replace(
                        edges=unscaled_graphs_gen.edges / D
                    )
                    unscaled_probe_graphs_gen = unscaled_probe_graphs_gen._replace(
                        edges=unscaled_probe_graphs_gen.edges / D
                    )
                    layout_result["node_positions_D"] = np.array(unscaled_node_positions)
                    graph_processed = True

                layout_result["predictions"][x_sel][U_flow] = {
                    "unscaled_predictions": np.array(unscaled_predictions),
                    "unscaled_targets": np.array(unscaled_targets),
                    "normalized_predictions": norm_preds,
                    "normalized_targets": norm_tgts,
                    "y/D": y_D,
                    "unscaled_node_positions": np.array(unscaled_node_positions),
                    "unscaled_graphs": unscaled_graphs_gen,
                    "unscaled_probe_graphs": unscaled_probe_graphs_gen,
                    "errors": error_stats,
                }

        result["layout_data"][layout_name] = layout_result

    pad_size = (global_max - global_min) * 0.1
    result["global_limits"]["velocity_range"] = [
        global_min - pad_size,
        global_max + pad_size,
    ]

    return result


def generate_all_wt_error_data(
    raw_data: dict,
    pred_fn,
    model,
    model_params,
    inverse_scale_target,
    scale_stats: dict,
    unscaler,
    wf_model,
    U_free_values: list[float],
    TI_flow: float = 0.05,
) -> dict:
    """Generate WT spatial error data for all layouts.

    Uses PyWake regeneration at each wind speed with TurbOPark model.
    """
    wt = DTU10MW()
    D = wt.diameter()

    result = {
        "metadata": {
            "U_free_values": U_free_values,
            "model_type_str": "best_model",
        },
        "aggregated_extremas": {},
        "layout_data": {},
    }

    agg_stats = {
        "mean": {"min": 1e9, "max": -1e9},
        "std": {"min": 1e9, "max": -1e9},
        "abs_max": {"min": 1e9, "max": -1e9},
    }

    for layout_name, layout_data in tqdm(
        raw_data["layout_data"].items(), desc="Calculating WT errors"
    ):
        node_positions = layout_data["node_positions"]
        wt_mask = layout_data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]

        wt_positions_m = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
        wt_positions_D = wt_positions_m / D

        wt_rel_errors_list = []
        for U_flow in U_free_values:
            wt_result = predict_wt_errors(
                pred_fn=pred_fn,
                model_params=model_params,
                model=model,
                inverse_scale_target=inverse_scale_target,
                scale_stats=scale_stats,
                wt_positions_m=wt_positions_m,
                U_flow=U_flow,
                TI_flow=TI_flow,
                wf_model=wf_model,
            )
            wt_rel_errors_list.append(wt_result["wt_rel_errors"])

        per_ws_errors = dict(zip(U_free_values, wt_rel_errors_list))
        aggregated = aggregate_wt_errors_across_windspeeds(per_ws_errors)

        for metric in ["mean", "std", "abs_max"]:
            agg_stats[metric]["min"] = min(agg_stats[metric]["min"], aggregated[metric].min())
            agg_stats[metric]["max"] = max(agg_stats[metric]["max"], aggregated[metric].max())

        result["layout_data"][layout_name] = {
            "wt_positions_D": np.array(wt_positions_D),
            "aggregated_errors": aggregated,
            "per_wind_speed_errors": per_ws_errors,
            "n_wts": int(np.sum(wt_mask)),
        }

    result["aggregated_extremas"] = {
        metric: (agg_stats[metric]["min"], agg_stats[metric]["max"])
        for metric in ["mean", "std", "abs_max"]
    }

    return result


def prepare_per_windspeed_plot_data(
    wt_errors_data: dict,
    U_free_values: list[float] | None = None,
) -> dict:
    """Prepare pre-computed data dict for plot_wt_spatial_errors_per_windspeed.

    Converts the wt_errors_data into the format expected by the plotting function.
    Wind speeds are indexed per-layout (each layout can have its own wind speeds).
    """
    padding = 8

    layout_errors = {}
    layout_positions = {}
    layout_bounds = {}
    layout_U_values = {}
    all_errors = []

    for layout_name, data in wt_errors_data["layout_data"].items():
        wt_positions_D = data["wt_positions_D"]
        layout_positions[layout_name] = wt_positions_D

        x_min = wt_positions_D[:, 0].min() - padding
        x_max = wt_positions_D[:, 0].max() + padding
        y_min = wt_positions_D[:, 1].min() - padding
        y_max = wt_positions_D[:, 1].max() + padding
        layout_bounds[layout_name] = {
            "x_lim": (x_min, x_max),
            "y_lim": (y_min, y_max),
            "x_range": x_max - x_min,
            "y_range": y_max - y_min,
        }

        # Use per-layout wind speeds from the actual data
        layout_U = sorted(data["per_wind_speed_errors"].keys())
        layout_U_values[layout_name] = layout_U

        layout_errors[layout_name] = {}
        for U in layout_U:
            wt_rel_errors = np.abs(data["per_wind_speed_errors"][U])
            layout_errors[layout_name][U] = wt_rel_errors
            all_errors.extend(wt_rel_errors.flatten())

    # Max number of wind speeds across all layouts (for grid rows)
    max_n_U = max(len(v) for v in layout_U_values.values()) if layout_U_values else 0

    return {
        "layout_errors": layout_errors,
        "layout_positions": layout_positions,
        "layout_bounds": layout_bounds,
        "layout_U_values": layout_U_values,
        "global_extrema": np.max(all_errors) if all_errors else 0,
        "n_wind_speed_rows": max_n_U,
    }


def generate_awf_crossstream_data(
    raw_data: dict,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    unscaler,
    x_downstream: list[int],
) -> dict:
    """Generate crossstream predictions for AWF model using RANS ground truth.

    For each layout and flowcase, extracts crossstream profiles from the cached
    NetCDF data and generates GNO predictions at the same positions.

    Returns data in the same format as generate_all_crossstream_data() so
    existing plotting functions work unchanged.
    """
    from jax import numpy as jnp

    D = AWF_TURBINE_DIAMETER

    result = {
        "metadata": {
            "x_downstream": x_downstream,
            "U_free": [],  # Will be populated from actual flowcase ws_inf values
            "model_type_str": "best_model",
        },
        "global_limits": {
            "velocity_range": None,
        },
        "layout_data": {},
    }

    global_min = None
    global_max = None

    # Collect all unique wind speeds across flowcases
    all_U_values = set()

    for layout_name, layout_data in tqdm(
        raw_data["layout_data"].items(), desc="Generating AWF crossstream predictions"
    ):
        raw_flow_fields = layout_data["raw_flow_fields"]
        crossstream_profiles = layout_data["crossstream_profiles"]

        # Get WT positions from the first flowcase
        first_ff = raw_flow_fields[0]
        wt_positions_m = np.c_[first_ff["wt_x_D"] * D, first_ff["wt_y_D"] * D]

        # Use per-flowcase wind speeds as our "U_free" equivalent
        U_values = sorted({ff["ws_inf"] for ff in raw_flow_fields.values()})
        all_U_values.update(U_values)

        layout_result = {
            "node_positions_D": None,
            "wt_mask": layout_data["wt_mask"],
            "wt_spacing": 0.0,  # AWF has no wt_spacing metadata
            "predictions": {},
        }

        graph_processed = False

        for x_sel in x_downstream:
            layout_result["predictions"][x_sel] = {}

            for fc_idx, flow_field in raw_flow_fields.items():
                U_flow = flow_field["ws_inf"]

                # Get RANS ground truth from cached crossstream profile
                cs_profile = crossstream_profiles[fc_idx].get(x_sel)
                if cs_profile is None:
                    continue

                y_D = cs_profile["y_D"]
                rans_U = cs_profile["U"]

                # Build probe positions along crossstream line
                probe_x_m = np.full_like(y_D, cs_profile["actual_x_D"] * D)
                probe_y_m = y_D * D
                probe_positions_m = np.c_[probe_x_m, probe_y_m]

                # Build AWF probe graph and predict
                jraph_graph, jraph_probe, node_tuple = construct_awf_probe_graph(
                    wt_positions_m=wt_positions_m,
                    wseff=flow_field["wseff"],
                    ws_inf=U_flow,
                    ti_inf=flow_field["ti_inf"],
                    probe_positions_m=probe_positions_m,
                    probe_velocities=rans_U,  # RANS ground truth as dummy targets
                    scale_stats=scale_stats,
                    return_positions=True,
                )
                targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_tuple

                prediction = pred_fn(
                    jraph_graph,
                    jraph_probe,
                    jnp.atleast_2d(wt_mask_gen).T,
                    jnp.atleast_2d(probe_mask_gen).T,
                ).squeeze()

                unscaled_predictions = inverse_scale_target(apply_mask(prediction, probe_mask_gen))
                # Use RANS ground truth directly (already in m/s)
                unscaled_targets = np.array(rans_U)

                # Compute errors
                u = np.array(unscaled_targets)
                u_hat = np.array(unscaled_predictions)
                errors = compute_crossstream_errors(u, u_hat, U_flow)
                error_stats = summarize_errors(errors, y_D)

                # Normalized profiles
                norm_preds, norm_tgts, _ = apply_normalizations(
                    u_hat,
                    u,
                    U_flow,
                    plot_velocity_deficit=True,
                    normalize_by_U=True,
                )

                current_min = min(norm_preds.min(), norm_tgts.min())
                current_max = max(norm_preds.max(), norm_tgts.max())
                if global_min is None or current_min < global_min:
                    global_min = current_min
                if global_max is None or current_max > global_max:
                    global_max = current_max

                if not graph_processed:
                    unscaled_node_positions = (
                        unscaler.inverse_scale_trunk_input(node_positions_gen) / D
                    )
                    unscaled_graphs_gen = unscaler.inverse_scale_graph(jraph_graph)
                    unscaled_probe_graphs_gen = unscaler.inverse_scale_graph(jraph_probe)
                    unscaled_graphs_gen = unscaled_graphs_gen._replace(
                        edges=unscaled_graphs_gen.edges / D
                    )
                    unscaled_probe_graphs_gen = unscaled_probe_graphs_gen._replace(
                        edges=unscaled_probe_graphs_gen.edges / D
                    )
                    layout_result["node_positions_D"] = np.array(unscaled_node_positions)
                    graph_processed = True

                layout_result["predictions"][x_sel][U_flow] = {
                    "unscaled_predictions": u_hat,
                    "unscaled_targets": u,
                    "normalized_predictions": norm_preds,
                    "normalized_targets": norm_tgts,
                    "y/D": y_D,
                    "unscaled_node_positions": np.array(unscaled_node_positions),
                    "unscaled_graphs": unscaled_graphs_gen,
                    "unscaled_probe_graphs": unscaled_probe_graphs_gen,
                    "errors": error_stats,
                    "ti_inf": flow_field["ti_inf"],
                }

        result["layout_data"][layout_name] = layout_result

    result["metadata"]["U_free"] = sorted(all_U_values)

    if global_min is not None and global_max is not None:
        pad_size = (global_max - global_min) * 0.1
        result["global_limits"]["velocity_range"] = [
            global_min - pad_size,
            global_max + pad_size,
        ]

    return result


def generate_awf_wt_error_data(
    raw_data: dict,
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    unscaler,
) -> dict:
    """Generate WT spatial error data for AWF model using cached graph data.

    Instead of PyWake regeneration, uses the per-flowcase cached graphs directly.
    Each flowcase has its own ws_inf, and we aggregate across flowcases.

    Returns data in the same format as generate_all_wt_error_data() so
    existing plotting functions work unchanged.
    """
    from jax import numpy as jnp

    D = AWF_TURBINE_DIAMETER

    result = {
        "metadata": {
            "U_free_values": [],
            "model_type_str": "best_model",
        },
        "aggregated_extremas": {},
        "layout_data": {},
    }

    agg_stats = {
        "mean": {"min": 1e9, "max": -1e9},
        "std": {"min": 1e9, "max": -1e9},
        "abs_max": {"min": 1e9, "max": -1e9},
    }

    all_U_values = set()

    for layout_name, layout_data in tqdm(
        raw_data["layout_data"].items(), desc="Calculating AWF WT errors"
    ):
        per_flowcase = layout_data.get("per_flowcase_graphs", {})
        raw_flow_fields = layout_data["raw_flow_fields"]

        if not per_flowcase:
            print(f"  Warning: No per-flowcase data for {layout_name}, skipping")
            continue

        # Get WT positions in D from any flowcase
        node_positions = layout_data["node_positions"]
        wt_mask = layout_data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]
        wt_positions_m = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
        wt_positions_D = wt_positions_m / D

        per_ws_errors = {}
        for fc_offset, fc_data in per_flowcase.items():
            flow_field = raw_flow_fields.get(fc_offset)
            if flow_field is None:
                continue

            U_flow = flow_field["ws_inf"]
            all_U_values.add(U_flow)

            # Reconstruct jraph graphs from cached data
            graphs = dict_to_graph(fc_data["graphs"])
            probe_graphs = dict_to_graph(fc_data["probe_graphs"])
            targets = fc_data["targets"]
            fc_wt_mask = fc_data["wt_mask"]
            fc_probe_mask = fc_data["probe_mask"]

            # Run prediction — masks from cache are already (N, 1)
            wt_mask_arr = jnp.array(fc_wt_mask)
            probe_mask_arr = jnp.array(fc_probe_mask)
            if wt_mask_arr.ndim == 1:
                wt_mask_arr = wt_mask_arr[:, None]
                probe_mask_arr = probe_mask_arr[:, None]
            prediction = pred_fn(
                graphs,
                probe_graphs,
                wt_mask_arr,
                probe_mask_arr,
            ).squeeze()

            # Extract WT predictions
            wt_idx_fc = np.where(fc_wt_mask != 0)[0]
            wt_predictions = np.array(inverse_scale_target(prediction[wt_idx_fc]).squeeze())
            wt_targets = np.array(
                inverse_scale_target(np.array(targets).squeeze()[wt_idx_fc]).squeeze()
            )

            wt_rel_errors = compute_wt_relative_errors(wt_targets, wt_predictions, U_flow)
            per_ws_errors[U_flow] = wt_rel_errors

        if not per_ws_errors:
            continue

        aggregated = aggregate_wt_errors_across_windspeeds(per_ws_errors)

        for metric in ["mean", "std", "abs_max"]:
            agg_stats[metric]["min"] = min(agg_stats[metric]["min"], aggregated[metric].min())
            agg_stats[metric]["max"] = max(agg_stats[metric]["max"], aggregated[metric].max())

        result["layout_data"][layout_name] = {
            "wt_positions_D": np.array(wt_positions_D),
            "aggregated_errors": aggregated,
            "per_wind_speed_errors": per_ws_errors,
            "n_wts": int(np.sum(wt_mask)),
        }

    result["metadata"]["U_free_values"] = sorted(all_U_values)
    result["aggregated_extremas"] = {
        metric: (agg_stats[metric]["min"], agg_stats[metric]["max"])
        for metric in ["mean", "std", "abs_max"]
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate Article 2 publication figures from cached data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="L2_global",
        choices=list(MODELS.keys()),
        help="Model to use (default: L2_global)",
    )
    parser.add_argument(
        "--force-predictions",
        action="store_true",
        help="Force recompute predictions (ignore prediction cache)",
    )
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    is_awf = model_cfg.dataset_name == "awf_graphs"
    cache_dir = SCRIPT_DIR / "cache" / "publication_figures" / model_cfg.dataset_name
    outputs_dir = SCRIPT_DIR / "outputs" / model_cfg.run_id
    outputs_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = cache_dir / "raw_graph_data.pkl"

    if not raw_cache.exists():
        print("Raw graph data cache not found!")
        print(f"Expected: {raw_cache}")
        if is_awf:
            print("\nPlease run generate_awf_plot_data.py on Sophia first, then copy cache file.")
        else:
            print("\nPlease run generate_plot_data.py on Sophia first, then copy cache file.")
        return

    # Load raw cached data
    print("Loading cached raw graph data...")
    with open(raw_cache, "rb") as f:
        raw_data = pickle.load(f)

    # Load model
    print("\nLoading model...")
    scale_stats = raw_data["scale_stats"]
    pred_fn, inverse_scale_target, scale_stats, model, model_params = load_model_for_prediction(
        model_cfg, scale_stats=scale_stats
    )

    # Setup unscaler
    from omegaconf import DictConfig

    model_cfg_path = model_cfg.portable_path / "model_config.json"
    with open(model_cfg_path) as f:
        restored_cfg_model = DictConfig(json.load(f))
    unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)

    # Parameters
    x_downstream = [50, 100, 200, 300]

    if is_awf:
        # Phase 2 model: AWF RANS ground truth (no PyWake needed)
        print("\n" + "=" * 60)
        print("AWF mode: Using RANS ground truth from NetCDF cache")
        print("=" * 60)

        # --- Crossstream Predictions ---
        print("\nGenerating AWF crossstream predictions...")
        crossstream_data = generate_awf_crossstream_data(
            raw_data,
            pred_fn,
            inverse_scale_target,
            scale_stats,
            unscaler,
            x_downstream=x_downstream,
        )

        # --- WT Spatial Errors ---
        print("\nGenerating AWF WT spatial error data...")
        wt_errors_data = generate_awf_wt_error_data(
            raw_data,
            pred_fn,
            inverse_scale_target,
            scale_stats,
            unscaler,
        )

    else:
        # Phase 1 model: TurbOPark PyWake ground truth
        print("\nCreating TurbOPark PyWake model...")
        wf_model = create_turbopark_wf_model()
        print("  Config: TurboGaussianDeficit + SquaredSum + PropagateDownwind")

        U_free = [6.0, 12.0, 18.0]
        TI_flow = 0.05

        # --- Crossstream Predictions ---
        print("\n" + "=" * 60)
        print("Generating crossstream predictions (with TurbOPark ground truth)...")
        print("=" * 60)
        crossstream_data = generate_all_crossstream_data(
            raw_data,
            pred_fn,
            inverse_scale_target,
            scale_stats,
            unscaler,
            wf_model,
            x_downstream=x_downstream,
            U_free=U_free,
            TI_flow=TI_flow,
        )

        # --- WT Spatial Errors ---
        print("\n" + "=" * 60)
        print("Generating WT spatial error data (with TurbOPark ground truth)...")
        print("=" * 60)
        wt_errors_data = generate_all_wt_error_data(
            raw_data,
            pred_fn,
            model,
            model_params,
            inverse_scale_target,
            scale_stats,
            unscaler,
            wf_model,
            U_free_values=U_free,
            TI_flow=TI_flow,
        )

    # --- Plotting (same for both paths) ---
    print("\nGenerating crossstream profiles figure...")
    plot_crossstream_figure(crossstream_data, str(outputs_dir), "best_model", per_row_legend=is_awf)

    print("\nExporting crossstream error statistics...")
    export_crossstream_errors_csv(crossstream_data, str(outputs_dir))

    print("\nGenerating deficit threshold sweep...")
    export_deficit_threshold_sweep(crossstream_data, str(outputs_dir))

    print("\nGenerating WT spatial errors figure (aggregated)...")
    plot_wt_spatial_errors_aggregated(wt_errors_data, str(outputs_dir), "best_model")

    print("\nGenerating per-wind-speed WT spatial errors figure...")
    per_ws_plot_data = prepare_per_windspeed_plot_data(wt_errors_data)
    plot_wt_spatial_errors_per_windspeed(per_ws_plot_data, str(outputs_dir))

    print("\n" + "=" * 60)
    print("Done! Outputs saved to:")
    print(f"  Outputs: {outputs_dir}")
    print(f"  Cache:   {cache_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
