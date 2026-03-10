"""
Master script for Article 1 publication figures.

This script contains:
1. All shared functions for data loading, model setup, and layout selection
2. Publication figure generation using cached data

Other scripts (plot_predictions_article.py, wt_predictions.py) import shared
functions from this module to avoid code duplication.

Usage:
    # Generate figures from cached data (local machine)
    python plot_publication_figures.py

    # Generate cache files on Sophia cluster first:
    python generate_plot_data.py
"""

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from utils.data_tools import setup_test_val_iterator
from utils.plotting import matplotlib_set_rcparams, plot_probe_graph_fn
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

# =============================================================================
# SHARED FUNCTIONS - Import these in other scripts
# =============================================================================


def get_model_paths(main_path: str) -> dict:
    """Get standard paths for model artifacts.

    Args:
        main_path: Base path to the model directory

    Returns:
        Dictionary with paths for config, model_config, and params
    """
    return {
        "cfg_path": os.path.join(main_path, ".hydra/config.yaml"),
        "model_cfg_path": os.path.abspath(os.path.join(main_path, "model_config.json")),
        "params_path": os.path.join(main_path, "best_params.msgpack"),
    }


def load_article1_model(
    main_path: str,
    test_data_path: str | None = None,
    use_dummy_input: bool = True,
):
    """Load GNO model for article 1 plotting.

    Args:
        main_path: Path to the model directory
        test_data_path: Optional path to test data (required if use_dummy_input=False)
        use_dummy_input: Whether to use dummy input for model initialization

    Returns:
        Tuple of (restored_params, restored_cfg_model, model, dropout_active, dataset)
    """
    paths = get_model_paths(main_path)

    # Load model config
    with open(paths["model_cfg_path"]) as f:
        nested_dict = json.load(f)
        restored_cfg_model = DictConfig(nested_dict)

    # Determine test data path
    if test_data_path is None:
        test_data_path = os.path.abspath("./data/zenodo_graphs/test_pre_processed")

    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

    if use_dummy_input and os.path.exists(os.path.join(main_path, "dummy_input.py")):
        # Load dummy input for faster model init
        dummy_file = os.path.join(main_path, "dummy_input.py")
        spec = importlib.util.spec_from_file_location("dummy_input", dummy_file)
        dummy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dummy_module)
        graphs = dummy_module.graphs
        probe_graphs = dummy_module.probe_graphs
        node_array_tuple = dummy_module.node_array_tuple

        restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
            paths["params_path"],
            paths["model_cfg_path"],
            dataset=None,
            inputs=(graphs, probe_graphs, node_array_tuple),
        )
    else:
        restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
            paths["params_path"], paths["model_cfg_path"], dataset
        )

    return restored_params, restored_cfg_model, model, dropout_active, dataset


def select_representative_layouts(
    dataset,
    target_n_wt: int = 50,
    target_wt_spacing: float = 5.0,
    idx_start: int = 0,
) -> dict[str, int | None]:
    """Select representative graph indices for each layout type based on y-range similarity.

    Finds one representative graph per layout type that has similar y-range to a
    reference cluster case (with specified n_wt and wt_spacing constraints).

    Args:
        dataset: Torch_Geomtric_Dataset to search through
        target_n_wt: Minimum number of wind turbines for reference case
        target_wt_spacing: Target wind turbine spacing for reference case
        idx_start: Starting index for search

    Returns:
        Dictionary mapping layout type names to dataset indices
    """
    layout_type_idxs = {
        "cluster": None,
        "single string": None,
        "multiple string": None,
        "parallel string": None,
    }

    # Track best y-range offsets for each layout
    best_offsets = {lt: 1e9 for lt in layout_type_idxs}
    y_chosen = False
    y_range_chosen = None

    for idx, data in tqdm(enumerate(dataset), desc="Selecting representative layouts"):
        layout_type = data.layout_type
        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()

        # Find reference y-range from first valid cluster case
        if (
            idx > idx_start
            and layout_type == "cluster"
            and not y_chosen
            and np.round(data.wt_spacing, 0) == target_wt_spacing
            and data.n_wt >= target_n_wt
        ):
            y_range_chosen = y_range
            y_chosen = True

        if y_chosen:
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < best_offsets[layout_type]:
                layout_type_idxs[layout_type] = idx
                best_offsets[layout_type] = y_range_offset

    return layout_type_idxs


def select_representative_layouts_per_windspeed(
    dataset,
    scale_stats: dict,
    target_windspeeds: list[float] = [6.0, 12.0, 18.0],
    windspeed_tolerance: float = 0.5,
    target_n_wt: int = 50,
    target_wt_spacing: float = 5.0,
    idx_start: int = 0,
) -> dict[str, dict[float, int | None]]:
    """Select representative graph indices for each layout type AND wind speed.

    For each combination of layout type and target wind speed, finds one
    representative graph that matches the criteria.

    Uses a two-pass approach:
    1. First pass: Find reference y_range from a valid cluster case
    2. Second pass: Select best samples for each (layout_type, wind_speed)

    Args:
        dataset: Torch_Geomtric_Dataset to search through
        scale_stats: Scaling statistics to convert scaled -> actual wind speeds
        target_windspeeds: List of target wind speeds in m/s (e.g., [6, 12, 18])
        windspeed_tolerance: Acceptable deviation from target wind speed in m/s
        target_n_wt: Minimum number of wind turbines for reference case
        target_wt_spacing: Target wind turbine spacing for reference case
        idx_start: Starting index for search

    Returns:
        Nested dictionary: {layout_type: {wind_speed: dataset_idx}}
    """
    layout_types = ["cluster", "single string", "multiple string", "parallel string"]

    # Initialize result structure
    layout_ws_idxs = {lt: {ws: None for ws in target_windspeeds} for lt in layout_types}

    # Track best y-range offsets for each (layout_type, wind_speed)
    best_offsets = {lt: {ws: 1e9 for ws in target_windspeeds} for lt in layout_types}

    # Get velocity scaling parameters
    vel_min = scale_stats["velocity"]["min"][0]
    vel_range = scale_stats["velocity"]["range"][0]

    # === PASS 1: Find reference y-range from first valid cluster case ===
    y_range_chosen = None
    for idx, data in enumerate(dataset):
        if idx <= idx_start:
            continue
        if data.layout_type != "cluster":
            continue
        if np.round(data.wt_spacing, 0) != target_wt_spacing:
            continue
        if data.n_wt < target_n_wt:
            continue
        # Found a valid cluster case
        y_range_chosen = data.pos[:, 1].max() - data.pos[:, 1].min()
        break

    if y_range_chosen is None:
        print("  Warning: No valid cluster reference found, using y_range=0")
        y_range_chosen = 0

    # === PASS 2: Select samples for each (layout_type, wind_speed) ===
    for idx, data in tqdm(
        enumerate(dataset), desc="Selecting per-windspeed layouts", total=len(dataset)
    ):
        layout_type = data.layout_type
        if layout_type not in layout_types:
            continue

        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()
        y_range_offset = np.abs(y_range_chosen - y_range)

        # Get actual wind speed from scaled global features
        scaled_U = data.global_features[0].item()
        actual_U = scaled_U * vel_range + vel_min

        # Check each target wind speed
        for target_ws in target_windspeeds:
            if np.abs(actual_U - target_ws) <= windspeed_tolerance:
                # This sample matches the wind speed criterion
                if y_range_offset < best_offsets[layout_type][target_ws]:
                    layout_ws_idxs[layout_type][target_ws] = idx
                    best_offsets[layout_type][target_ws] = y_range_offset

    # Report what was found
    print("\nPer-windspeed selection results:")
    for lt in layout_types:
        for ws in target_windspeeds:
            idx = layout_ws_idxs[lt][ws]
            status = f"idx={idx}" if idx is not None else "NOT FOUND"
            print(f"  {lt}, U={ws} m/s: {status}")

    return layout_ws_idxs


def get_max_plot_distance(dataset, layout_type_idxs: dict) -> tuple[float, float]:
    """Compute max distance and y_range for consistent plot scaling.

    Args:
        dataset: Torch_Geomtric_Dataset
        layout_type_idxs: Dictionary mapping layout types to dataset indices

    Returns:
        Tuple of (plot_distance, max_y_range) for use in plotting
    """
    max_distance = 0
    max_y_range = 0
    for idx in layout_type_idxs.values():
        if idx is None:
            continue
        data = dataset[idx]
        x_range = data.pos[:, 0].max() - data.pos[:, 0].min()
        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()
        distance = max(x_range, y_range)
        if distance > max_distance:
            max_distance = distance
        if y_range > max_y_range:
            max_y_range = y_range

    plot_distance = max_distance * 1.5
    return plot_distance, max_y_range


def setup_plot_iterator(cfg, test_data_path: str, dataset, layout_idxs: dict):
    """Setup iterator and retrieve data for specified layout indices.

    Args:
        cfg: Model configuration
        test_data_path: Path to test data
        dataset: Torch_Geomtric_Dataset
        layout_idxs: Dictionary mapping layout types to indices

    Returns:
        Dictionary mapping layout types to their graph data tuples
    """
    from copy import deepcopy

    get_plot_data_iterator, test_dataset, _, _ = setup_test_val_iterator(
        cfg,
        type_str="test",
        return_idxs=True,
        return_positions=True,
        path=test_data_path,
        cache=False,
        return_layout_info=True,
        dataset=dataset,
        num_workers=0,  # Disable multiprocessing to avoid JAX fork issues
    )

    iterator = get_plot_data_iterator()
    plot_graphs = deepcopy(layout_idxs)

    for i, data_in in tqdm(enumerate(iterator), desc="Loading plot data"):
        if i in layout_idxs.values():
            for key, val in layout_idxs.items():
                if val == i:
                    plot_graphs[key] = data_in

    return plot_graphs, test_dataset


def apply_normalizations(
    predictions: np.ndarray,
    targets: np.ndarray,
    U_flow: float,
    plot_velocity_deficit: bool = True,
    normalize_by_U: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply velocity deficit and normalization transformations.

    Args:
        predictions: Predicted velocity values
        targets: Target velocity values
        U_flow: Freestream velocity
        plot_velocity_deficit: If True, subtract U_flow to show deficit
        normalize_by_U: If True, normalize by U_flow

    Returns:
        Tuple of (transformed_predictions, transformed_targets, label_string)
    """
    label = ""
    if plot_velocity_deficit:
        predictions = predictions - U_flow
        targets = targets - U_flow
        label += r"$\Delta u"
    else:
        label += r"$u"

    if normalize_by_U:
        predictions = predictions / U_flow
        targets = targets / U_flow
        label += r"/U$ [-]"
    else:
        label += r"$ [$\mathrm{ms}^{-1}$]"

    return predictions, targets, label


def apply_mask(arr, mask):
    """Apply mask and remove NaN values.

    Args:
        arr: Input array
        mask: Boolean mask (0 for masked, non-zero for valid)

    Returns:
        Array with masked values removed (as numpy array)
    """
    # Convert to numpy for memory-efficient masking
    arr = np.asarray(arr)
    mask = np.asarray(mask)
    arr = np.where(mask != 0, arr, np.nan)
    arr = arr[~np.isnan(arr)]
    return arr


# =============================================================================
# PUBLICATION FIGURE FUNCTIONS
# =============================================================================


def plot_crossstream_figure(
    cache_data: dict,
    output_path: str,
    model_type_str: str = "best_model",
):
    """Generate the crossstream profiles publication figure with improved visibility.

    Args:
        cache_data: Dictionary with cached crossstream prediction data
        output_path: Path to save the figure
        model_type_str: Model type string for filename
    """
    matplotlib_set_rcparams("paper")

    # Increase base font sizes
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    metadata = cache_data["metadata"]
    global_limits = cache_data["global_limits"]
    layout_data = cache_data["layout_data"]

    x_downstream = metadata["x_downstream"]
    U_free = metadata["U_free"]
    velocity_range = global_limits["velocity_range"]

    # Get colors from style
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

    # Layout names in order (matches wt_spatial_errors figure)
    layout_names = ["cluster", "multiple string", "parallel string", "single string"]

    # Pre-calculate layout-specific bounds for better space usage
    layout_bounds = {}
    max_y_extent = 0

    for layout_name in layout_names:
        data = layout_data[layout_name]
        first_pred = data["predictions"][x_downstream[0]][U_free[0]]
        positions = first_pred["unscaled_node_positions"]
        wt_mask = data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]
        wt_pos = positions[wt_idx]

        # Calculate bounds with padding
        padding = 10  # D units
        x_min, x_max = wt_pos[:, 0].min() - padding, wt_pos[:, 0].max() + padding
        y_min, y_max = wt_pos[:, 1].min() - padding, wt_pos[:, 1].max() + padding

        # Store shrink factors for manual subplot adjustment (applied later)
        # These shrink the actual subplot, not the axis range
        shrink_factors = {
            "single string": 0.70,
            "multiple string": 0.70,
        }  # 30% shrink for both

        # For profile plots, we want y_range centered on farm
        y_center = (wt_pos[:, 1].min() + wt_pos[:, 1].max()) / 2
        y_extent = max(abs(wt_pos[:, 1].min() - y_center), abs(wt_pos[:, 1].max() - y_center))
        y_extent = y_extent + padding  # Add padding

        layout_bounds[layout_name] = {
            "x_lim": (x_min, x_max),
            "y_lim": (y_min, y_max),
            "y_center": y_center,
            "y_extent": y_extent,
            "x_range": x_max - x_min,
            "y_range": y_max - y_min,
            "shrink_factor": shrink_factors.get(layout_name, 1.0),
        }
        max_y_extent = max(max_y_extent, y_extent)

    # Figure setup
    n_rows = len(layout_names)
    n_cols = 1 + len(x_downstream)

    # Standard figure size
    fig = plt.figure(figsize=(14, 3.5 * n_rows))

    # Single GridSpec for all columns - ensures row alignment
    # All columns share the same row structure
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        width_ratios=[2.5] + [1.0] * len(x_downstream),
        hspace=0.18,
        wspace=0.12,
    )

    for row_idx, layout_name in enumerate(layout_names):
        data = layout_data[layout_name]
        bounds = layout_bounds[layout_name]

        # Get graph plotting data from first prediction
        first_pred = data["predictions"][x_downstream[0]][U_free[0]]
        unscaled_node_positions = first_pred["unscaled_node_positions"]
        unscaled_graphs = first_pred["unscaled_graphs"]
        unscaled_probe_graphs = first_pred["unscaled_probe_graphs"]

        wt_mask = data["wt_mask"]
        wt_spacing = data["wt_spacing"]

        # Create axes from single GridSpec - ensures row alignment
        farm_ax = fig.add_subplot(gs[row_idx, 0])  # Farm plot

        # Apply shrink factor to farm plot if needed
        shrink = bounds["shrink_factor"]
        if shrink < 1.0:
            # Get current position and shrink from center
            pos = farm_ax.get_position()
            new_width = pos.width * shrink
            new_height = pos.height * shrink
            new_x0 = pos.x0 + (pos.width - new_width) / 2
            new_y0 = pos.y0 + (pos.height - new_height) / 2
            farm_ax.set_position([new_x0, new_y0, new_width, new_height])

        row_axes = [farm_ax]
        for j in range(len(x_downstream)):
            row_axes.append(fig.add_subplot(gs[row_idx, j + 1]))  # Velocity plots

        # Plot farm layout with refined style
        plot_probe_graph_fn(
            unscaled_graphs,
            unscaled_probe_graphs,
            unscaled_node_positions,
            include_probe_edges=False,
            include_probe_nodes=False,
            ax=row_axes[0],
            edge_linewidth=0.8,
            wt_node_size=90,  # Marker size (bigger and longer)
            wt_color="black",  # Black inner marker
            wt_edgecolor="white",  # White outline (thicker marker underneath)
            wt_linewidth=1.2,  # Inner line width (outline will be 3x thicker)
            edge_color="gray",  # Muted gray edges
            edge_alpha=0.4,  # Subtle edges (less important)
            wt_marker="2",  # Tri-up marker
        )

        # All farm plots get x-axis label and ticks
        row_axes[0].set_xlabel(r"$x/D$ [-]")
        row_axes[0].set_ylabel(r"$y/D$ [-]")

        # Use layout-specific bounds with equal aspect ratio
        row_axes[0].set_xlim(bounds["x_lim"])
        row_axes[0].set_ylim(bounds["y_lim"])
        row_axes[0].set_aspect("equal")

        # Remove legend from individual plots - will add figure-level legend
        if row_axes[0].legend_ is not None:
            row_axes[0].legend_.remove()

        # Add row label as title above the farm plot (centered)
        letter = chr(97 + row_idx)
        n_wt = int(np.sum(wt_mask))
        row_title = f"({letter}) {layout_name}, $n_\\mathrm{{wt}}={n_wt}$, $s_\\mathrm{{wt}}={wt_spacing:.1f}D$"
        row_axes[0].set_title(row_title, loc="center", fontsize=12, pad=8)

        # Plot cross-stream profiles
        for ds_col, x_sel in enumerate(x_downstream):
            ax = row_axes[ds_col + 1]

            for U_flow_val, color in zip(U_free, colors):
                pred_data = data["predictions"][x_sel][U_flow_val]

                predictions = pred_data["normalized_predictions"]
                targets = pred_data["normalized_targets"]
                y_D = pred_data["y/D"]

                # C. Thicker lines for visibility
                ax.plot(
                    predictions,
                    y_D,
                    ls="-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.75,
                )
                # D. Improved dash pattern
                ax.plot(
                    targets,
                    y_D,
                    ls="--",
                    dashes=(4, 3),
                    linewidth=1.5,
                    color=color,
                    alpha=0.75,
                )

            ax.set_xlim(velocity_range)
            # Use layout-specific y-limits matching the farm plot
            ax.set_ylim(bounds["y_lim"])

            # X-axis: show tick markers on all rows, but labels only on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\Delta u/U$ [-]")
                ax.tick_params(axis="x", rotation=30)
            else:
                # Non-bottom rows: keep tick markers, hide labels
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            # Y-axis: show ylabel and yticks on first velocity column (column 2)
            if ds_col == 0:
                ax.set_ylabel(r"$y/D$ [-]")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # Only show column title on first row (use \widetilde for wider tilde)
            if row_idx == 0:
                ax.set_title(f"$\\widetilde{{x}}={x_sel}D$", fontsize=12, pad=8)

    # Calculate legend positions
    # Width ratios are [2.5, 1, 1, 1], total = 5.5
    total_width = 2.5 + len(x_downstream)  # 5.5

    # First legend above column 1 (farm plots) - shift right to center on plot (a)
    farm_center_x = (2.5 / 2) / total_width + 0.06  # Moved right (~2 letters)

    # Second legend above velocity deficit plots
    deficit_center_x = (2.5 + total_width) / 2 / total_width - 0.03  # Moved right (~2 letters)

    # First legend (WT/edges) - above farm plots column, closer to plots
    first_ax = fig.axes[0]
    wt_handles, wt_labels = first_ax.get_legend_handles_labels()
    legend_y = 0.95  # Moved down (~1.5 letters)
    if wt_handles:
        fig.legend(
            wt_handles[::-1],
            wt_labels[::-1],
            loc="upper center",
            bbox_to_anchor=(farm_center_x, legend_y),
            fontsize=11,
            frameon=True,
            ncol=2,
        )

    # Second legend (line styles and colors) - above velocity deficit plots
    color_legend_labels = [f"{U} m/s" for U in U_free]
    lines = [
        plt.Line2D([0], [0], color="black", ls="-", alpha=0.75, lw=2),
        plt.Line2D([0], [0], color="black", ls="--", dashes=(4, 3), alpha=0.75, lw=2),
    ]
    legend_labels = ["GNO", "Target"]
    for c, lab in zip(colors, color_legend_labels):
        lines.append(plt.Line2D([0], [0], color=c, ls="-", alpha=0.75, lw=2))
        legend_labels.append(lab)

    fig.legend(
        lines,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(deficit_center_x, legend_y),
        fontsize=11,
        frameon=True,
        ncol=5,
    )

    output_file = os.path.join(output_path, f"crossstream_profiles_{model_type_str}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def export_crossstream_errors_csv(cache_data: dict, output_path: str) -> None:
    """Export crossstream profile error statistics to CSV.

    Args:
        cache_data: Dictionary with cached crossstream prediction data
        output_path: Path to save the CSV file
    """
    import csv
    import math

    rows = []
    metadata = cache_data["metadata"]

    for layout_name, layout_data in cache_data["layout_data"].items():
        for x_sel in metadata["x_downstream"]:
            for U_flow in metadata["U_free"]:
                errors = layout_data["predictions"][x_sel][U_flow].get("errors", {})

                for metric_name, stats in errors.items():
                    rows.append(
                        {
                            "layout_type": layout_name,
                            "x_downstream_D": x_sel,
                            "U_free_ms": U_flow,
                            "metric": metric_name,
                            "min": stats["min"],
                            "max": stats["max"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "max_y_D": stats.get("max_y_D", float("nan")),
                        }
                    )

    output_file = os.path.join(output_path, "crossstream_profiles_errors.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved error statistics: {output_file}")

    # Generate summary CSV aggregating across all layouts/positions/wind speeds
    metric_descriptions = {
        "freestream_normalized": "|u-û|/U",
        "percentage": "|u-û|/|u|",
        "deficit_normalized": "|u-û|/|U-u|",
        "target_deficit_ms": "U-u [m/s]",
        "prediction_error_ms": "u-û [m/s]",
        "deficit_normalized_filtered": "|u-û|/|U-u| (deficit>5%)",
        "avg_deficit_rel_error": "|mean(U-û)-mean(U-u)|/mean(U-u)",
        "avg_velocity_rel_error": "|mean(u)-mean(û)|/mean(u)",
        "velocity_rel_error": "|u-û|/u",
    }

    summary_rows = []
    metric_names = list(metric_descriptions.keys())

    for metric_name in metric_names:
        metric_rows = [r for r in rows if r["metric"] == metric_name]
        if not metric_rows:
            continue

        # Filter out NaN values for aggregation
        valid_rows = [r for r in metric_rows if not (math.isnan(r["max"]) or math.isnan(r["mean"]))]
        if not valid_rows:
            continue

        # Find overall statistics
        overall_min = min(r["min"] for r in valid_rows if not math.isnan(r["min"]))
        overall_max = max(r["max"] for r in valid_rows)
        mean_min = min(r["mean"] for r in valid_rows)
        mean_max = max(r["mean"] for r in valid_rows)

        # Find range of max errors across different layout/inflow combinations
        max_values = [r["max"] for r in valid_rows]
        max_min = min(max_values)  # Smallest max error across cases
        max_max = max(max_values)  # Largest max error across cases

        # Find where max occurred
        max_row = max(valid_rows, key=lambda r: r["max"])

        description = metric_descriptions.get(metric_name, "")
        display_name = f"{metric_name} ({description})" if description else metric_name

        summary_rows.append(
            {
                "metric": display_name,
                "overall_min_pct": round(overall_min, 2),
                "overall_max_pct": round(overall_max, 2),
                "max_min_pct": round(max_min, 2),
                "max_max_pct": round(max_max, 2),
                "mean_min_pct": round(mean_min, 2),
                "mean_max_pct": round(mean_max, 2),
                "max_at_layout": max_row["layout_type"],
                "max_at_x_D": max_row["x_downstream_D"],
                "max_at_y_D": round(max_row["max_y_D"], 2)
                if not math.isnan(max_row["max_y_D"])
                else "nan",
                "max_at_U_ms": max_row["U_free_ms"],
            }
        )

    summary_file = os.path.join(output_path, "crossstream_errors_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved error summary: {summary_file}")


def export_deficit_threshold_sweep(cache_data: dict, output_path: str) -> None:
    """Sweep deficit filter threshold from 1% to 10% and export statistics.

    This creates a table showing how the deficit-normalized error metric
    changes as we vary the minimum deficit threshold for filtering.

    Args:
        cache_data: Dictionary with cached crossstream prediction data
        output_path: Path to save the CSV file
    """
    import csv

    metadata = cache_data["metadata"]
    # Use logarithmic spacing from 0.01% to 10%
    threshold_pcts = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,  # 10^-2 range
        0.1,
        0.2,
        0.3,
        0.5,
        0.7,  # 10^-1 range
        1,
        2,
        3,
        5,
        7,  # 10^0 range
        10,  # 10^1 range
    ]
    thresholds = [p / 100.0 for p in threshold_pcts]

    # Collect all raw data needed for recomputing the metric
    all_results = []

    for layout_name, layout_data in cache_data["layout_data"].items():
        for x_sel in metadata["x_downstream"]:
            for U_flow in metadata["U_free"]:
                pred_data = layout_data["predictions"][x_sel][U_flow]
                u = np.array(pred_data["unscaled_targets"])
                u_hat = np.array(pred_data["unscaled_predictions"])
                U = U_flow

                error_abs = np.abs(u - u_hat)
                target_deficit = np.abs(U - u)

                all_results.append(
                    {
                        "layout": layout_name,
                        "x_D": x_sel,
                        "U": U_flow,
                        "error_abs": error_abs,
                        "target_deficit": target_deficit,
                        "U_freestream": U,
                    }
                )

    # Compute statistics for each threshold
    rows = []
    for i, threshold in enumerate(thresholds):
        threshold_pct = threshold_pcts[i]
        all_filtered_errors = []
        n_valid_points = 0
        n_total_points = 0

        for result in all_results:
            min_deficit = threshold * result["U_freestream"]
            mask = result["target_deficit"] > min_deficit

            n_total_points += len(result["error_abs"])
            n_valid_points += np.sum(mask)

            if np.any(mask):
                filtered_error = (result["error_abs"][mask] / result["target_deficit"][mask]) * 100
                all_filtered_errors.extend(filtered_error.tolist())

        if all_filtered_errors:
            all_filtered_errors = np.array(all_filtered_errors)
            rows.append(
                {
                    "threshold_pct": threshold_pct,
                    "min": round(float(np.min(all_filtered_errors)), 2),
                    "max": round(float(np.max(all_filtered_errors)), 2),
                    "mean": round(float(np.mean(all_filtered_errors)), 2),
                    "std": round(float(np.std(all_filtered_errors)), 2),
                    "median": round(float(np.median(all_filtered_errors)), 2),
                    "p95": round(float(np.percentile(all_filtered_errors, 95)), 2),
                    "p99": round(float(np.percentile(all_filtered_errors, 99)), 2),
                    "n_valid_points": n_valid_points,
                    "pct_data_retained": round(100 * n_valid_points / n_total_points, 1),
                }
            )
        else:
            # No valid data at this threshold - add row with NaN
            rows.append(
                {
                    "threshold_pct": threshold_pct,
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "p95": float("nan"),
                    "p99": float("nan"),
                    "n_valid_points": 0,
                    "pct_data_retained": 0.0,
                }
            )

    # Export CSV
    output_file = os.path.join(output_path, "deficit_threshold_sweep.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved threshold sweep: {output_file}")

    # Generate a simple plot with logarithmic x-axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    thresholds_pct = [r["threshold_pct"] for r in rows]

    # Key thresholds to highlight
    key_thresholds = [0.1, 1.0]

    # Left plot: max, mean, p95, p99
    ax1 = axes[0]
    ax1.plot(thresholds_pct, [r["max"] for r in rows], "o-", label="Max", color="C3")
    ax1.plot(thresholds_pct, [r["p99"] for r in rows], "s-", label="99th percentile", color="C1")
    ax1.plot(thresholds_pct, [r["p95"] for r in rows], "^-", label="95th percentile", color="C2")
    ax1.plot(thresholds_pct, [r["mean"] for r in rows], "d-", label="Mean", color="C0")
    # Add vertical lines at key thresholds
    for thresh in key_thresholds:
        ax1.axvline(x=thresh, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Deficit threshold (% of freestream)")
    ax1.set_ylabel("Deficit-normalized error [%]")
    ax1.set_title("Error statistics vs. filter threshold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0.01, 10)
    ax1.set_xticks([0.01, 0.1, 1, 10])
    ax1.set_xticklabels(["0.01", "0.1", "1", "10"])

    # Right plot: percentage of data retained
    ax2 = axes[1]
    ax2.plot(thresholds_pct, [r["pct_data_retained"] for r in rows], "o-", color="C4")
    # Add vertical lines at key thresholds
    for thresh in key_thresholds:
        ax2.axvline(x=thresh, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Deficit threshold (% of freestream)")
    ax2.set_ylabel("Data retained [%]")
    ax2.set_title("Data coverage vs. filter threshold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(0.01, 10)
    ax2.set_xticks([0.01, 0.1, 1, 10])
    ax2.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plot_file = os.path.join(output_path, "deficit_threshold_sweep.pdf")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved threshold sweep plot: {plot_file}")


def plot_wt_spatial_errors_figure(
    cache_data: dict,
    output_path: str,
    model_type_str: str = "best_model",
    metric_to_plot: str = "abs_max",
):
    """Generate the wind turbine spatial errors publication figure.

    Args:
        cache_data: Dictionary with cached WT error data
        output_path: Path to save the figure
        model_type_str: Model type string for filename
        metric_to_plot: Which metric to plot ("mean", "std", or "abs_max")
    """
    matplotlib_set_rcparams("paper")

    aggregated_extremas = cache_data["aggregated_extremas"]
    layout_data = cache_data["layout_data"]

    # Get turbine diameter
    from py_wake.examples.data.dtu10mw import DTU10MW

    wt = DTU10MW()
    wt.diameter()

    # All layout names for bounds calculation
    all_layout_names = ["cluster", "single string", "multiple string", "parallel string"]

    # Pre-calculate bounds for each layout
    layout_bounds = {}
    padding = 8  # D units
    for layout_name in all_layout_names:
        data = layout_data[layout_name]
        # Handle both per-windspeed (dict) and aggregated (array) position formats
        pos_data = data["wt_positions_D"]
        if isinstance(pos_data, dict):
            # Use first available wind speed's positions for bounds
            first_U = next(iter(pos_data.keys()))
            wt_positions = pos_data[first_U]
        else:
            wt_positions = pos_data
        x_min, x_max = (
            wt_positions[:, 0].min() - padding,
            wt_positions[:, 0].max() + padding,
        )
        y_min, y_max = (
            wt_positions[:, 1].min() - padding,
            wt_positions[:, 1].max() + padding,
        )
        layout_bounds[layout_name] = {
            "x_lim": (x_min, x_max),
            "y_lim": (y_min, y_max),
            "x_range": x_max - x_min,
            "y_range": y_max - y_min,
        }

    # Calculate common y-range for all plots to ensure equal heights with aspect="equal"
    max_y_range = max(layout_bounds[l]["y_range"] for l in all_layout_names)

    # Update bounds to use common y-range (centered on each layout's data)
    for layout_name in all_layout_names:
        bounds = layout_bounds[layout_name]
        y_center = (bounds["y_lim"][0] + bounds["y_lim"][1]) / 2
        bounds["y_lim"] = (y_center - max_y_range / 2, y_center + max_y_range / 2)

    # Layout order for plotting
    # Row 1: cluster, multiple string, parallel string
    # Row 2: single string (full width)
    row1_layouts = ["cluster", "multiple string", "parallel string"]

    # Width ratios based on actual x-ranges (for equal height with aspect="equal")
    row1_ratios = [layout_bounds[l]["x_range"] for l in row1_layouts]

    # Create figure - narrower width forces plots closer together
    fig = plt.figure(figsize=(10, 6))

    # Create nested gridspec
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Row 1: three subplots with width ratios based on x-ranges
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs[0], width_ratios=row1_ratios, wspace=0
    )
    # Row 2: single string takes full width
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[1])

    # Create axes in the new order: cluster, multiple string, parallel string, single string
    axes = [
        fig.add_subplot(gs1[0]),  # (a) cluster
        fig.add_subplot(gs1[1]),  # (b) multiple string
        fig.add_subplot(gs1[2]),  # (c) parallel string
        fig.add_subplot(gs2[0]),  # (d) single string
    ]

    # Reorder layout_names to match the new axes order
    layout_names = ["cluster", "multiple string", "parallel string", "single string"]

    # Determine colorbar limits
    extrema = np.max(np.abs(aggregated_extremas[metric_to_plot]))
    if metric_to_plot == "abs_max":
        vmin = 0
        vmax = extrema
        colormap = "gist_heat_r"
        cbar_legend = r"$\max| (\boldsymbol{u}-\hat{\boldsymbol{u}})/\boldsymbol{U} |$ [$\%$]"
        cbar_extend = "max"  # Pointy end on max side (values can exceed)
    else:
        vmin = -extrema
        vmax = extrema
        colormap = "seismic"
        cbar_legend = (
            r"$\overline{u_\mathrm{err}/U}$"
            if metric_to_plot == "mean"
            else r"$\sigma(u_\mathrm{err}/U)$"
        )
        cbar_extend = "both"  # Pointy ends on both sides

    sc = None
    for i, layout_name in enumerate(layout_names):
        ax = axes[i]
        data = layout_data[layout_name]
        bounds = layout_bounds[layout_name]

        wt_positions = data["wt_positions_D"]
        wt_rel_errors_agg = data["aggregated_errors"][metric_to_plot]

        sc = ax.scatter(
            wt_positions[:, 0],
            wt_positions[:, 1],
            c=wt_rel_errors_agg,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            s=25,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.5,
            marker="o",
        )

        ax.set_xlabel("$x/D$ [-]")

        # Only show y-label on first plot of each row: (a) cluster and (d) single string
        if i == 0 or i == 3:
            ax.set_ylabel("$y/D$ [-]")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # Use computed bounds
        ax.set_xlim(bounds["x_lim"])
        ax.set_ylim(bounds["y_lim"])
        ax.set_aspect("equal")

        # Anchor plots to reduce gaps between them in row 1
        if i == 0:  # (a) cluster - anchor right to push toward (b)
            ax.set_anchor("E")
        elif i == 1:  # (b) multiple string - anchor center
            ax.set_anchor("C")
        elif i == 2:  # (c) parallel string - anchor left to push toward (b)
            ax.set_anchor("W")
        # (d) single string - default center anchor is fine

        # Add subplot label
        ax.set_title(f"({chr(97 + i)}) {layout_name}", loc="left", pad=5)

    # Add colorbar with pointy end(s) indicating values can extend beyond range
    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, extend=cbar_extend)
    cbar.set_label(cbar_legend)

    # Manually align left edges of row 1 with row 2
    # Get the left edge position of subplot (d)
    fig.canvas.draw()  # Need to draw first to get accurate positions
    pos_d = axes[3].get_position()

    # Shift row 1 subplots to align with row 2's left edge
    for i in range(3):  # axes 0, 1, 2 are row 1
        pos = axes[i].get_position()
        # Calculate the offset needed to align left edges
        if i == 0:
            offset = pos_d.x0 - pos.x0
        # Apply offset to all row 1 subplots
        axes[i].set_position([pos.x0 + offset, pos.y0, pos.width, pos.height])

    output_file = os.path.join(output_path, f"wt_spatial_errors_all_layouts_{model_type_str}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_wt_spatial_errors_per_windspeed(
    raw_data: dict,
    model,
    model_params,
    unscaler,
    output_path: str,
    U_free_values: list[float] | None = None,
    TI_flow: float = 0.05,
    prototype: bool = False,
) -> None:
    """Generate per-wind-speed WT spatial errors figure.

    Creates a figure showing WT spatial errors for each wind speed separately,
    using the original figure layout structure:
    - Block 1: cluster, multiple, parallel side-by-side (3 rows for wind speeds)
    - Block 2: single string full-width (3 rows for wind speeds)

    Uses construct_on_the_fly_probe_graph to regenerate graphs and targets at
    each target wind speed, producing physically meaningful error variations.

    Args:
        raw_data: Raw cached data containing layout_data
        model: The trained model
        model_params: Model parameters
        unscaler: Unscaler for inverse transformations
        output_path: Directory to save the figure
        U_free_values: List of wind speeds (defaults to [6.0, 12.0, 18.0])
        TI_flow: Turbulence intensity for PyWake simulations (default 0.05 = 5%)
        prototype: If True, use lower DPI for faster generation
    """
    import jax
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.run_pywake import construct_on_the_fly_probe_graph

    matplotlib_set_rcparams("paper")

    if U_free_values is None:
        U_free_values = [6.0, 12.0, 18.0]

    wt = DTU10MW()
    D = wt.diameter()
    scale_stats = raw_data["scale_stats"]

    pred_fn = jax.jit(model.apply)
    _inverse_scale_target = unscaler.inverse_scale_output

    # Layout names in order
    layout_names = ["cluster", "single string", "multiple string", "parallel string"]
    row1_layouts = ["cluster", "multiple string", "parallel string"]

    # First pass: compute errors for all layouts and wind speeds
    # This ensures consistent positions and allows computing global color limits
    layout_errors = {}
    layout_positions = {}
    layout_bounds = {}
    padding = 8  # D units

    all_errors = []

    for layout_name in layout_names:
        layout_data = raw_data["layout_data"].get(layout_name)
        if layout_data is None:
            print(f"  Warning: No data for {layout_name}, skipping")
            continue

        # Extract WT positions from layout_data (SAME source as aggregated figure)
        node_positions = layout_data["node_positions"]
        wt_mask = layout_data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]

        # Get WT positions in meters (consistent for all wind speeds)
        wt_positions_m = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
        wt_positions_D = wt_positions_m / D
        layout_positions[layout_name] = wt_positions_D

        # Calculate bounds
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

        # Create minimal grid for WT-only predictions (just need one probe point)
        grid = HorizontalGrid(
            x=wt_positions_m[:, 0][:1],
            y=wt_positions_m[:, 1][:1],
            h=wt.hub_height(),
        )

        # Run model at EACH target wind speed using PyWake regeneration
        layout_errors[layout_name] = {}
        for U in U_free_values:
            # Regenerate graph with PyWake at this wind speed
            jraph_graph, jraph_probe_graph, node_array_tuple = construct_on_the_fly_probe_graph(
                positions=wt_positions_m,
                U=[U],
                TI=[TI_flow],
                grid=grid,
                scale_stats=scale_stats,
                return_positions=True,
            )
            targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_array_tuple

            # Run model prediction
            prediction = pred_fn(
                model_params,
                jraph_graph,
                jraph_probe_graph,
                jnp.atleast_2d(wt_mask_gen).T,
                jnp.atleast_2d(probe_mask_gen).T,
            ).squeeze()

            # Extract WT predictions and targets
            wt_idx_gen = np.where(wt_mask_gen != 0)[0]
            wt_predictions = np.array(_inverse_scale_target(prediction[wt_idx_gen]).squeeze())
            wt_targets = np.array(
                _inverse_scale_target(np.array(targets_gen)[wt_idx_gen]).squeeze()
            )

            # Compute relative error at this wind speed
            wt_rel_errors = np.abs((wt_targets - wt_predictions) / U) * 100

            layout_errors[layout_name][U] = wt_rel_errors
            all_errors.extend(wt_rel_errors.flatten())

    if not layout_errors:
        print("  No layout data available, skipping per-windspeed figure")
        return

    # Calculate global color limits
    global_extrema = np.max(all_errors)

    # Colormap settings
    vmin = 0
    vmax = global_extrema
    colormap = "gist_heat_r"
    cbar_legend = r"$|(\boldsymbol{u}-\hat{\boldsymbol{u}})/\boldsymbol{U}|$ [$\%$]"
    cbar_extend = "max"

    # Layout G structure:
    # Block 1: cluster, multiple, parallel (3 cols x 3 rows)
    # Block 2: single string (1 col x 3 rows)

    # Get bounds for layouts
    cluster_bounds = layout_bounds.get("cluster", layout_bounds[list(layout_bounds.keys())[0]])
    multiple_bounds = layout_bounds.get("multiple string", cluster_bounds)
    parallel_bounds = layout_bounds.get("parallel string", cluster_bounds)
    single_bounds = layout_bounds.get("single string", cluster_bounds)

    # Width ratios for row 1 layouts
    row1_width_ratios = [layout_bounds[ln]["x_range"] for ln in row1_layouts if ln in layout_bounds]

    # Height ratios
    block1_y_range = max(
        cluster_bounds["y_range"],
        multiple_bounds["y_range"],
        parallel_bounds["y_range"],
    )
    block2_y_range = single_bounds["y_range"]

    # Create figure (wider, less tall to match squeezed aspect ratio)
    fig = plt.figure(figsize=(12, 10))

    # Outer GridSpec: 2 rows (Block 1, Block 2)
    outer_gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[3 * block1_y_range, 3 * block2_y_range],
        hspace=0.25,  # More space between blocks
    )

    # Block 1: 3 rows x 3 columns
    gs_block1 = gridspec.GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=outer_gs[0],
        width_ratios=row1_width_ratios,
        wspace=0.15,  # Initial spacing (will be adjusted manually)
        hspace=0.12,  # Vertical space for titles
    )

    # Block 2: 3 rows x 1 column
    gs_block2 = gridspec.GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=outer_gs[1],
        hspace=0.12,  # Match Block 1 vertical spacing
    )

    axes = {}
    all_axes = []

    # Create axes for Block 1
    for row, U in enumerate(U_free_values):
        for col, layout_name in enumerate(row1_layouts):
            ax = fig.add_subplot(gs_block1[row, col])
            axes[(layout_name, U)] = ax
            all_axes.append(ax)

    # Create axes for Block 2
    for row, U in enumerate(U_free_values):
        ax = fig.add_subplot(gs_block2[row])
        axes[("single string", U)] = ax
        all_axes.append(ax)

    # Plot data
    sc = None
    subplot_labels = {
        "cluster": "a",
        "multiple string": "b",
        "parallel string": "c",
        "single string": "d",
    }

    # Plot Block 1: cluster, multiple, parallel
    for col, layout_name in enumerate(row1_layouts):
        if layout_name not in layout_errors:
            continue

        bounds = layout_bounds[layout_name]
        positions = layout_positions[layout_name]

        for row, U in enumerate(U_free_values):
            ax = axes[(layout_name, U)]
            wt_errors = layout_errors[layout_name][U]

            sc = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=wt_errors,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                s=20,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
                marker="o",
            )

            ax.set_xlim(bounds["x_lim"])
            ax.set_ylim(bounds["y_lim"])
            ax.set_aspect(0.6)  # Wider, less tall

            # Y-labels: Column 0 only (with wind speed info)
            if col == 0:
                ax.set_ylabel(f"$U = {int(U)}$ m/s\n$y/D$ [-]", fontsize=9)
            else:
                ax.set_yticklabels([])

            # X-labels: Bottom row only
            if row == len(U_free_values) - 1:
                ax.set_xlabel("$x/D$ [-]")
                # Rotate x-tick labels for parallel string column (col 2) to fit better
                if col == 2:
                    ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xticklabels([])

            # Layout labels: Top of each column (first row only)
            if row == 0:
                ax.set_title(
                    f"({subplot_labels[layout_name]}) {layout_name}",
                    fontsize=10,
                    pad=8,
                )

    # Plot Block 2: single string
    if "single string" in layout_errors:
        bounds = layout_bounds["single string"]
        positions = layout_positions["single string"]

        for row, U in enumerate(U_free_values):
            ax = axes[("single string", U)]
            wt_errors = layout_errors["single string"][U]

            sc = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=wt_errors,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                s=20,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
                marker="o",
            )

            ax.set_xlim(bounds["x_lim"])
            ax.set_ylim(bounds["y_lim"])
            ax.set_aspect(0.6)  # Wider, less tall

            # Y-labels: All rows (with wind speed info)
            ax.set_ylabel(f"$U = {int(U)}$ m/s\n$y/D$ [-]", fontsize=9)

            # X-labels: Bottom row only
            if row == len(U_free_values) - 1:
                ax.set_xlabel("$x/D$ [-]")
            else:
                ax.set_xticklabels([])

            # Layout label: Top of column (first row only)
            if row == 0:
                ax.set_title(
                    f"({subplot_labels['single string']}) single string",
                    fontsize=10,
                    pad=8,
                )

    # Horizontal colorbar at bottom
    cbar = fig.colorbar(
        sc,
        ax=all_axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.06,
        aspect=40,
        extend=cbar_extend,
    )
    cbar.set_label(cbar_legend)

    # Manually reposition Block 1 axes to match Block 2's width
    fig.canvas.draw()  # Need to draw first to get accurate positions

    # Get Block 2 extent as target
    block2_ax = axes[("single string", U_free_values[0])]
    block2_pos = block2_ax.get_position()
    target_left = block2_pos.x0
    target_right = block2_pos.x1
    target_width = target_right - target_left

    # Get Block 1 axes grouped by column
    block1_axes_by_col = {
        col: [axes[(ln, U)] for U in U_free_values]
        for col, ln in enumerate(row1_layouts)
        if ln in layout_errors
    }

    if len(block1_axes_by_col) == 3:
        # Get positions of each column (use first row as reference)
        col_positions = []
        for col in range(3):
            ax = block1_axes_by_col[col][0]
            pos = ax.get_position()
            col_positions.append({"left": pos.x0, "right": pos.x1, "width": pos.width})

        # Calculate current Block 1 extent and total width
        block1_left = col_positions[0]["left"]
        block1_right = col_positions[2]["right"]
        current_width = block1_right - block1_left

        # Calculate gaps between columns
        gap_01 = col_positions[1]["left"] - col_positions[0]["right"]
        gap_12 = col_positions[2]["left"] - col_positions[1]["right"]

        # Target gap (enough space for titles not to overlap)
        target_gap = 0.025

        # Width after closing gaps
        width_after_gaps = current_width - (gap_01 - target_gap) - (gap_12 - target_gap)

        # Scale factor to match Block 2's width
        scale = target_width / width_after_gaps

        # Calculate new positions for each column
        # Column 0: starts at target_left, scaled width
        new_col0_width = col_positions[0]["width"] * scale
        new_col0_left = target_left
        new_col0_right = new_col0_left + new_col0_width

        # Column 1: starts after col0 + scaled gap
        new_col1_width = col_positions[1]["width"] * scale
        new_col1_left = new_col0_right + target_gap * scale
        new_col1_right = new_col1_left + new_col1_width

        # Column 2: starts after col1 + scaled gap
        new_col2_width = col_positions[2]["width"] * scale
        new_col2_left = new_col1_right + target_gap * scale

        # Apply new positions to all axes in each column
        for ax in block1_axes_by_col[0]:
            pos = ax.get_position()
            ax.set_position([new_col0_left, pos.y0, new_col0_width, pos.height])

        for ax in block1_axes_by_col[1]:
            pos = ax.get_position()
            ax.set_position([new_col1_left, pos.y0, new_col1_width, pos.height])

        for ax in block1_axes_by_col[2]:
            pos = ax.get_position()
            ax.set_position([new_col2_left, pos.y0, new_col2_width, pos.height])

    # Save figure
    output_file = os.path.join(output_path, "wt_spatial_errors_per_windspeed.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_debug_predictions_targets(
    raw_data: dict,
    model,
    model_params,
    unscaler,
    output_path: str,
    U_free_values: list[float] | None = None,
) -> None:
    """Generate debug plots showing raw predictions vs targets at WT locations.

    Creates one figure per layout type, showing 2 columns (predictions | targets)
    x 3 rows (wind speeds). Uses the same layout sample for all wind speeds
    (scaled by wind speed ratio) to ensure consistent turbine positions.

    This helps verify that error calculations make sense by visualizing
    the actual velocity values in m/s.

    Args:
        raw_data: Raw cached data containing layout_data
        model: The trained model
        model_params: Model parameters
        unscaler: Unscaler for inverse transformations
        output_path: Directory to save the figures
        U_free_values: List of wind speeds to include (defaults to [6.0, 12.0, 18.0])
    """
    import jax
    from jax import numpy as jnp
    from py_wake.examples.data.dtu10mw import DTU10MW

    matplotlib_set_rcparams("paper")

    if U_free_values is None:
        U_free_values = [6.0, 12.0, 18.0]

    wt = DTU10MW()
    D = wt.diameter()

    pred_fn = jax.jit(model.apply)
    _inverse_scale_target = unscaler.inverse_scale_output

    layout_names = ["cluster", "single string", "multiple string", "parallel string"]

    for layout_name in layout_names:
        layout_data = raw_data["layout_data"].get(layout_name)

        if layout_data is None:
            print(f"  No data for {layout_name}, skipping debug plot")
            continue

        # Get data from main layout_data (consistent sample for all wind speeds)
        graphs = dict_to_graph(layout_data["graphs"])
        probe_graphs = dict_to_graph(layout_data["probe_graphs"])
        targets = layout_data["targets"]
        wt_mask = layout_data["wt_mask"]
        probe_mask = layout_data["probe_mask"]
        node_positions = layout_data["node_positions"]

        wt_idx = np.where(wt_mask != 0)[0]

        # Get WT positions (consistent for all wind speeds)
        wt_positions = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
        wt_positions_D = wt_positions / D

        # Get original wind speed from the graph
        original_U_free = unscaler.inverse_scale_graph(graphs).globals[0][0]

        # Run model ONCE at the original wind speed
        prediction = pred_fn(
            model_params,
            graphs,
            probe_graphs,
            jnp.atleast_2d(wt_mask),
            jnp.atleast_2d(probe_mask),
        ).squeeze()

        # Get unscaled predictions and targets at original wind speed
        wt_predictions_original = np.array(_inverse_scale_target(prediction[wt_idx]).squeeze())
        wt_targets_original = np.array(_inverse_scale_target(np.array(targets)[wt_idx]).squeeze())

        # Collect scaled predictions and targets for all wind speeds
        plot_data = {}
        global_vmin = np.inf
        global_vmax = -np.inf

        for U in U_free_values:
            # Scale predictions and targets by wind speed ratio
            scale_factor = U / original_U_free
            wt_predictions = wt_predictions_original * scale_factor
            wt_targets = wt_targets_original * scale_factor

            plot_data[U] = {
                "positions_D": wt_positions_D,
                "predictions": wt_predictions,
                "targets": wt_targets,
            }

            # Update global limits for consistent colorbar
            global_vmin = min(global_vmin, wt_predictions.min(), wt_targets.min())
            global_vmax = max(global_vmax, wt_predictions.max(), wt_targets.max())

        # Create figure: 2 columns (predictions | targets) x N rows (wind speeds)
        n_rows = len(U_free_values)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, 2)

        # Calculate bounds for this layout
        padding = 8
        x_lim = (
            wt_positions_D[:, 0].min() - padding,
            wt_positions_D[:, 0].max() + padding,
        )
        y_lim = (
            wt_positions_D[:, 1].min() - padding,
            wt_positions_D[:, 1].max() + padding,
        )

        sc = None
        for row, U in enumerate(U_free_values):
            data = plot_data[U]
            positions = data["positions_D"]
            predictions = data["predictions"]
            targets = data["targets"]

            # Plot predictions
            ax_pred = axes[row, 0]
            sc = ax_pred.scatter(
                positions[:, 0],
                positions[:, 1],
                c=predictions,
                cmap="viridis",
                vmin=global_vmin,
                vmax=global_vmax,
                s=30,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
            )
            ax_pred.set_xlim(x_lim)
            ax_pred.set_ylim(y_lim)
            ax_pred.set_aspect("equal")
            ax_pred.set_ylabel(f"$U = {U}$ m/s\n$y/D$ [-]")

            if row == 0:
                ax_pred.set_title("Predictions", fontsize=11)
            if row == n_rows - 1:
                ax_pred.set_xlabel("$x/D$ [-]")
            else:
                ax_pred.set_xticklabels([])

            # Plot targets
            ax_tgt = axes[row, 1]
            ax_tgt.scatter(
                positions[:, 0],
                positions[:, 1],
                c=targets,
                cmap="viridis",
                vmin=global_vmin,
                vmax=global_vmax,
                s=30,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
            )
            ax_tgt.set_xlim(x_lim)
            ax_tgt.set_ylim(y_lim)
            ax_tgt.set_aspect("equal")
            ax_tgt.set_yticklabels([])

            if row == 0:
                ax_tgt.set_title("Targets", fontsize=11)
            if row == n_rows - 1:
                ax_tgt.set_xlabel("$x/D$ [-]")
            else:
                ax_tgt.set_xticklabels([])

        # Add colorbar
        cbar = fig.colorbar(
            sc,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            shrink=0.6,
            pad=0.08,
            aspect=40,
        )
        cbar.set_label(r"$u$ [m/s]")

        fig.suptitle(f"Debug: {layout_name}", fontsize=12, y=1.02)

        # Save figure
        safe_name = layout_name.replace(" ", "_")
        output_file = os.path.join(output_path, f"debug_predictions_targets_{safe_name}.pdf")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()


# =============================================================================
# DATA RECONSTRUCTION AND PREDICTION GENERATION
# =============================================================================


def dict_to_graph(graph_dict):
    """Convert a serialized dictionary back to a jraph GraphsTuple."""
    import jraph

    return jraph.GraphsTuple(
        nodes=graph_dict["nodes"],
        edges=graph_dict["edges"],
        receivers=graph_dict["receivers"],
        senders=graph_dict["senders"],
        globals=graph_dict["globals"],
        n_node=graph_dict["n_node"],
        n_edge=graph_dict["n_edge"],
    )


def generate_crossstream_predictions_local(
    raw_data: dict,
    model,
    model_params,
    unscaler,
    x_downstream: list[int],
    U_free: list[float],
    TI_flow: float = 0.05,
    grid_density: int = 3,
) -> dict:
    """Generate crossstream prediction data locally from raw graph data."""
    import jax
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.run_pywake import construct_on_the_fly_probe_graph

    wt = DTU10MW()
    D = wt.diameter()
    scale_stats = raw_data["scale_stats"]
    plot_distance = raw_data["metadata"]["plot_distance"]

    pred_fn = jax.jit(model.apply)
    _inverse_scale_target = unscaler.inverse_scale_output

    # Calculate y probe positions
    y_plot_range = plot_distance * 1.0 * scale_stats["distance"]["range"]
    # grid_density is points per rotor diameter, not per meter
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
        raw_data["layout_data"].items(), desc="Generating predictions"
    ):
        # wt_positions in raw data contains ALL node positions - filter to only WTs
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

                jraph_graph_gen, jraph_probe_graphs_gen, node_array_tuple_gen = (
                    construct_on_the_fly_probe_graph(
                        positions=wt_positions,
                        U=[U_flow],
                        TI=[TI_flow],
                        grid=grid,
                        scale_stats=scale_stats,
                        return_positions=True,
                    )
                )
                targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_array_tuple_gen

                prediction = pred_fn(
                    model_params,
                    jraph_graph_gen,
                    jraph_probe_graphs_gen,
                    jnp.atleast_2d(wt_mask_gen).T,
                    jnp.atleast_2d(probe_mask_gen).T,
                ).squeeze()

                unscaled_predictions = _inverse_scale_target(apply_mask(prediction, probe_mask_gen))
                unscaled_targets = _inverse_scale_target(
                    apply_mask(targets_gen.squeeze(), probe_mask_gen)
                )

                # Compute error metrics for CSV export
                # Notation: u = target (ground truth), û = prediction, U = freestream
                u = np.array(unscaled_targets)  # target velocity
                u_hat = np.array(unscaled_predictions)  # predicted velocity
                U = U_flow  # freestream velocity

                error_abs = np.abs(u - u_hat)  # |u - û| in m/s
                freestream_norm_error = error_abs / U  # |u - û| / U [-]
                percentage_error = (error_abs / np.abs(u)) * 100  # |u - û| / |u| [%]

                # Velocity deficit normalized error: |u - û| / |U - u|
                # This normalizes by the target (true) velocity deficit
                target_deficit = np.abs(U - u)
                # Avoid division by zero where target equals freestream
                deficit_norm_error = (
                    np.where(
                        target_deficit > 1e-6,
                        error_abs / target_deficit,
                        np.nan,
                    )
                    * 100
                )  # [%]

                # Diagnostic metrics
                prediction_error = u - u_hat  # Signed error (u - û) in m/s

                # Average deficit relative error (scalar)
                # Compares bulk wake predictions vs targets
                pred_deficit = U - u_hat  # predicted velocity deficit
                avg_pred_deficit = np.mean(pred_deficit)
                avg_target_deficit = np.mean(target_deficit)  # target_deficit already computed

                # Relative error between averaged deficits [%]
                avg_deficit_rel_error = (
                    np.abs(avg_pred_deficit - avg_target_deficit) / avg_target_deficit * 100
                    if avg_target_deficit > 1e-6
                    else float("nan")
                )

                # Average velocity relative error (scalar)
                # Compares average effective wind speeds
                avg_u = np.mean(u)
                avg_u_hat = np.mean(u_hat)

                # Relative error between averaged velocities [%]
                avg_velocity_rel_error = (
                    np.abs(avg_u - avg_u_hat) / avg_u * 100 if avg_u > 1e-6 else float("nan")
                )

                # Velocity relative error (point-by-point array)
                # |u - û| / u normalized by target velocity
                velocity_rel_error = np.where(
                    u > 1e-6,
                    error_abs / u * 100,
                    np.nan,
                )  # [%]

                # Filtered deficit-normalized (only where deficit > 5% of freestream)
                # This excludes wake edges where tiny deficits cause metric explosion
                min_meaningful_deficit = 0.05 * U
                deficit_norm_filtered = np.where(
                    target_deficit > min_meaningful_deficit,
                    (error_abs / target_deficit) * 100,
                    np.nan,
                )

                # Helper to get y/D at max value (handles NaN)
                y_D = y / D

                def get_max_y_D(arr):
                    """Get y/D position where array has its maximum value."""
                    if np.all(np.isnan(arr)):
                        return float("nan")
                    idx = np.nanargmax(arr)
                    return float(y_D[idx])

                norm_predictions, norm_targets, _ = apply_normalizations(
                    np.array(unscaled_predictions),
                    np.array(unscaled_targets),
                    U_flow,
                    plot_velocity_deficit=True,
                    normalize_by_U=True,
                )

                current_min = min(norm_predictions.min(), norm_targets.min())
                current_max = max(norm_predictions.max(), norm_targets.max())
                if global_min is None or current_min < global_min:
                    global_min = current_min
                if global_max is None or current_max > global_max:
                    global_max = current_max

                if not graph_processed:
                    unscaled_node_positions = (
                        unscaler.inverse_scale_trunk_input(node_positions_gen) / D
                    )
                    unscaled_graphs_gen = unscaler.inverse_scale_graph(jraph_graph_gen)
                    unscaled_probe_graphs_gen = unscaler.inverse_scale_graph(jraph_probe_graphs_gen)
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
                    "normalized_predictions": norm_predictions,
                    "normalized_targets": norm_targets,
                    "y/D": y / D,
                    "unscaled_node_positions": np.array(unscaled_node_positions),
                    "unscaled_graphs": unscaled_graphs_gen,
                    "unscaled_probe_graphs": unscaled_probe_graphs_gen,
                    "errors": {
                        "freestream_normalized": {
                            "min": float(np.min(freestream_norm_error)),
                            "max": float(np.max(freestream_norm_error)),
                            "mean": float(np.mean(freestream_norm_error)),
                            "std": float(np.std(freestream_norm_error)),
                            "max_y_D": get_max_y_D(freestream_norm_error),
                        },
                        "percentage": {
                            "min": float(np.nanmin(percentage_error)),
                            "max": float(np.nanmax(percentage_error)),
                            "mean": float(np.nanmean(percentage_error)),
                            "std": float(np.nanstd(percentage_error)),
                            "max_y_D": get_max_y_D(percentage_error),
                        },
                        "deficit_normalized": {
                            "min": float(np.nanmin(deficit_norm_error)),
                            "max": float(np.nanmax(deficit_norm_error)),
                            "mean": float(np.nanmean(deficit_norm_error)),
                            "std": float(np.nanstd(deficit_norm_error)),
                            "max_y_D": get_max_y_D(deficit_norm_error),
                        },
                        "target_deficit_ms": {  # U - u in m/s (velocity deficit)
                            "min": float(np.min(target_deficit)),
                            "max": float(np.max(target_deficit)),
                            "mean": float(np.mean(target_deficit)),
                            "std": float(np.std(target_deficit)),
                            "max_y_D": get_max_y_D(target_deficit),
                        },
                        "prediction_error_ms": {  # u - û in m/s (signed error)
                            "min": float(np.min(prediction_error)),
                            "max": float(np.max(prediction_error)),
                            "mean": float(np.mean(prediction_error)),
                            "std": float(np.std(prediction_error)),
                            "max_y_D": get_max_y_D(prediction_error),
                        },
                        "deficit_normalized_filtered": {  # Only where deficit > 5% of U
                            "min": float(np.nanmin(deficit_norm_filtered)),
                            "max": float(np.nanmax(deficit_norm_filtered)),
                            "mean": float(np.nanmean(deficit_norm_filtered)),
                            "std": float(np.nanstd(deficit_norm_filtered)),
                            "max_y_D": get_max_y_D(deficit_norm_filtered),
                        },
                        "avg_deficit_rel_error": {  # |mean(U-û) - mean(U-u)| / mean(U-u) [%]
                            "min": float(avg_deficit_rel_error),
                            "max": float(avg_deficit_rel_error),
                            "mean": float(avg_deficit_rel_error),
                            "std": 0.0,  # Single value, no spread
                            "max_y_D": float("nan"),  # Not applicable (scalar metric)
                        },
                        "avg_velocity_rel_error": {  # |mean(u) - mean(û)| / mean(u) [%]
                            "min": float(avg_velocity_rel_error),
                            "max": float(avg_velocity_rel_error),
                            "mean": float(avg_velocity_rel_error),
                            "std": 0.0,  # Single value, no spread
                            "max_y_D": float("nan"),  # Not applicable (scalar metric)
                        },
                        "velocity_rel_error": {  # |u - û| / u [%] point-by-point
                            "min": float(np.nanmin(velocity_rel_error)),
                            "max": float(np.nanmax(velocity_rel_error)),
                            "mean": float(np.nanmean(velocity_rel_error)),
                            "std": float(np.nanstd(velocity_rel_error)),
                            "max_y_D": get_max_y_D(velocity_rel_error),
                        },
                    },
                }

        result["layout_data"][layout_name] = layout_result

    pad_size = (global_max - global_min) * 0.1
    result["global_limits"]["velocity_range"] = [
        global_min - pad_size,
        global_max + pad_size,
    ]

    return result


def generate_wt_errors_local(
    raw_data: dict,
    model,
    model_params,
    unscaler,
    U_free_values: list[float],
    TI_flow: float = 0.05,
) -> dict:
    """Generate wind turbine spatial error data locally from raw graph data.

    Uses PyWake to regenerate graphs and targets at each wind speed for
    physically meaningful error calculations (not linear scaling).

    Args:
        raw_data: Raw cached data containing layout_data and scale_stats
        model: The trained model
        model_params: Model parameters
        unscaler: Unscaler for inverse transformations
        U_free_values: List of wind speeds to evaluate
        TI_flow: Turbulence intensity for PyWake simulations (default 0.05 = 5%)

    Returns:
        Dictionary with layout errors and aggregated statistics
    """
    import jax
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.run_pywake import construct_on_the_fly_probe_graph

    wt = DTU10MW()
    D = wt.diameter()
    scale_stats = raw_data["scale_stats"]

    pred_fn = jax.jit(model.apply)
    _inverse_scale_target = unscaler.inverse_scale_output

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
        # Extract WT positions from layout_data
        node_positions = layout_data["node_positions"]
        wt_mask = layout_data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]

        # Get WT positions in meters
        wt_positions_m = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
        wt_positions_D = wt_positions_m / D

        # Create minimal grid for WT-only predictions
        grid = HorizontalGrid(
            x=wt_positions_m[:, 0][:1],
            y=wt_positions_m[:, 1][:1],
            h=wt.hub_height(),
        )

        # Run model at EACH wind speed using PyWake regeneration
        wt_rel_errors_list = []
        for U_flow in U_free_values:
            # Regenerate graph with PyWake at this wind speed
            jraph_graph, jraph_probe_graph, node_array_tuple = construct_on_the_fly_probe_graph(
                positions=wt_positions_m,
                U=[U_flow],
                TI=[TI_flow],
                grid=grid,
                scale_stats=scale_stats,
                return_positions=True,
            )
            targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_array_tuple

            # Run model prediction
            prediction = pred_fn(
                model_params,
                jraph_graph,
                jraph_probe_graph,
                jnp.atleast_2d(wt_mask_gen).T,
                jnp.atleast_2d(probe_mask_gen).T,
            ).squeeze()

            # Extract WT predictions and targets
            wt_idx_gen = np.where(wt_mask_gen != 0)[0]
            wt_predictions = np.array(_inverse_scale_target(prediction[wt_idx_gen]).squeeze())
            wt_targets = np.array(
                _inverse_scale_target(np.array(targets_gen)[wt_idx_gen]).squeeze()
            )

            # Compute signed relative error at this wind speed
            wt_errors = wt_targets - wt_predictions
            wt_rel_errors = (wt_errors / U_flow) * 100
            wt_rel_errors_list.append(wt_rel_errors)

        wt_rel_errors_arr = np.array(wt_rel_errors_list)
        aggregated_errors = {
            "mean": np.mean(wt_rel_errors_arr, axis=0),
            "std": np.std(wt_rel_errors_arr, axis=0),
            "abs_max": np.max(np.abs(wt_rel_errors_arr), axis=0),
        }

        for metric in ["mean", "std", "abs_max"]:
            agg_stats[metric]["min"] = min(
                agg_stats[metric]["min"], aggregated_errors[metric].min()
            )
            agg_stats[metric]["max"] = max(
                agg_stats[metric]["max"], aggregated_errors[metric].max()
            )

        result["layout_data"][layout_name] = {
            "wt_positions_D": np.array(wt_positions_D),
            "aggregated_errors": aggregated_errors,
            "per_wind_speed_errors": {
                U_free_values[i]: wt_rel_errors_list[i] for i in range(len(U_free_values))
            },
            "n_wts": int(np.sum(wt_mask)),
        }

    result["aggregated_extremas"] = {
        metric: (agg_stats[metric]["min"], agg_stats[metric]["max"])
        for metric in ["mean", "std", "abs_max"]
    }

    return result


def generate_wt_errors_per_windspeed(
    raw_data: dict,
    model,
    model_params,
    unscaler,
    U_free_values: list[float],
) -> dict:
    """Generate WT spatial error data using actual per-wind-speed test cases.

    Unlike generate_wt_errors_local which uses linear scaling, this function
    uses actual test cases at each wind speed for true per-wind-speed errors.

    Args:
        raw_data: Raw cached data containing 'per_windspeed_data' key
        model: The trained model
        model_params: Model parameters
        unscaler: Unscaler for inverse transformations
        U_free_values: List of wind speeds to include (must match cached data)

    Returns:
        Dictionary with per-wind-speed error data for plotting
    """
    import jax
    from jax import numpy as jnp
    from py_wake.examples.data.dtu10mw import DTU10MW

    # Check if per-windspeed data is available
    if "per_windspeed_data" not in raw_data:
        raise ValueError(
            "Per-windspeed data not found in cache. "
            "Please regenerate cache using updated generate_plot_data.py on Sophia."
        )

    per_ws_data = raw_data["per_windspeed_data"]
    cached_windspeeds = per_ws_data["target_windspeeds"]

    # Verify requested wind speeds are in cache
    for U in U_free_values:
        if U not in cached_windspeeds:
            raise ValueError(
                f"Wind speed {U} m/s not in cached data. " f"Available: {cached_windspeeds}"
            )

    wt = DTU10MW()
    D = wt.diameter()

    pred_fn = jax.jit(model.apply)
    _inverse_scale_target = unscaler.inverse_scale_output

    result = {
        "metadata": {
            "U_free_values": U_free_values,
            "model_type_str": "best_model",
            "data_source": "per_windspeed",  # Flag to indicate real per-WS data
        },
        "aggregated_extremas": {},
        "layout_data": {},
    }

    agg_stats = {
        "mean": {"min": 1e9, "max": -1e9},
        "std": {"min": 1e9, "max": -1e9},
        "abs_max": {"min": 1e9, "max": -1e9},
    }

    layout_names = ["cluster", "single string", "multiple string", "parallel string"]

    for layout_name in tqdm(layout_names, desc="Calculating WT errors (per-WS)"):
        ws_data = per_ws_data["layout_data"].get(layout_name, {})

        wt_rel_errors_dict = {}
        wt_positions_D_dict = {}

        for U in U_free_values:
            if U not in ws_data:
                print(f"  Warning: No data for {layout_name} at U={U} m/s, skipping")
                continue

            sample_data = ws_data[U]
            graphs = dict_to_graph(sample_data["graphs"])
            targets = sample_data["targets"]
            wt_mask = sample_data["wt_mask"]
            probe_mask = sample_data.get("probe_mask", wt_mask)  # Use wt_mask if no probe
            node_positions = sample_data["node_positions"]

            wt_idx = np.where(wt_mask != 0)[0]

            # Get WT positions for this wind speed sample
            wt_positions = unscaler.inverse_scale_trunk_input(node_positions)[wt_idx]
            wt_positions_D = wt_positions / D

            # Run prediction
            # For WT-only graphs, probe_graphs may be the same as graphs
            prediction = pred_fn(
                model_params,
                graphs,
                graphs,  # Use same graph for probe (WT nodes only)
                jnp.atleast_2d(wt_mask),
                jnp.atleast_2d(wt_mask),
            ).squeeze()

            # Get predictions and targets for WT nodes
            wt_predictions = _inverse_scale_target(prediction[wt_idx]).squeeze()
            wt_targets = _inverse_scale_target(np.array(targets)[wt_idx]).squeeze()

            # Calculate errors
            wt_errors = np.array(wt_targets) - np.array(wt_predictions)
            wt_rel_errors = (wt_errors / U) * 100  # Relative error in %

            wt_rel_errors_dict[U] = wt_rel_errors
            wt_positions_D_dict[U] = wt_positions_D

            # Track global extrema for colorbar
            agg_stats["abs_max"]["min"] = min(
                agg_stats["abs_max"]["min"], np.abs(wt_rel_errors).min()
            )
            agg_stats["abs_max"]["max"] = max(
                agg_stats["abs_max"]["max"], np.abs(wt_rel_errors).max()
            )

        if not wt_rel_errors_dict:
            print(f"  Warning: No valid data for {layout_name}, skipping")
            continue

        # For per-windspeed data, we store positions per wind speed since
        # different samples may have different wind turbine counts
        result["layout_data"][layout_name] = {
            "wt_positions_D": wt_positions_D_dict,  # Dict: {U: positions}
            "per_wind_speed_errors": wt_rel_errors_dict,
            "n_wts": {U: len(errors) for U, errors in wt_rel_errors_dict.items()},
        }

    result["aggregated_extremas"] = {
        metric: (agg_stats[metric]["min"], agg_stats[metric]["max"])
        for metric in ["mean", "std", "abs_max"]
    }

    return result


# =============================================================================
# MAIN - Generate figures from cached raw data
# =============================================================================


def main():
    """Generate publication figures from cached raw graph data."""
    script_dir = Path(__file__).parent
    cache_dir = script_dir / "cache"
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    raw_cache = cache_dir / "raw_graph_data.pkl"

    if not raw_cache.exists():
        print("Raw graph data cache not found!")
        print(f"Expected: {raw_cache}")
        print("\nPlease run generate_plot_data.py on Sophia first, then copy cache file.")
        return

    # Load raw cached data
    print("Loading cached raw graph data...")
    with open(raw_cache, "rb") as f:
        raw_data = pickle.load(f)

    # Load model locally
    print("\nLoading model...")
    main_path = "./assets/best_model_Vj8"
    model_params, cfg, model, _, _ = load_article1_model(main_path)

    # Setup unscaler
    from utils.data_tools import setup_unscaler

    scale_stats = raw_data["scale_stats"]
    unscaler = setup_unscaler(cfg, scale_stats=scale_stats)

    # Parameters
    x_downstream = [25, 50, 100]
    U_free = [6, 12, 18]
    TI_flow = 0.05

    # Generate predictions locally
    print("\nGenerating crossstream predictions...")
    crossstream_data = generate_crossstream_predictions_local(
        raw_data,
        model,
        model_params,
        unscaler,
        x_downstream=x_downstream,
        U_free=U_free,
        TI_flow=TI_flow,
    )

    # Generate WT error data for the aggregated figure
    print("\nGenerating WT error data (aggregated)...")
    wt_errors_aggregated = generate_wt_errors_local(
        raw_data,
        model,
        model_params,
        unscaler,
        U_free_values=U_free,
    )

    model_type_str = "best_model"

    # Generate figures
    print("\nGenerating crossstream profiles figure...")
    plot_crossstream_figure(crossstream_data, str(output_dir), model_type_str)

    # Export error statistics to CSV
    export_crossstream_errors_csv(crossstream_data, str(output_dir))

    # Export deficit threshold sensitivity analysis
    print("\nGenerating deficit threshold sweep analysis...")
    export_deficit_threshold_sweep(crossstream_data, str(output_dir))

    print("\nGenerating WT spatial errors figure (aggregated)...")
    plot_wt_spatial_errors_figure(wt_errors_aggregated, str(output_dir), model_type_str)

    # Generate per-wind-speed WT spatial errors figure
    print("\nGenerating per-wind-speed WT spatial errors figure...")
    plot_wt_spatial_errors_per_windspeed(
        raw_data,
        model,
        model_params,
        unscaler,
        str(output_dir),
        U_free_values=U_free,
        prototype=True,
    )

    # Generate debug plots (predictions vs targets)
    print("\nGenerating debug prediction/target plots...")
    plot_debug_predictions_targets(
        raw_data,
        model,
        model_params,
        unscaler,
        str(output_dir),
        U_free_values=U_free,
    )

    # Export crossstream data for companion repo
    companion_export = Path("/home/jpsch/code/rans-awf-cae-vs-gnn/data/crossstream_data.pkl")
    export_crossstream_data(crossstream_data, companion_export)

    print("\nDone! Figures saved to:", output_dir)


def export_crossstream_data(crossstream_data: dict, output_path: Path) -> None:
    """Export crossstream data as a self-contained pickle (no jraph/JAX objects).

    Converts jraph GraphsTuple objects to plain dicts with numpy arrays so the
    companion plotting repo can load them without GNO dependencies.
    """

    def _graphs_to_dict(g) -> dict:
        """Convert a jraph-like GraphsTuple to a plain dict."""
        return {
            "senders": np.asarray(g.senders),
            "receivers": np.asarray(g.receivers),
            "edges": np.asarray(g.edges),
        }

    def _probe_graphs_to_dict(g) -> dict:
        return {
            "senders": np.asarray(g.senders),
            "receivers": np.asarray(g.receivers),
        }

    def _to_python(v):
        """Convert any array-like scalar to a plain Python type."""
        if hasattr(v, "item"):
            return v.item()
        return v

    export = {
        "metadata": crossstream_data["metadata"],
        "global_limits": {k: _to_python(v) for k, v in crossstream_data["global_limits"].items()},
        "layout_data": {},
    }

    for layout_name, layout in crossstream_data["layout_data"].items():
        export_layout = {
            "wt_mask": np.asarray(layout["wt_mask"]),
            "wt_spacing": _to_python(layout["wt_spacing"]),
            "predictions": {},
        }
        for x_sel, by_x in layout["predictions"].items():
            export_layout["predictions"][x_sel] = {}
            for U_flow, pred in by_x.items():
                export_layout["predictions"][x_sel][U_flow] = {
                    "normalized_predictions": np.asarray(pred["normalized_predictions"]),
                    "normalized_targets": np.asarray(pred["normalized_targets"]),
                    "y/D": np.asarray(pred["y/D"]),
                    "unscaled_node_positions": np.asarray(pred["unscaled_node_positions"]),
                    "unscaled_graphs": _graphs_to_dict(pred["unscaled_graphs"]),
                    "unscaled_probe_graphs": _probe_graphs_to_dict(
                        pred["unscaled_probe_graphs"]
                    ),
                }
        export["layout_data"][layout_name] = export_layout

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(export, f)
    print(f"Exported crossstream data to {output_path}")


if __name__ == "__main__":
    main()
