"""
AWF data extraction script for Article 2 publication figures.

Extracts raw graph data and NetCDF flow fields from the AWF RANS dataset.
This is a separate script (not a parameterization of generate_plot_data.py) because
the AWF data flow is fundamentally different:
  - Ground truth comes from NetCDF (awf_database.nc), not PyWake
  - AWF graphs lack layout_type/wt_spacing metadata
  - Crossstream profiles are extracted from the raw 2D velocity field

Usage:
    python Experiments/article_2/generate_awf_plot_data.py [--force]

Outputs:
    - Experiments/article_2/cache/publication_figures/awf_graphs/raw_graph_data.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from article2_utils import (  # noqa: E402
    AWF_TURBINE_DIAMETER,
    MODELS,
    convert_graph_to_serializable,
    get_awf_database_path,
    get_awf_test_path,
)

from utils.data_tools import retrieve_dataset_stats, setup_test_val_iterator  # noqa: E402
from utils.torch_loader import Torch_Geomtric_Dataset  # noqa: E402

# =============================================================================
# AWF Layout Selection
# =============================================================================


def select_awf_representative_layouts(dataset, n_layouts=4) -> dict[str, int]:
    """Select AWF test layouts with diverse turbine counts.

    Since AWF graphs lack layout_type/wt_spacing metadata, selects layouts
    spanning min/median/max n_wt range for visual diversity.

    Args:
        dataset: Torch_Geomtric_Dataset with AWF test graphs
        n_layouts: Number of layouts to select (default 4)

    Returns:
        Dict mapping descriptive names to dataset indices (first flowcase of each layout)
    """
    # Collect (idx, n_wt) for all test samples
    # AWF has 4 flowcases per layout; use first flowcase (idx % 4 == 0) per layout
    layout_info = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        n_wt = int(data.n_wt) if hasattr(data, "n_wt") else int(data.pos.shape[0])
        layout_info.append({"idx": idx, "n_wt": n_wt})

    # Group by layout (consecutive groups of 4 flowcases)
    layouts = []
    for i in range(0, len(layout_info), 4):
        layouts.append({"first_idx": i, "n_wt": layout_info[i]["n_wt"]})

    if len(layouts) < n_layouts:
        print(f"  Warning: Only {len(layouts)} layouts available, requested {n_layouts}")
        n_layouts = len(layouts)

    # Sort by n_wt and pick diverse sizes
    layouts_sorted = sorted(layouts, key=lambda x: x["n_wt"])
    n_total = len(layouts_sorted)

    # Pick: smallest, ~33%, ~66%, largest
    pick_indices = [
        0,
        n_total // 3,
        2 * n_total // 3,
        n_total - 1,
    ][:n_layouts]
    # Ensure unique picks
    pick_indices = list(dict.fromkeys(pick_indices))

    names = ["small_farm", "medium_farm", "large_farm", "xlarge_farm"]
    result = {}
    for i, pick_idx in enumerate(pick_indices):
        layout = layouts_sorted[pick_idx]
        name = names[i] if i < len(names) else f"farm_{i}"
        result[name] = layout["first_idx"]
        print(f"  {name}: idx={layout['first_idx']}, n_wt={layout['n_wt']}")

    return result


# =============================================================================
# NetCDF Flow Field Extraction
# =============================================================================


def get_layout_idx_from_test_idx(dataset, test_idx: int) -> int:
    """Extract the layout index from the test dataset zip filename.

    AWF test graphs come from zips named _layout{N}.zip. The test_idx is the
    index into the flattened test dataset (each layout has 4 flowcases).
    """
    # For preprocessed test data, files are named as sequential indices.
    # The layout mapping is determined during preprocessing. Since we can't
    # trivially recover the original layout index from the preprocessed files,
    # we use position-based matching via match_graph_to_netcdf_layout().
    return -1  # Sentinel indicating we need position-based matching


def match_graph_to_netcdf_layout(
    dataset,
    test_idx: int,
    nc_dataset: xr.Dataset,
    scale_stats: dict,
    D: float = AWF_TURBINE_DIAMETER,
    tolerance_D: float = 0.5,
) -> tuple[int, int]:
    """Find the NetCDF layout and flowcase indices matching a test graph.

    Preprocessed graph positions are scaled (pos / distance_range with run4).
    This function unscales them back to meters, then to diameters for comparison
    with NetCDF positions (stored in diameters).

    Returns:
        (layout_idx, flowcase_idx) tuple
    """
    data = dataset[test_idx]
    n_wt = int(data.n_wt) if hasattr(data, "n_wt") else int(data.pos.shape[0])

    # Unscale: preprocessed pos = raw_meters / distance_range (run4: min=0)
    dist_range = scale_stats["distance"]["range"]
    if isinstance(dist_range, list):
        dist_range = dist_range[0]
    graph_pos_m = data.pos.numpy() * dist_range
    graph_pos_D = graph_pos_m / D

    n_layouts = len(nc_dataset.layout)
    n_flowcases = len(nc_dataset.flowcase)

    # Pre-filter: build n_wt index for fast lookup
    for layout_idx in range(n_layouts):
        for flowcase_idx in range(n_flowcases):
            fc = nc_dataset.isel(layout=layout_idx, flowcase=flowcase_idx)
            nc_nwt = int(fc.Nwt.values)
            if nc_nwt != n_wt:
                continue

            nc_pos = np.c_[fc.wt_x.values[:nc_nwt], fc.wt_y.values[:nc_nwt]]
            if nc_pos.shape != graph_pos_D.shape:
                continue

            # Sort both by x then y for consistent comparison
            graph_sorted = graph_pos_D[np.lexsort((graph_pos_D[:, 1], graph_pos_D[:, 0]))]
            nc_sorted = nc_pos[np.lexsort((nc_pos[:, 1], nc_pos[:, 0]))]

            max_diff = np.max(np.abs(graph_sorted - nc_sorted))
            if max_diff < tolerance_D:
                return layout_idx, flowcase_idx

    raise ValueError(f"No NetCDF match found for test_idx={test_idx} (n_wt={n_wt})")


def extract_raw_flow_field(
    nc_dataset: xr.Dataset,
    layout_idx: int,
    flowcase_idx: int,
) -> dict:
    """Extract raw RANS flow field from NetCDF for one flowcase.

    Args:
        nc_dataset: Opened xarray Dataset
        layout_idx: Layout dimension index
        flowcase_idx: Flowcase dimension index

    Returns:
        Dict with denormalized flow field data
    """
    fc = nc_dataset.isel(layout=layout_idx, flowcase=flowcase_idx)

    nwt = int(fc.Nwt.values)
    ws_inf = float(fc.ws_inf.values)

    return {
        "U_field": fc.U.values * ws_inf,  # Denormalized: (n_x, n_y) in m/s
        "x_grid_D": fc.x.values,
        "y_grid_D": fc.y.values,
        "wt_x_D": fc.wt_x.values[:nwt],
        "wt_y_D": fc.wt_y.values[:nwt],
        "ws_inf": ws_inf,
        "ti_inf": float(fc.ti_inf.values),
        "nwt": nwt,
        "wseff": nc_dataset.lut_wseff.isel(layout=layout_idx, flowcase=flowcase_idx).values[:nwt],
    }


# =============================================================================
# Crossstream Profile Extraction
# =============================================================================


def extract_crossstream_from_field(flow_field: dict, x_downstream_D: list[int]) -> dict:
    """Extract crossstream profiles from raw 2D RANS field.

    Args:
        flow_field: Dict from extract_raw_flow_field()
        x_downstream_D: List of downstream distances in D from last WT

    Returns:
        Dict mapping x_D -> {y_D, U, actual_x_D}
    """
    x_max_wt_D = flow_field["wt_x_D"].max()

    profiles = {}
    for x_D in x_downstream_D:
        target_x_D = x_max_wt_D + x_D
        # Find nearest x grid index
        x_idx = np.argmin(np.abs(flow_field["x_grid_D"] - target_x_D))
        actual_x_D = float(flow_field["x_grid_D"][x_idx])

        profiles[x_D] = {
            "y_D": flow_field["y_grid_D"].copy(),
            "U": flow_field["U_field"][x_idx, :].copy(),  # Velocity along crossstream line
            "actual_x_D": actual_x_D,
        }
    return profiles


# =============================================================================
# Verification
# =============================================================================


def verify_graph_netcdf_match(
    dataset,
    test_idx: int,
    flow_field: dict,
    scale_stats: dict,
    D: float = AWF_TURBINE_DIAMETER,
    tolerance: float = 0.05,
) -> dict:
    """Verify graph targets match NetCDF velocities at probe positions.

    Preprocessed graphs have scaled features (run4: value / range).
    This function unscales them before comparing with raw NetCDF values.

    Returns:
        Dict with max_error, mean_error, passed
    """
    data = dataset[test_idx]

    # Graph probe positions are in trunk_inputs (SCALED: pos / distance_range)
    if hasattr(data, "trunk_inputs"):
        probe_pos_scaled = data.trunk_inputs.numpy()
        if probe_pos_scaled.ndim == 3:
            probe_pos_scaled = probe_pos_scaled.squeeze(0)
    else:
        return {"max_error": float("nan"), "mean_error": float("nan"), "passed": False}

    # Unscale positions: run4 → pos_m = pos_scaled * distance_range
    dist_range = scale_stats["distance"]["range"]
    if isinstance(dist_range, list):
        dist_range = dist_range[0]
    probe_pos_m = probe_pos_scaled * dist_range

    # Graph probe velocities (SCALED: vel / velocity_range)
    graph_vel_scaled = data.output_features.numpy().flatten()
    if data.output_features.ndim == 3:
        graph_vel_scaled = data.output_features.squeeze(0).numpy().flatten()

    # Unscale velocities: run4 → vel_ms = vel_scaled * velocity_range
    vel_range = scale_stats["velocity"]["range"]
    if isinstance(vel_range, list):
        vel_range = vel_range[0]
    graph_vel = graph_vel_scaled * vel_range

    # Convert probe positions to diameters
    probe_pos_D = probe_pos_m / D

    # Look up nearest grid point in NetCDF for each probe
    x_grid = flow_field["x_grid_D"]
    y_grid = flow_field["y_grid_D"]
    U_field = flow_field["U_field"]

    n_probes = min(len(probe_pos_D), len(graph_vel))
    nc_vel = np.zeros(n_probes)

    for i in range(n_probes):
        xi = np.argmin(np.abs(x_grid - probe_pos_D[i, 0]))
        yi = np.argmin(np.abs(y_grid - probe_pos_D[i, 1]))
        nc_vel[i] = U_field[xi, yi]

    errors = np.abs(graph_vel[:n_probes] - nc_vel)
    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    passed = max_error < tolerance

    return {"max_error": max_error, "mean_error": mean_error, "passed": passed}


# =============================================================================
# Graph Data Setup
# =============================================================================


def setup_awf_plot_iterator(cfg, test_data_path: str, dataset, layout_idxs: dict):
    """Setup iterator and retrieve data for specified layout indices.

    For AWF, each layout has 4 flowcases. layout_idxs maps to the first
    flowcase of each layout. We load all 4 flowcases per layout.
    """
    # AWF graphs lack layout_type/wt_spacing → use return_layout_info=False
    get_plot_data_iterator, test_dataset, _, _ = setup_test_val_iterator(
        cfg,
        type_str="test",
        return_idxs=True,
        return_positions=True,
        path=test_data_path,
        cache=False,
        return_layout_info=False,
        dataset=dataset,
        num_workers=0,
    )

    # Collect all indices we need (all 4 flowcases per layout)
    needed_idxs = set()
    for base_idx in layout_idxs.values():
        for fc in range(4):
            needed_idxs.add(base_idx + fc)

    iterator = get_plot_data_iterator()
    plot_graphs = {}

    for i, data_in in tqdm(enumerate(iterator), desc="Loading AWF plot data"):
        if i in needed_idxs:
            # Wrap in 5-tuple matching TurbOPark format for consistent unpacking
            # (graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing)
            graphs, probe_graphs, node_array_tuple = data_in
            plot_graphs[i] = (graphs, probe_graphs, node_array_tuple, "awf", 0.0)

    return plot_graphs, test_dataset


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract AWF graph data for Article 2 publication figures"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    model_cfg = MODELS["L1_L3_phase2"]
    cache_dir = script_dir / "cache" / "publication_figures" / model_cfg.dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = cache_dir / "raw_graph_data.pkl"

    if not args.force and raw_cache.exists():
        print("Cache file already exists. Use --force to regenerate.")
        return

    # Get paths
    test_data_path = get_awf_test_path()
    nc_path = get_awf_database_path()
    print(f"Test data path: {test_data_path}")
    print(f"NetCDF path: {nc_path}")

    # Load dataset
    print("\nLoading AWF test dataset...")
    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)
    print(f"  {len(dataset)} samples loaded")

    # Get dataset stats
    stats, scale_stats = retrieve_dataset_stats(dataset)

    # Open NetCDF
    print("\nOpening AWF database...")
    nc_dataset = xr.open_dataset(nc_path)
    print(f"  Layouts: {len(nc_dataset.layout)}, Flowcases: {len(nc_dataset.flowcase)}")
    print(f"  Grid: {len(nc_dataset.x)} x {len(nc_dataset.y)}")

    # Select representative layouts
    print("\nSelecting representative AWF layouts...")
    layout_type_idxs = select_awf_representative_layouts(dataset)
    print(f"Selected indices: {layout_type_idxs}")

    # Load model config for iterator setup
    model_cfg_path = model_cfg.portable_path / "model_config.json"
    if not model_cfg_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {model_cfg_path}. "
            "Please export the model first by running plot_publication_figures.py."
        )

    with open(model_cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)
    cfg.data.test_path = test_data_path

    # Setup plot iterator for all flowcases of selected layouts
    print("\nLoading plot data...")
    plot_graphs, test_dataset = setup_awf_plot_iterator(
        cfg, test_data_path, dataset, layout_type_idxs
    )

    # Build serialized data
    print("\nSerializing graph data and extracting flow fields...")
    D = AWF_TURBINE_DIAMETER
    x_downstream_D = [50, 100, 200, 300]

    serialized_data = {
        "metadata": {
            "layout_idxs": layout_type_idxs,
            "source": str(nc_path),
            "turbine_diameter_m": D,
            "x_downstream_D": x_downstream_D,
        },
        "scale_stats": scale_stats,
        "layout_data": {},
    }

    for layout_name, base_idx in tqdm(layout_type_idxs.items(), desc="Processing layouts"):
        # Match graph to NetCDF to get layout_idx
        print(f"\n  Matching {layout_name} (test_idx={base_idx}) to NetCDF...")
        nc_layout_idx, nc_fc0_idx = match_graph_to_netcdf_layout(
            dataset, base_idx, nc_dataset, scale_stats, D=D
        )
        print(f"    Matched: layout_idx={nc_layout_idx}, flowcase_idx={nc_fc0_idx}")

        # Process first flowcase for graph data
        data_in = plot_graphs.get(base_idx)
        if data_in is None:
            print(f"  Warning: No graph data for {layout_name} at idx={base_idx}")
            continue

        graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = data_in
        targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

        wt_positions = test_dataset[base_idx].pos.numpy()

        layout_entry = {
            "graphs": convert_graph_to_serializable(graphs),
            "probe_graphs": convert_graph_to_serializable(probe_graphs),
            "targets": np.array(targets),
            "wt_mask": np.array(wt_mask),
            "probe_mask": np.array(probe_mask),
            "node_positions": np.array(node_positions),
            "trunk_idxs": np.array(trunk_idxs),
            "wt_positions": wt_positions,
            "test_idx": base_idx,
            "nc_layout_idx": nc_layout_idx,
            "raw_flow_fields": {},
            "crossstream_profiles": {},
            "verification": {},
        }

        # Extract all 4 flowcases from NetCDF
        n_flowcases = len(nc_dataset.flowcase)
        for fc_idx in range(n_flowcases):
            print(f"    Extracting flowcase {fc_idx}...")
            flow_field = extract_raw_flow_field(nc_dataset, nc_layout_idx, fc_idx)
            layout_entry["raw_flow_fields"][fc_idx] = flow_field

            # Extract crossstream profiles
            cs_profiles = extract_crossstream_from_field(flow_field, x_downstream_D)
            layout_entry["crossstream_profiles"][fc_idx] = cs_profiles

            # Verify graph-NetCDF match (only for the flowcase matching this graph idx)
            graph_fc_idx = base_idx % 4
            if fc_idx == graph_fc_idx:
                verification = verify_graph_netcdf_match(
                    dataset, base_idx, flow_field, scale_stats, D=D
                )
                layout_entry["verification"] = verification
                status = "PASS" if verification["passed"] else "FAIL"
                print(
                    f"    Verification: {status} "
                    f"(max_error={verification['max_error']:.6f} m/s, "
                    f"mean_error={verification['mean_error']:.6f} m/s)"
                )

        # Also store per-flowcase graph data
        layout_entry["per_flowcase_graphs"] = {}
        for fc_offset in range(4):
            fc_test_idx = base_idx + fc_offset
            fc_data = plot_graphs.get(fc_test_idx)
            if fc_data is None:
                continue
            fc_graphs, fc_probe_graphs, fc_node_tuple, _, _ = fc_data
            fc_targets, fc_wt_mask, fc_probe_mask, fc_node_pos, fc_trunk_idxs = fc_node_tuple

            layout_entry["per_flowcase_graphs"][fc_offset] = {
                "graphs": convert_graph_to_serializable(fc_graphs),
                "probe_graphs": convert_graph_to_serializable(fc_probe_graphs),
                "targets": np.array(fc_targets),
                "wt_mask": np.array(fc_wt_mask),
                "probe_mask": np.array(fc_probe_mask),
                "node_positions": np.array(fc_node_pos),
                "trunk_idxs": np.array(fc_trunk_idxs),
                "wt_positions": test_dataset[fc_test_idx].pos.numpy(),
                "test_idx": fc_test_idx,
            }

        serialized_data["layout_data"][layout_name] = layout_entry

    nc_dataset.close()

    print(f"\nSaving to {raw_cache}...")
    with open(raw_cache, "wb") as f:
        pickle.dump(serialized_data, f)

    print("\nAWF data extraction complete!")
    print(f"  - {raw_cache}")
    print("\nCopy this file to your local machine and run:")
    print("  python plot_publication_figures.py --model L1_L3_phase2")


if __name__ == "__main__":
    main()
