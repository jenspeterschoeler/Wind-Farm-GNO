"""
Generate GNO predictions for ALL AWF test cases.

Evaluates every test case and produces two prediction versions per case:
1. Training region — predictions at pre-processed graph probe positions (cropped domain)
2. Full flow field — predictions on the complete NetCDF simulation grid

The 9 hand-picked publication cases get semantic keys (e.g. "small_low_ws_low_ti"),
all other cases get indexed keys ("case_000", "case_001", ...).  This way the
co-author's plotting scripts (which look up named keys) still work, while the full
pickle supports broader analyses across all ~160 test cases.

Usage:
    python Experiments/article_2/generate_selected_case_predictions.py
    python Experiments/article_2/generate_selected_case_predictions.py --skip-full-field
    python Experiments/article_2/generate_selected_case_predictions.py --model L1_global_scratch
    python Experiments/article_2/generate_selected_case_predictions.py --force
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
from article2_utils import (  # noqa: E402
    AWF_TURBINE_DIAMETER,
    MODELS,
    apply_mask,
    construct_awf_probe_graph,
    get_awf_database_path,
    get_awf_test_path,
    load_model_for_prediction,
)
from generate_awf_plot_data import extract_raw_flow_field  # noqa: E402
from select_awf_test_cases import (  # noqa: E402
    build_index_mapping,
    catalog_test_graphs,
    classify_farm_sizes,
    classify_inflow,
)

from utils.data_tools import retrieve_dataset_stats, setup_test_val_iterator  # noqa: E402
from utils.torch_loader import Torch_Geomtric_Dataset  # noqa: E402

# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "cache" / "selected_case_predictions"
SELECTED_CASES_JSON = SCRIPT_DIR / "figures" / "awf_case_selection" / "selected_cases.json"
D = AWF_TURBINE_DIAMETER  # 178.3 m


# =============================================================================
# Helpers
# =============================================================================


def apply_mask_2d(arr, mask):
    """Apply 1D mask to 2D array, keeping rows where mask != 0."""
    mask = np.asarray(mask).squeeze()
    keep = mask != 0
    return arr[keep]


# =============================================================================
# Full flow field predictions
# =============================================================================


def _predict_full_field_single(
    pred_fn, inverse_scale_target, wt_pos_m, wseff, ws_inf, ti_inf, probe_pos_m, U_flat, scale_stats
):
    """Run full-field prediction for one set of probe positions."""
    from jax import numpy as jnp

    jraph_graph, jraph_probe, node_tuple = construct_awf_probe_graph(
        wt_positions_m=wt_pos_m,
        wseff=wseff,
        ws_inf=ws_inf,
        ti_inf=ti_inf,
        probe_positions_m=probe_pos_m,
        probe_velocities=U_flat,
        scale_stats=scale_stats,
    )
    targets, wt_mask, probe_mask = node_tuple

    wt_mask_2d = jnp.asarray(wt_mask).reshape(-1, 1)
    probe_mask_2d = jnp.asarray(probe_mask).reshape(-1, 1)
    prediction = pred_fn(
        jraph_graph,
        jraph_probe,
        wt_mask_2d,
        probe_mask_2d,
    ).squeeze()

    # jax.device_get: PyWake monkey-patches np.asarray with a
    # __cuda_array_interface__ handler that is incompatible with JAX.
    prediction = jax.device_get(prediction)
    raw_pred = apply_mask(prediction, probe_mask)
    raw_targets = apply_mask(np.asarray(targets).squeeze(), probe_mask)

    pred_ms = np.asarray(inverse_scale_target(raw_pred))
    truth_ms = np.asarray(inverse_scale_target(raw_targets))
    return pred_ms, truth_ms


# Max probe-WT edges before chunking.
# 150k keeps peak GPU memory under ~8 GB (model + activations + edges).
_MAX_EDGES = 150_000


def predict_full_field(pred_fn, inverse_scale_target, scale_stats, nc_dataset, case):
    """Run predictions on the complete NetCDF simulation grid.

    For large wind farms the grid is split into chunks to avoid OOM.
    Returns dict with x_grid_D, y_grid_D, ground_truth_2d, predictions_2d.
    """
    layout_idx = case["netcdf_layout_idx"]
    flowcase_idx = case["netcdf_flowcase_idx"]

    # Extract raw flow field from NetCDF
    flow_field = extract_raw_flow_field(nc_dataset, layout_idx, flowcase_idx)

    x_grid_D = flow_field["x_grid_D"]
    y_grid_D = flow_field["y_grid_D"]
    wt_x_D = flow_field["wt_x_D"]
    wt_y_D = flow_field["wt_y_D"]
    ws_inf = flow_field["ws_inf"]
    ti_inf = flow_field["ti_inf"]
    wseff = flow_field["wseff"]
    U_field = flow_field["U_field"]  # (n_x, n_y) in m/s

    # Build meshgrid and flatten to probe positions
    xx, yy = np.meshgrid(x_grid_D, y_grid_D, indexing="ij")
    probe_pos_D = np.c_[xx.ravel(), yy.ravel()]
    probe_pos_m = probe_pos_D * D

    # WT positions in meters
    wt_pos_m = np.c_[wt_x_D, wt_y_D] * D
    n_wt = len(wt_pos_m)
    n_probes = len(probe_pos_m)

    # Ground truth velocities at all grid points
    U_flat = U_field.ravel()

    # Decide whether to chunk
    total_edges = n_probes * n_wt
    if total_edges <= _MAX_EDGES:
        pred_ms, truth_ms = _predict_full_field_single(
            pred_fn,
            inverse_scale_target,
            wt_pos_m,
            wseff,
            ws_inf,
            ti_inf,
            probe_pos_m,
            U_flat,
            scale_stats,
        )
    else:
        # Chunk probes to stay under edge limit
        chunk_size = max(1, _MAX_EDGES // n_wt)
        n_chunks = (n_probes + chunk_size - 1) // chunk_size
        print(
            f"    Chunking: {n_probes} probes x {n_wt} WTs = {total_edges:.1e} edges"
            f" -> {n_chunks} chunks of {chunk_size}"
        )

        pred_chunks = []
        truth_chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_probes)
            p, t = _predict_full_field_single(
                pred_fn,
                inverse_scale_target,
                wt_pos_m,
                wseff,
                ws_inf,
                ti_inf,
                probe_pos_m[start:end],
                U_flat[start:end],
                scale_stats,
            )
            pred_chunks.append(p)
            truth_chunks.append(t)
            # Free JIT cache for this graph shape
            jax.clear_caches()
            gc.collect()

        pred_ms = np.concatenate(pred_chunks)
        truth_ms = np.concatenate(truth_chunks)

    # Reshape back to 2D grid
    n_x = len(x_grid_D)
    n_y = len(y_grid_D)
    predictions_2d = pred_ms.reshape(n_x, n_y)
    ground_truth_2d = truth_ms.reshape(n_x, n_y)

    return {
        "x_grid_D": x_grid_D,
        "y_grid_D": y_grid_D,
        "ground_truth_2d": ground_truth_2d,
        "predictions_2d": predictions_2d,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate GNO predictions for ALL AWF test cases")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    parser.add_argument(
        "--skip-full-field", action="store_true", help="Skip full flow field predictions"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="L1_L3_phase2",
        help="Model key to use for predictions (default: L1_L3_phase2)",
    )
    args = parser.parse_args()

    model_key = args.model
    cache_file = CACHE_DIR / f"predictions_{model_key}.pkl"

    # Check cache
    if not args.force and cache_file.exists():
        print(f"Cache exists: {cache_file}")
        print("Use --force to regenerate.")
        return

    # =========================================================================
    # Step 1: Load selected cases (9 named keys) and build lookup dicts
    # =========================================================================
    print(f"Loading selected cases from {SELECTED_CASES_JSON}")
    with open(SELECTED_CASES_JSON) as f:
        cases_json = json.load(f)
    selected_cases = cases_json["cases"]
    print(f"  {len(selected_cases)} selected cases loaded")

    # dataset_idx -> semantic key  (e.g. 105 -> "small_low_ws_low_ti")
    selected_idx_to_key = {}
    # dataset_idx -> case metadata dict
    selected_idx_to_case = {}
    for case in selected_cases:
        key = f"{case['size_category']}_{case['inflow_target']}"
        selected_idx_to_key[case["dataset_idx"]] = key
        selected_idx_to_case[case["dataset_idx"]] = case

    # =========================================================================
    # Step 2: Load model
    # =========================================================================
    model_cfg = MODELS[model_key]
    print(f"\nLoading model: {model_key}")
    pred_fn, inverse_scale_target, scale_stats, model, params = load_model_for_prediction(model_cfg)

    # =========================================================================
    # Step 3: Load AWF test dataset, catalog all cases, build NetCDF mapping
    # =========================================================================
    test_data_path = get_awf_test_path()
    nc_path = get_awf_database_path()
    print(f"\nTest data: {test_data_path}")
    print(f"NetCDF: {nc_path}")

    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)
    n_total = len(dataset)
    print(f"  {n_total} test samples")

    _, scale_stats_ds = retrieve_dataset_stats(dataset)

    # Catalog all test graphs with metadata
    print("\nCataloging all test graphs...")
    catalog_df = catalog_test_graphs(dataset, scale_stats_ds)
    catalog_df = classify_farm_sizes(catalog_df)
    catalog_df = classify_inflow(catalog_df)

    # Build dataset_idx -> (netcdf_layout_idx, netcdf_flowcase_idx) mapping
    # using the same approach as select_awf_test_cases.py
    print("\nBuilding NetCDF index mapping...")
    layout_mapping = build_index_mapping(test_data_path)
    print(f"  {len(layout_mapping)} layout zips mapped")

    # For each dataset_idx, compute netcdf indices
    netcdf_indices = {}
    for idx in range(n_total):
        zip_position = idx // 4
        flowcase_idx = idx % 4
        netcdf_layout_idx = layout_mapping[zip_position]
        netcdf_indices[idx] = (netcdf_layout_idx, flowcase_idx)

    # Verify selected cases match expected netcdf indices
    for case in selected_cases:
        idx = case["dataset_idx"]
        expected = (case["netcdf_layout_idx"], case["netcdf_flowcase_idx"])
        actual = netcdf_indices[idx]
        if actual != expected:
            raise RuntimeError(
                f"NetCDF index mismatch for selected case idx={idx}: "
                f"expected {expected}, got {actual}"
            )
    print("  Selected case NetCDF indices verified.")

    # Open NetCDF
    nc_dataset = xr.open_dataset(nc_path)
    print(f"  NetCDF: {len(nc_dataset.layout)} layouts, {len(nc_dataset.flowcase)} flowcases")

    # =========================================================================
    # Step 4: Setup standard test iterator (all cases, batch_size=1)
    # =========================================================================
    model_cfg_path = model_cfg.portable_path / "model_config.json"
    with open(model_cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)
    cfg.data.test_path = test_data_path

    print("\nSetting up test iterator...")
    get_data_iterator, test_dataset, _, _ = setup_test_val_iterator(
        cfg,
        type_str="test",
        return_positions=True,
        dataset=dataset,
        num_workers=0,
        cache=False,
        return_layout_info=False,
    )

    # Precompute unscaling constants
    dist_range = scale_stats["distance"]["range"]
    if isinstance(dist_range, list):
        dist_range = dist_range[0]

    # =========================================================================
    # Step 5: Iterate ALL cases — training-region predictions
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Generating training-region predictions for {n_total} cases...")
    print(f"{'='*60}")

    from jax import numpy as jnp  # noqa: E402

    output_cases = {}
    iterator = get_data_iterator()

    for i, data_in in tqdm(enumerate(iterator), total=n_total, desc="Training-region predictions"):
        # Clear caches periodically
        if i % 20 == 0:
            jax.clear_caches()
            gc.collect()

        graphs, probe_graphs, node_array_tuple = data_in
        targets, wt_mask, probe_mask, node_positions = node_array_tuple

        # Run prediction
        wt_mask_2d = (
            jnp.asarray(wt_mask).reshape(-1, 1) if wt_mask.ndim == 1 else jnp.asarray(wt_mask)
        )
        probe_mask_2d = (
            jnp.asarray(probe_mask).reshape(-1, 1)
            if probe_mask.ndim == 1
            else jnp.asarray(probe_mask)
        )
        prediction = pred_fn(
            graphs,
            probe_graphs,
            wt_mask_2d,
            probe_mask_2d,
        ).squeeze()

        # Extract probe-only values and unscale
        # NOTE: jax.device_get() converts JAX GPU arrays to numpy before
        # apply_mask, because PyWake monkey-patches np.asarray with a
        # __cuda_array_interface__ handler that is incompatible with JAX.
        prediction = jax.device_get(prediction)
        raw_pred = apply_mask(prediction, probe_mask)
        raw_targets = apply_mask(np.asarray(targets).squeeze(), probe_mask)
        pred_ms = np.asarray(inverse_scale_target(raw_pred))
        truth_ms = np.asarray(inverse_scale_target(raw_targets))

        # Probe positions (unscaled)
        all_positions = np.asarray(node_positions)
        probe_positions_scaled = apply_mask_2d(all_positions, np.asarray(probe_mask))
        probe_positions_m = probe_positions_scaled * dist_range
        probe_positions_D = probe_positions_m / D

        # WT positions
        test_data = test_dataset[i]
        wt_pos_scaled = test_data.pos.numpy()
        wt_pos_m = wt_pos_scaled * dist_range
        wt_pos_D = wt_pos_m / D

        # Get metadata from catalog
        row = catalog_df.iloc[i]
        nc_layout_idx, nc_flowcase_idx = netcdf_indices[i]

        # Assign key: semantic name for selected cases, indexed for the rest
        key = selected_idx_to_key.get(i, f"case_{i:03d}")

        output_cases[key] = {
            "size_category": row["size_category"],
            "inflow_target": row.get("inflow_category", ""),
            "n_wt": int(row["n_wt"]),
            "ws_inf": float(row["ws_inf"]),
            "ti_inf": float(row["ti_inf"]),
            "dataset_idx": i,
            "netcdf_layout_idx": nc_layout_idx,
            "netcdf_flowcase_idx": nc_flowcase_idx,
            "wt_positions_D": wt_pos_D,
            "wt_positions_m": wt_pos_m,
            "wseff": None,  # filled during full-field step
            "training_region": {
                "probe_positions_D": probe_positions_D,
                "probe_positions_m": probe_positions_m,
                "ground_truth": truth_ms,
                "predictions": pred_ms,
            },
        }

    print(f"  Training-region predictions done for {len(output_cases)} cases.")

    # =========================================================================
    # Step 6: Full-field predictions for ALL cases
    # =========================================================================
    if not args.skip_full_field:
        print(f"\n{'='*60}")
        print(f"Generating full-field predictions for {len(output_cases)} cases...")
        print(f"{'='*60}")

        for key in tqdm(output_cases, desc="Full-field predictions"):
            case_data = output_cases[key]
            case_info = {
                "netcdf_layout_idx": case_data["netcdf_layout_idx"],
                "netcdf_flowcase_idx": case_data["netcdf_flowcase_idx"],
            }

            idx = case_data["dataset_idx"]
            n_wt = case_data["n_wt"]
            print(f"\n  {key} (idx={idx}, n_wt={n_wt})")

            # Extract wseff from NetCDF
            flow_field = extract_raw_flow_field(
                nc_dataset,
                case_data["netcdf_layout_idx"],
                case_data["netcdf_flowcase_idx"],
            )
            case_data["wseff"] = flow_field["wseff"]

            # Full-field prediction
            full_field_result = predict_full_field(
                pred_fn, inverse_scale_target, scale_stats, nc_dataset, case_info
            )
            case_data["full_field"] = full_field_result

            nx, ny = full_field_result["predictions_2d"].shape
            print(f"    Full field: {nx} x {ny} grid")

            # Memory management
            jax.clear_caches()
            gc.collect()
    else:
        # Still extract wseff even when skipping full-field
        print("\nExtracting wseff for all cases (skipping full-field)...")
        for key in tqdm(output_cases, desc="Extracting wseff"):
            case_data = output_cases[key]
            flow_field = extract_raw_flow_field(
                nc_dataset,
                case_data["netcdf_layout_idx"],
                case_data["netcdf_flowcase_idx"],
            )
            case_data["wseff"] = flow_field["wseff"]

    nc_dataset.close()

    # =========================================================================
    # Step 7: Save output
    # =========================================================================
    selected_case_keys = list(selected_idx_to_key.values())

    output = {
        "metadata": {
            "model_name": model_key,
            "turbine_diameter_m": D,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "skip_full_field": args.skip_full_field,
            "n_total_cases": n_total,
            "selected_case_keys": selected_case_keys,
        },
        "scale_stats": scale_stats,
        "cases": output_cases,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(output, f)

    # Summary
    size_bytes = cache_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nDone! File size: {size_mb:.1f} MB")
    print(f"Total cases: {len(output_cases)}")
    print(f"Selected (named) cases: {len(selected_case_keys)}")
    print(f"  Keys: {selected_case_keys}")

    n_with_ff = sum(1 for c in output_cases.values() if "full_field" in c)
    print(f"Cases with full_field: {n_with_ff}")

    print(f"\nOutput: {cache_file}")


if __name__ == "__main__":
    main()
