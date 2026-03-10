"""
Data extraction script for Article 2 publication figures.

Run this script on the Sophia cluster to extract the raw graph data
from the turbopark_2500layouts dataset. The actual prediction generation
happens locally in plot_publication_figures.py.

Usage:
    python Experiments/article_2/generate_plot_data.py [--force] [--model L2_global]

Outputs:
    - Experiments/article_2/cache/publication_figures/<dataset_name>/raw_graph_data.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from article2_utils import (  # noqa: E402
    MODELS,
    Article2Config,
    convert_graph_to_serializable,
    get_max_plot_distance,
    get_sophia_test_path,
    select_representative_layouts,
    select_representative_layouts_per_windspeed,
)

from utils.data_tools import retrieve_dataset_stats, setup_test_val_iterator  # noqa: E402
from utils.torch_loader import Torch_Geomtric_Dataset  # noqa: E402


def setup_plot_iterator(cfg, test_data_path: str, dataset, layout_idxs: dict):
    """Setup iterator and retrieve data for specified layout indices."""
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
        num_workers=0,
    )

    iterator = get_plot_data_iterator()
    plot_graphs = deepcopy(layout_idxs)

    for i, data_in in tqdm(enumerate(iterator), desc="Loading plot data"):
        if i in layout_idxs.values():
            for key, val in layout_idxs.items():
                if val == i:
                    plot_graphs[key] = data_in

    return plot_graphs, test_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extract graph data for Article 2 publication figures"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    parser.add_argument(
        "--model",
        type=str,
        default="L2_global",
        choices=list(MODELS.keys()),
        help="Model to use for config (default: L2_global)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Configuration
    model_cfg = MODELS[args.model]
    cache_dir = script_dir / "cache" / "publication_figures" / model_cfg.dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = cache_dir / "raw_graph_data.pkl"

    if not args.force and raw_cache.exists():
        print("Cache file already exists. Use --force to regenerate.")
        return

    # Get paths
    test_data_path = get_sophia_test_path()
    print(f"Test data path: {test_data_path}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

    # Get dataset stats
    stats, scale_stats = retrieve_dataset_stats(dataset)

    article2_cfg = Article2Config(
        model=model_cfg,
        test_data_path=Path(test_data_path),
        dataset_root=model_cfg.dataset_path or Path(test_data_path).parent,
    )

    # Select representative layouts
    print("\nSelecting representative layouts...")
    layout_type_idxs = select_representative_layouts(dataset)
    print(f"Selected indices: {layout_type_idxs}")

    # Get plot distance
    plot_distance, max_y_range = get_max_plot_distance(dataset, layout_type_idxs)

    # Load model config for iterator setup
    model_cfg_path = model_cfg.portable_path / "model_config.json"
    if not model_cfg_path.exists():
        # Fallback: check Sophia path
        sophia_model_path = Path(
            "/work/users/jpsch/gno/outputs/transfer_learning/phase1/multirun/2026-01-22/10-44-19/2_+experiment=phase1/S2_dropout_layernorm"
        )
        model_cfg_path = sophia_model_path / "model_config.json"

    with open(model_cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Override test path to point to actual data
    cfg.data.test_path = test_data_path

    # Setup plot iterator for main layouts
    print("\nLoading plot data (main layouts)...")
    plot_graphs, test_dataset = setup_plot_iterator(cfg, test_data_path, dataset, layout_type_idxs)

    # Extract and serialize main layout data
    print("\nSerializing graph data...")
    serialized_data = {
        "metadata": {
            "layout_type_idxs": layout_type_idxs,
            "plot_distance": plot_distance,
            "max_y_range": max_y_range,
        },
        "scale_stats": scale_stats,
        "layout_data": {},
    }

    for layout_name, val in tqdm(plot_graphs.items(), desc="Processing layouts"):
        if isinstance(val, int):
            print(f"  Warning: No data loaded for {layout_name} (still idx={val})")
            continue

        graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
        targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

        test_idx = layout_type_idxs[layout_name]
        wt_positions = test_dataset[test_idx].pos.numpy()

        serialized_data["layout_data"][layout_name] = {
            "graphs": convert_graph_to_serializable(graphs),
            "probe_graphs": convert_graph_to_serializable(probe_graphs),
            "targets": np.array(targets),
            "wt_mask": np.array(wt_mask),
            "probe_mask": np.array(probe_mask),
            "node_positions": np.array(node_positions),
            "trunk_idxs": np.array(trunk_idxs),
            "layout_type": layout_type,
            "wt_spacing": float(wt_spacing.numpy()[0])
            if hasattr(wt_spacing, "numpy")
            else float(wt_spacing),
            "wt_positions": wt_positions,
            "test_idx": test_idx,
        }

    # === Per-wind-speed data for WT spatial errors ===
    print("\nSelecting per-wind-speed representative layouts...")
    target_windspeeds = article2_cfg.U_free
    layout_ws_idxs = select_representative_layouts_per_windspeed(
        dataset,
        scale_stats,
        target_windspeeds=target_windspeeds,
        windspeed_tolerance=0.5,
    )

    # Create flat index dict
    flat_ws_idxs = {}
    for layout_name, ws_dict in layout_ws_idxs.items():
        for ws, idx in ws_dict.items():
            if idx is not None:
                flat_ws_idxs[f"{layout_name}_{ws}"] = idx

    # Load per-windspeed graphs
    print("\nLoading per-wind-speed graph data...")
    ws_plot_graphs, _ = setup_plot_iterator(cfg, test_data_path, dataset, flat_ws_idxs)

    vel_min = scale_stats["velocity"]["min"][0]
    vel_range = scale_stats["velocity"]["range"][0]

    print("\nSerializing per-wind-speed graph data...")
    serialized_data["per_windspeed_data"] = {
        "target_windspeeds": target_windspeeds,
        "layout_ws_idxs": layout_ws_idxs,
        "layout_data": {},
    }

    for layout_name in layout_ws_idxs:
        serialized_data["per_windspeed_data"]["layout_data"][layout_name] = {}

        for ws, idx in layout_ws_idxs[layout_name].items():
            if idx is None:
                print(f"  Warning: No data found for {layout_name} at U={ws} m/s")
                continue

            key = f"{layout_name}_{ws}"
            data_in = ws_plot_graphs.get(key)
            if data_in is None or isinstance(data_in, int):
                print(f"  Warning: Failed to load data for {key}")
                continue

            graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = data_in
            targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

            actual_U = graphs.globals[0][0] * vel_range + vel_min

            serialized_data["per_windspeed_data"]["layout_data"][layout_name][ws] = {
                "graphs": convert_graph_to_serializable(graphs),
                "targets": np.array(targets),
                "wt_mask": np.array(wt_mask),
                "probe_mask": np.array(probe_mask),
                "node_positions": np.array(node_positions),
                "trunk_idxs": np.array(trunk_idxs),
                "wt_positions": dataset[idx].pos.numpy(),
                "test_idx": idx,
                "actual_U": float(actual_U),
            }

    print(f"\nSaving to {raw_cache}...")
    with open(raw_cache, "wb") as f:
        pickle.dump(serialized_data, f)

    print("\nData extraction complete!")
    print(f"  - {raw_cache}")
    print("\nCopy this file to your local machine and run plot_publication_figures.py")


if __name__ == "__main__":
    main()
