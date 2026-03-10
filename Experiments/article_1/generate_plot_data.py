"""
Data extraction script for Article 1 publication figures.

Run this script on the Sophia cluster to extract the raw graph data
from the dataset. The actual prediction generation happens locally.

Usage:
    python generate_plot_data.py [--force]

Outputs:
    - cache/raw_graph_data.pkl  (graph data for plotting)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from plot_publication_figures import (
    get_max_plot_distance,
    select_representative_layouts,
    select_representative_layouts_per_windspeed,
    setup_plot_iterator,
)

from utils.data_tools import retrieve_dataset_stats
from utils.torch_loader import Torch_Geomtric_Dataset


def get_paths():
    """Get appropriate paths based on environment."""
    # Check if running on Sophia
    if os.path.exists("/work/users/jpsch"):
        test_data_path = (
            "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
        )
    else:
        # Local paths
        test_data_path = "./data/zenodo_graphs/test_pre_processed"

    return test_data_path


def convert_graph_to_serializable(graph):
    """Convert a jraph GraphsTuple to a serializable dictionary."""
    return {
        "nodes": np.array(graph.nodes) if graph.nodes is not None else None,
        "edges": np.array(graph.edges) if graph.edges is not None else None,
        "receivers": np.array(graph.receivers) if graph.receivers is not None else None,
        "senders": np.array(graph.senders) if graph.senders is not None else None,
        "globals": np.array(graph.globals) if graph.globals is not None else None,
        "n_node": np.array(graph.n_node) if graph.n_node is not None else None,
        "n_edge": np.array(graph.n_edge) if graph.n_edge is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract graph data for Article 1")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cache_dir = script_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    raw_cache = cache_dir / "raw_graph_data.pkl"

    if not args.force and raw_cache.exists():
        print("Cache file already exists. Use --force to regenerate.")
        return

    # Get paths
    test_data_path = get_paths()
    print(f"Test data path: {test_data_path}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

    # Get dataset stats
    stats, scale_stats = retrieve_dataset_stats(dataset)

    # Select representative layouts
    print("\nSelecting representative layouts...")
    layout_type_idxs = select_representative_layouts(dataset)
    print(f"Selected indices: {layout_type_idxs}")

    # Get plot distance
    plot_distance, max_y_range = get_max_plot_distance(dataset, layout_type_idxs)

    # We need to load the model config to setup the iterator
    # Use a minimal config approach - just load what we need
    from omegaconf import OmegaConf

    # Load config from the best model
    main_path = "./assets/best_model_Vj8"
    if not os.path.exists(main_path):
        # On Sophia, use absolute path
        main_path = "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"

    import json

    model_cfg_path = os.path.join(main_path, "model_config.json")
    with open(model_cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)

    # Setup plot iterator
    print("\nLoading plot data...")
    plot_graphs, test_dataset = setup_plot_iterator(cfg, test_data_path, dataset, layout_type_idxs)

    # Extract and serialize the graph data
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
        graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
        targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

        # Get test positions for this layout
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
            "wt_positions": wt_positions,  # Unscaled positions from dataset
            "test_idx": test_idx,
        }

    # === Per-wind-speed data for WT spatial errors figure ===
    print("\nSelecting per-wind-speed representative layouts...")
    target_windspeeds = [6.0, 12.0, 18.0]
    layout_ws_idxs = select_representative_layouts_per_windspeed(
        dataset,
        scale_stats,
        target_windspeeds=target_windspeeds,
        windspeed_tolerance=0.5,
    )

    # Create flat index dict for setup_plot_iterator (key -> idx)
    flat_ws_idxs = {}
    for layout_name, ws_dict in layout_ws_idxs.items():
        for ws, idx in ws_dict.items():
            if idx is not None:
                flat_ws_idxs[f"{layout_name}_{ws}"] = idx

    # Load per-windspeed graphs using same iterator approach
    print("\nLoading per-wind-speed graph data...")
    ws_plot_graphs, _ = setup_plot_iterator(cfg, test_data_path, dataset, flat_ws_idxs)

    # Get velocity scaling for actual U computation
    vel_min = scale_stats["velocity"]["min"][0]
    vel_range = scale_stats["velocity"]["range"][0]

    # Reorganize into nested structure and serialize
    print("\nSerializing per-wind-speed graph data...")
    serialized_data["per_windspeed_data"] = {
        "target_windspeeds": target_windspeeds,
        "layout_ws_idxs": layout_ws_idxs,
        "layout_data": {},
    }

    for layout_name in layout_ws_idxs.keys():
        serialized_data["per_windspeed_data"]["layout_data"][layout_name] = {}

        for ws, idx in layout_ws_idxs[layout_name].items():
            if idx is None:
                print(f"  Warning: No data found for {layout_name} at U={ws} m/s")
                continue

            key = f"{layout_name}_{ws}"
            data_in = ws_plot_graphs[key]

            # Unpack the data tuple (same structure as main plot_graphs)
            graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = data_in
            targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

            # Get actual wind speed from graph globals
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
