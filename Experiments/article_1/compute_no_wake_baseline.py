"""Compute error metrics for a no-wake baseline prediction.

This script computes error metrics for the baseline assumption that there are
no wake effects, meaning the predicted flow velocity at every location equals
the freestream wind speed (u = U_freestream).

This provides a reference point to demonstrate the value of the learned GNO model.

Run this script on the Sophia cluster where the test data is located.
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Add project root to path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils.data_tools import (  # noqa: E402
    retrieve_dataset_stats,
    setup_test_val_iterator,
    setup_unscaler,
)
from utils.misc import add_to_hydra_cfg, convert_ndarray  # noqa: E402
from utils.torch_loader import Torch_Geomtric_Dataset  # noqa: E402

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

jax.config.update("jax_default_device", jax.devices("cpu")[0])


def compute_no_wake_baseline(cfg: DictConfig) -> None:
    """Compute error metrics for the no-wake baseline prediction.

    The baseline assumes u = U_freestream everywhere, i.e., no wake effects.
    """
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "no_wake_baseline"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and retrieve stats (needed for config)
    dataset = Torch_Geomtric_Dataset(root_path=os.path.abspath(cfg.data.test_path), in_mem=False)
    dataset_stats, dataset_scale_stats = retrieve_dataset_stats(dataset)

    # Add stats to config (required by setup_test_val_iterator)
    cfg.data = add_to_hydra_cfg(cfg.data, "stats", dataset_stats)
    cfg.data = add_to_hydra_cfg(cfg.data, "scale_stats", dataset_scale_stats)

    get_data_iterator, pyg_dataset, stats, scale_stats = setup_test_val_iterator(
        cfg,
        type_str="test",
        trunk_sample_strategy="evenly_distributed",
        idxs_per_sample=1000,
        cache=False,
        return_layout_info=True,
        return_positions=True,
        dataset=dataset,
        num_workers=0,
    )

    data_iterator = get_data_iterator()

    unscaler = setup_unscaler(cfg, scale_stats=scale_stats)

    # Initialize accumulators
    graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = next(data_iterator)
    targets, wt_mask, probe_mask, positions = node_array_tuple

    errors = np.zeros(targets.shape[-1])
    sq_errors = np.copy(errors)
    abs_errors = np.copy(errors)
    ape_errors = np.copy(errors)
    n_samples = np.copy(errors).astype(np.int64)
    df_local_metrics = pd.DataFrame()

    print("#######################")
    print("Computing no-wake baseline metrics")
    print(f"Output directory: {output_dir}")
    print("#######################")

    for _i, (
        graphs,
        probe_graphs,
        node_array_tuple,
        layout_type,
        wt_spacing,
    ) in tqdm(enumerate(data_iterator)):
        targets, wt_mask, probe_mask, positions = node_array_tuple

        # Unscale to physical units
        unscaled_graphs = unscaler.inverse_scale_graph(graphs)
        assert unscaled_graphs.globals is not None, "Graph globals must be present"
        U_freestream = unscaled_graphs.globals[0, 0]  # type: ignore[index]
        TI_ambient = unscaled_graphs.globals[0, 1]  # type: ignore[index]

        # Unscale targets to physical units
        targets_unscaled = unscaler.inverse_scale_output(targets.squeeze())

        # The baseline prediction is just U_freestream everywhere
        prediction = jnp.full_like(targets_unscaled, U_freestream)

        # Compute errors
        error = targets_unscaled - prediction
        sq_error = error**2
        abs_error = jnp.abs(error)

        # MAPE: absolute percentage error relative to freestream
        ape = jnp.abs(targets_unscaled - prediction) / U_freestream * 100

        # Sum over valid samples (using masks)
        n_samples_ = jnp.sum(wt_mask) + jnp.sum(probe_mask)
        err_sum_ = jnp.sum(error, axis=0)
        sq_err_sum_ = jnp.sum(sq_error, axis=0)
        abs_err_sum_ = jnp.sum(abs_error, axis=0)
        ape_sum = jnp.sum(ape, axis=0)

        # Local metrics for this flowcase
        mape_local = jnp.mean(ape)
        rmse_local = jnp.sqrt(jnp.mean(sq_error))
        n_wt = int(jnp.sum(wt_mask))

        local_row = {
            "U_freestream": float(U_freestream),
            "TI_ambient": float(TI_ambient),
            "n_edges": int(graphs.n_edge[0]),
            "n_probe_edges": int(probe_graphs.n_edge[0]),
            "n_wt": int(n_wt),
            "n_probes": int(jnp.sum(probe_mask)),
            "layout_type": str(layout_type[0]),
            "wt_spacing": wt_spacing[0].numpy(),
            "mape_local": float(mape_local),
            "rmse_local": float(rmse_local),
        }
        df_local_metrics = pd.concat(
            [df_local_metrics, pd.DataFrame([local_row])], ignore_index=True
        )

        # Accumulate global metrics
        errors += np.float64(err_sum_)
        sq_errors += np.float64(sq_err_sum_)
        abs_errors += np.float64(abs_err_sum_)
        ape_errors += np.float64(ape_sum)
        n_samples += np.int64(n_samples_)

    # Compute final metrics
    mse = sq_errors / n_samples
    mae = abs_errors / n_samples
    mape = ape_errors / n_samples
    rmse = np.sqrt(mse)

    # Handle multi-output case (convert to dict if needed)
    if len(cfg.data.io.target_node_features) > 1:
        mse = dict(zip(cfg.data.io.target_node_features, mse))
        mae = dict(zip(cfg.data.io.target_node_features, mae))
        rmse = dict(zip(cfg.data.io.target_node_features, rmse))
        mape = dict(zip(cfg.data.io.target_node_features, mape))

    # Structure metrics like post_process_GNO_probe.py for compatibility
    error_metrics = {
        "test/best/mse": mse,
        "test/best/mae": mae,
        "test/best/rmse": rmse,
        "test/best/mape": mape,
    }

    # Convert ndarrays to lists for JSON serialization
    error_metrics = convert_ndarray(error_metrics)

    # Save outputs
    with open(os.path.join(output_dir, "error_metrics.json"), "w") as f:
        json.dump(error_metrics, f, indent=4)

    df_local_metrics.to_csv(os.path.join(output_dir, "local_metrics.csv"), index=False)

    # Also save a summary CSV for easy viewing
    summary_df = pd.DataFrame(
        {
            "metric": ["MSE", "MAE", "RMSE", "MAPE"],
            "value": [
                error_metrics["test/best/mse"],
                error_metrics["test/best/mae"],
                error_metrics["test/best/rmse"],
                error_metrics["test/best/mape"],
            ],
        }
    )
    summary_df.to_csv(os.path.join(output_dir, "no_wake_baseline_metrics.csv"), index=False)

    print("\n#######################")
    print("No-wake baseline metrics computed:")
    print(f"  MSE:  {error_metrics['test/best/mse']}")
    print(f"  MAE:  {error_metrics['test/best/mae']}")
    print(f"  RMSE: {error_metrics['test/best/rmse']}")
    print(f"  MAPE: {error_metrics['test/best/mape']}")
    print(f"\nResults saved to: {output_dir}")
    print("#######################")


if __name__ == "__main__":
    import warnings

    from hydra import compose, initialize

    from utils.misc import add_to_hydra_cfg

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Use the main GNO_probe config which has the correct data paths for Sophia
    # Hydra config_path must be relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path_abs = os.path.join(script_dir, "..", "..", "configurations")
    config_path = os.path.relpath(config_path_abs, script_dir)
    config_name = "GNO_probe"

    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name=config_name)

    cfg = add_to_hydra_cfg(cfg, "wandb", DictConfig({"use": False}))

    # Override test path to use the correct test set
    cfg.data.test_path = (
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )

    compute_no_wake_baseline(cfg)
