"""Post-processing and evaluation of trained GNO models."""

import json
import multiprocessing as mp
import os

import jax
import jraph
import numpy as np
import pandas as pd
from jax import numpy as jnp
from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_tools import retrieve_dataset_stats, setup_test_val_iterator, setup_unscaler
from utils.GNO_probe import inverse_scale_rel_ws
from utils.misc import convert_ndarray, get_model_paths
from utils.model_tools import load_model, setup_model
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set by caller; ignore
    pass

jax.config.update("jax_default_device", jax.devices("cpu")[0])


def post_process_GNO_probe(
    cfg: DictConfig, wandb_run=None, scale_stats_path: str | None = None
) -> None:
    """
    Evaluate trained models on test set.

    Supports multi-metric checkpoint structure:
    - final: Final epoch model
    - best_mse: Best model by MSE
    - best_mae: Best model by MAE
    - best_hybrid: Best model by hybrid metric

    Also supports legacy single-checkpoint format (best maps to best_mse).
    """
    paths = get_model_paths(cfg)  # Returns dict with all checkpoint types

    error_metrics = {}

    # Define checkpoint types to evaluate (in order of priority)
    # Legacy 'best' is mapped to 'best_mse' in get_model_paths
    checkpoint_types = [
        ("final", paths.get("final")),
        ("best_mse", paths.get("best_mse")),
        ("best_mae", paths.get("best_mae")),
        ("best_hybrid", paths.get("best_hybrid")),
    ]

    for model_type_str, model_path in checkpoint_types:
        # Load model
        if model_path is None:
            continue

        parent_path = os.path.dirname(model_path)
        # For best_* checkpoints, go up one more directory level
        # (path is model/checkpoints_best_*/epoch/ -> need model/)
        if model_type_str.startswith("best_"):
            parent_path = os.path.dirname(parent_path)

        plot_folder_path = os.path.join(parent_path, f"plots_{model_type_str}")
        os.makedirs(plot_folder_path, exist_ok=True)
        os.makedirs(os.path.join(plot_folder_path, "flowcases"), exist_ok=True)

        # use the portable loading if possible
        cfg_json_path = os.path.abspath(
            os.path.join(model_path.split("model")[0], "model_config.json")
        )
        if os.path.exists(cfg_json_path):
            params_path = os.path.abspath(
                os.path.join(model_path.split("model")[0], f"{model_type_str}_params.msgpack")
            )
            dataset = Torch_Geomtric_Dataset(
                root_path=os.path.abspath(cfg.data.test_path), in_mem=False
            )
            stats, scale_stats = retrieve_dataset_stats(dataset)
            params, cfg_model, model, _ = load_portable_model(params_path, cfg_json_path, dataset)
        else:
            model, params, metrics, cfg_model = load_model(model_path)
            model = setup_model(cfg_model)

        dataset = Torch_Geomtric_Dataset(root_path=os.path.abspath(cfg.data.test_path), in_mem=True)

        # Try with layout info first; fall back if dataset lacks layout_type/wt_spacing
        has_layout_info = True
        try:
            get_data_iterator, pyg_dataset, stats, scale_stats = setup_test_val_iterator(
                cfg_model,
                type_str="test",
                trunk_sample_strategy="evenly_distributed",
                idxs_per_sample=1000,
                cache=True,
                return_layout_info=True,
                return_positions=True,
                dataset=dataset,
                num_workers=0,
            )
            data_iterator = get_data_iterator()
            first_batch = next(data_iterator)
            graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = first_batch
        except (AttributeError, ValueError):
            has_layout_info = False
            get_data_iterator, pyg_dataset, stats, scale_stats = setup_test_val_iterator(
                cfg_model,
                type_str="test",
                trunk_sample_strategy="evenly_distributed",
                idxs_per_sample=1000,
                cache=True,
                return_layout_info=False,
                return_positions=True,
                dataset=dataset,
                num_workers=0,
            )
            data_iterator = get_data_iterator()
            first_batch = next(data_iterator)
            graphs, probe_graphs, node_array_tuple = first_batch

        targets, wt_mask, probe_mask, positions = node_array_tuple

        # Override scale_stats if an external path was provided (e.g. recall evaluation)
        if scale_stats_path:
            with open(scale_stats_path) as f:
                scale_stats = json.load(f)

        if "scale_rel_ws" not in cfg_model:
            scale_rel_ws = False

        unscaler = setup_unscaler(cfg_model, scale_stats=scale_stats)

        def model_prediction_fn(
            input_graphs: jraph.GraphsTuple,
            input_probe_graphs: jraph.GraphsTuple,
            input_wt_mask: jnp.ndarray,
            input_probe_mask: jnp.ndarray,
            _model=model,
            _params=params,
            _scale_rel_ws=scale_rel_ws,
            _graphs=graphs,
        ):
            """This function assumes the graphs are padded"""

            prediction = _model.apply(
                _params,
                input_graphs,
                input_probe_graphs,
                input_wt_mask,
                input_probe_mask,
            )
            if _scale_rel_ws:
                combined_mask = input_wt_mask + input_probe_mask
                prediction = inverse_scale_rel_ws(
                    _graphs,
                    prediction,
                    combined_mask,
                )
            return prediction

        _inverse_scale_target = jax.jit(unscaler.inverse_scale_output)

        pred_fn = jax.jit(model_prediction_fn)

        def test_errors_fn(
            input_graphs: jraph.GraphsTuple,
            raw_targets: jnp.ndarray,
            raw_prediction: jnp.ndarray,
            wt_mask: jnp.ndarray,
            probe_mask: jnp.ndarray,
            _scale_rel_ws=scale_rel_ws,
            _unscaler=unscaler,
        ):
            """This function assumes the graphs are padded"""
            prediction = raw_prediction.squeeze()
            if _scale_rel_ws:
                combined_mask = wt_mask + probe_mask
                prediction = inverse_scale_rel_ws(
                    input_graphs,
                    prediction,
                    combined_mask,
                )

            targets = raw_targets.squeeze()
            prediction = _unscaler.inverse_scale_output(raw_prediction.squeeze())
            targets = _unscaler.inverse_scale_output(raw_targets.squeeze())

            error = targets - prediction
            sq_error = error**2
            abs_error = jnp.abs(error)

            # Calculating the samples is based on the padding mask implemented in DeepOGraphNet see model for explanation
            n_samples = jnp.sum(wt_mask) + jnp.sum(probe_mask)
            return (
                prediction,
                targets,
                jnp.sum(error, axis=0),
                jnp.sum(sq_error, axis=0),
                jnp.sum(abs_error, axis=0),
                n_samples,
            )

        test_errors_fn_jit = jax.jit(test_errors_fn)

        print("#######################")
        print("#######################")
        print(f"Testing {model_type_str} model: {model_path}")
        print(f"Saved plots to {plot_folder_path}")
        print("#######################")
        print("#######################")

        errors = np.zeros(targets.shape[-1])
        sq_errors = np.copy(errors)
        abs_errors = np.copy(errors)
        ape_errors = np.copy(errors)
        n_samples = np.copy(errors).astype(np.int64)
        df_local_metrics = pd.DataFrame()

        for _i, batch in tqdm(enumerate(data_iterator)):
            if has_layout_info:
                graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = batch
            else:
                graphs, probe_graphs, node_array_tuple = batch
                layout_type, wt_spacing = None, None
            targets, wt_mask, probe_mask, positions = node_array_tuple

            raw_pred = pred_fn(graphs, probe_graphs, wt_mask, probe_mask)

            (
                prediction,
                targets,
                err_sum_,
                sq_err_sum_,
                abs_err_sum_,
                n_samples_,
            ) = test_errors_fn_jit(
                graphs, targets, raw_pred, wt_mask, probe_mask
            )  # probe_graphs, wt_mask, probe_mask, targets)
            unscaled_graphs = unscaler.inverse_scale_graph(graphs)
            assert unscaled_graphs.globals is not None, "Graph globals must be present"
            U_freestream = unscaled_graphs.globals[0, 0]  # type: ignore[index]
            TI_ambient = unscaled_graphs.globals[0, 1]  # type: ignore[index]
            n_wt = int(jnp.sum(wt_mask))
            ape = jnp.abs(targets - prediction) / U_freestream * 100
            ape_sum = jnp.sum(ape, axis=0)
            mape_local = jnp.mean(ape)
            rmse_local = jnp.sqrt(jnp.mean((targets - prediction) ** 2))
            local_row = {
                "U_freestream": float(U_freestream),
                "TI_ambient": float(TI_ambient),
                "n_edges": int(graphs.n_edge[0]),
                "n_probe_edges": int(probe_graphs.n_edge[0]),
                "n_wt": int(n_wt),
                "n_probes": int(jnp.sum(probe_mask)),
                "layout_type": str(layout_type[0]) if layout_type is not None else "unknown",
                "wt_spacing": wt_spacing[0].numpy() if wt_spacing is not None else np.nan,
                "mape_local": float(mape_local),
                "rmse_local": float(rmse_local),
            }
            df_local_metrics = pd.concat(
                [df_local_metrics, pd.DataFrame([local_row])], ignore_index=True
            )

            errors += np.float64(err_sum_)
            sq_errors += np.float64(sq_err_sum_)
            abs_errors += np.float64(abs_err_sum_)
            ape_errors += np.float64(ape_sum)
            n_samples += np.int64(n_samples_)

        # save local metrics
        df_local_metrics.to_csv(os.path.join(plot_folder_path, "local_metrics.csv"), index=False)

        mse = sq_errors / n_samples
        mae = abs_errors / n_samples
        mape = ape_errors / n_samples
        rmse = jnp.sqrt(mse)

        if len(cfg.data.io.target_node_features) > 1:
            # Turn lists into dictionaries to make them displayable in Wandb
            mse = dict(zip(cfg.data.io.target_node_features, mse))
            mae = dict(zip(cfg.data.io.target_node_features, mae))
            rmse = dict(zip(cfg.data.io.target_node_features, rmse))
            mape = dict(zip(cfg.data.io.target_node_features, mape))

        error_metrics[f"test/{model_type_str}/mse"] = mse
        error_metrics[f"test/{model_type_str}/mae"] = mae
        error_metrics[f"test/{model_type_str}/rmse"] = rmse
        error_metrics[f"test/{model_type_str}/mape"] = mape

        # Convert ndarrays to lists for JSON serialization
        error_metrics = convert_ndarray(error_metrics)

        # Save metrics JSON to plot folder (per checkpoint type)
        with open(os.path.join(plot_folder_path, "error_metrics.json"), "w") as f:
            json.dump({k: v for k, v in error_metrics.items() if model_type_str in k}, f, indent=4)

    # Save comprehensive metrics to model directory (all checkpoint types)
    model_dir = cfg.model_save_path
    with open(os.path.join(model_dir, "error_metrics_all.json"), "w") as f:
        json.dump(error_metrics, f, indent=4)

    if cfg.wandb.use and wandb_run is not None:
        # Log all available checkpoint metrics
        wandb_metrics = {}
        for ckpt_type in ["final", "best_mse", "best_mae", "best_hybrid"]:
            for metric in ["mse", "mae", "rmse", "mape"]:
                key = f"test/{ckpt_type}/{metric}"
                if key in error_metrics:
                    wandb_metrics[key] = error_metrics[key]
        if wandb_metrics:
            wandb_run.log(wandb_metrics)


if __name__ == "__main__":
    import warnings

    from hydra import compose, initialize

    from utils.misc import add_to_hydra_cfg

    warnings.simplefilter(action="ignore", category=FutureWarning)

    config_path = os.path.relpath(
        "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1/.hydra"
    )
    config_name = "config"
    output_dir = os.path.dirname(os.path.abspath(config_path))

    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name=config_name)

    cfg = add_to_hydra_cfg(cfg, "wandb", DictConfig({"use": False}))
    cfg = add_to_hydra_cfg(cfg, "model_save_path", os.path.join(output_dir, "model"))

    post_process_GNO_probe(cfg)
