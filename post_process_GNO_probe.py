import json
import multiprocessing as mp
import os
import pickle
import time
from functools import partial
from typing import Tuple

import jax
import jraph
import numpy as np
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_tools import (
    retrieve_dataset_stats,
    setup_test_val_iterator,
    setup_unscaler,
)
from utils.GNO_probe import inverse_scale_rel_ws, scale_rel_ws
from utils.misc import convert_ndarray, get_model_paths
from utils.model_tools import load_model, setup_model
from utils.plotting import (
    plot_crossstream_predictions,
    plot_loss_history,
    plot_probe_graph_fn,
)
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set by caller; ignore
    pass

jax.config.update("jax_default_device", jax.devices("cpu")[0])


def post_process_GNO_probe(cfg: DictConfig, wandb_run=None) -> None:
    """This function is Temporary it uses the training set to test on for quick predictions"""

    paths = get_model_paths(cfg)  # order is final then best

    error_metrics = {}

    for model_path, model_type_str in zip(paths, ["final", "best"]):
        # Load model
        if model_path is None:
            continue

        parent_path = os.path.dirname(model_path)
        if model_type_str == "best":
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
                os.path.join(
                    model_path.split("model")[0], f"{model_type_str}_params.msgpack"
                )
            )
            dataset = Torch_Geomtric_Dataset(
                root_path=os.path.abspath(cfg.data.test_path), in_mem=False
            )
            stats, scale_stats = retrieve_dataset_stats(dataset)
            params, cfg_model, model, _ = load_portable_model(
                params_path, cfg_json_path, dataset
            )
        else:
            model, params, metrics, cfg_model = load_model(model_path)
            model = setup_model(cfg_model)

        #! REMOVE THIS AFTER DEBUGGING/ TESTING
        cfg_model.data.test_path = "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
        dataset = Torch_Geomtric_Dataset(
            root_path=os.path.abspath(cfg.data.test_path), in_mem=True
        )

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

        if "scale_rel_ws" not in cfg_model:
            scale_rel_ws = False

        data_iterator = get_data_iterator()

        unscaler = setup_unscaler(cfg_model, scale_stats=scale_stats)

        graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = next(
            data_iterator
        )
        targets, wt_mask, probe_mask, positions = node_array_tuple

        # model_prediction_fn = jax.jit(partial(model.apply, params))

        def model_prediction_fn(
            input_graphs: jraph.GraphsTuple,
            input_probe_graphs: jraph.GraphsTuple,
            input_wt_mask: jnp.ndarray,
            input_probe_mask: jnp.ndarray,
        ) -> jnp.ndarray:
            """This function assumes the graphs are padded"""

            prediction = model.apply(
                params,
                input_graphs,
                input_probe_graphs,
                input_wt_mask,
                input_probe_mask,
            )
            if scale_rel_ws:
                prediction = inverse_scale_rel_ws(
                    graphs,
                    prediction,
                )
            return prediction

        _inverse_scale_target = jax.jit(unscaler.inverse_scale_output)

        pred_fn = jax.jit(model_prediction_fn)

        @jax.jit
        def test_errors_fn(
            input_graphs: jraph.GraphsTuple,
            raw_targets: jnp.ndarray,
            raw_prediction: jnp.ndarray,
            wt_mask: jnp.ndarray,
            probe_mask: jnp.ndarray,
        ) -> Tuple[float, float, int]:
            """This function assumes the graphs are padded"""
            prediction = raw_prediction.squeeze()
            if scale_rel_ws:
                prediction = inverse_scale_rel_ws(
                    input_graphs,
                    prediction,
                )

            targets = raw_targets.squeeze()
            prediction = unscaler.inverse_scale_output(raw_prediction.squeeze())
            targets = unscaler.inverse_scale_output(raw_targets.squeeze())

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

        for i, (
            graphs,
            probe_graphs,
            node_array_tuple,
            layout_type,
            wt_spacing,
        ) in tqdm(enumerate(data_iterator)):
            targets, wt_mask, probe_mask, positions = node_array_tuple

            raw_pred = pred_fn(graphs, probe_graphs, wt_mask, probe_mask)

            (
                prediction,
                targets,
                err_sum_,
                sq_err_sum_,
                abs_err_sum_,
                n_samples_,
            ) = test_errors_fn(
                graphs, targets, raw_pred, wt_mask, probe_mask
            )  # probe_graphs, wt_mask, probe_mask, targets)
            unscaled_graphs = unscaler.inverse_scale_graph(graphs)
            U_freestream = unscaled_graphs.globals[0, 0]
            TI_ambient = unscaled_graphs.globals[0, 1]
            n_wt = int(jnp.sum(wt_mask))
            ape = jnp.abs((targets - prediction)) / U_freestream * 100
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
                "layout_type": str(layout_type[0]),
                "wt_spacing": wt_spacing[0].numpy(),
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
        df_local_metrics.to_csv(
            os.path.join(plot_folder_path, "local_metrics.csv"), index=False
        )

        mse = sq_errors / n_samples
        mae = abs_errors / n_samples
        mape = ape_errors / n_samples
        rmse = jnp.sqrt(mse)

        if len(cfg.data.io.target_node_features) > 1:
            # Turn lists into dictionaries to make them displayable in Wandb
            mse = {
                var: mse_ for var, mse_ in zip(cfg.data.io.target_node_features, mse)
            }
            mae = {
                var: mae_ for var, mae_ in zip(cfg.data.io.target_node_features, mae)
            }
            rmse = {
                var: rmse_ for var, rmse_ in zip(cfg.data.io.target_node_features, rmse)
            }
            mape = {
                var: mape_ for var, mape_ in zip(cfg.data.io.target_node_features, mape)
            }

        error_metrics[f"test/{model_type_str}/mse"] = mse
        error_metrics[f"test/{model_type_str}/mae"] = mae
        error_metrics[f"test/{model_type_str}/rmse"] = rmse
        error_metrics[f"test/{model_type_str}/mape"] = mape

        # Assuming error_metrics is the dictionary containing ndarrays
        error_metrics = convert_ndarray(error_metrics)
        # save as json
        with open(os.path.join(parent_path, "error_metrics.json"), "w") as f:
            json.dump(error_metrics, f, indent=4)

        if cfg.wandb.use:
            wandb_run.log(
                {
                    "test/best/mse": error_metrics["test/best/mse"],
                    "test/best/mae": error_metrics["test/best/mae"],
                    "test/best/rmse": error_metrics["test/best/rmse"],
                    "test/best/mape": error_metrics["test/best/mape"],
                    "test/final/mse": error_metrics["test/final/mse"],
                    "test/final/mae": error_metrics["test/final/mae"],
                    "test/final/rmse": error_metrics["test/final/rmse"],
                    "test/final/mape": error_metrics["test/final/mape"],
                },
            )


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
