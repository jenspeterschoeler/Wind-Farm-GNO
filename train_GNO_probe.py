import logging
import os
import pickle
import sys
import warnings
from copy import deepcopy
from typing import Tuple

import jax

if any(d.platform == "gpu" for d in jax.devices()):
    print("GPU available")
    os.environ["JAX_PLATFORMS"] = "gpu"
else:
    print("CPU only")
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import jraph
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from utils import (
    add_to_hydra_cfg,
    convert_to_wandb_format,
    dynamically_batch_graph_probe_operator,
    setup_optimizer,
)
from utils.data_tools import (
    setup_refresh_iterator,
    setup_test_val_iterator,
    setup_train_dataset,
)
from utils.GNO_probe import initialize_GNO_probe, inverse_scale_rel_ws, scale_rel_ws
from utils.model_tools import model_parameter_stats, setup_model
from utils.plotting import plot_crossstream_predictions, plot_probe_graph_fn

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


def train_GNO_probe(cfg: DictConfig, wandb_run=None) -> None:

    #### Load the dataset
    train_dataset, cfg = setup_train_dataset(cfg)
    get_refreshed_train_fn, unpadded_train_iterator = setup_refresh_iterator(
        cfg, train_dataset
    )
    get_refreshed_val_fn, val_dataset, _, _ = setup_test_val_iterator(
        cfg, type_str="val"
    )

    #### Create a data iterator
    train_iterator = get_refreshed_train_fn()

    # Get the first batch for initialization
    graphs, probe_graphs, node_array_tuple = next(train_iterator)
    targets, wt_mask, probe_mask = node_array_tuple

    cfg.model = add_to_hydra_cfg(
        cfg.model,
        "output_shape",
        targets.shape[-1],
    )
    model = setup_model(cfg)

    rng_key = jax.random.PRNGKey(0)
    params, dropout_active = initialize_GNO_probe(
        cfg,
        model,
        rng_key,
        graphs,
        probe_graphs,
        wt_mask,
        probe_mask,
    )

    params_stats = model_parameter_stats(params)
    logger.info(f"Total parameters: {params_stats['total_params']}")
    if wandb_run is not None:
        wandb_params_stats = convert_to_wandb_format(params_stats)
        for dict_key, value in wandb_params_stats.items():
            wandb.summary[f"params/{dict_key}"] = value

    # Create an optimizer
    optimizer = setup_optimizer(cfg)
    if "early_stop" in cfg.optimizer:
        early_stop = EarlyStopping(
            min_delta=cfg.optimizer.early_stop.criteria,
            patience=int(
                cfg.optimizer.early_stop.patience
                / cfg.optimizer.validation.rate_of_validation
            ),
        )

    # Create a train step function
    if dropout_active:
        train_state = TrainState.create(
            apply_fn=lambda params, graphs, probe_graphs, wt_mask, probe_mask, rngs: model.apply(
                params,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                train=True,
                rngs=rngs,
            ),
            params=params,
            tx=optimizer,
        )
    else:
        train_state = TrainState.create(
            apply_fn=lambda params, graphs, probe_graphs, wt_mask, probe_mask: model.apply(
                params,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                train=True,
            ),
            params=params,
            tx=optimizer,
        )
    prediction_fn = jax.jit(
        lambda params, graphs, probe_graphs, wt_mask, probe_mask: model.apply(
            params,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
            train=False,
        )
    )

    @jax.jit
    def train_step_fn(
        train_state: TrainState,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask: jnp.ndarray,
        probe_mask: jnp.ndarray,
        targets: jnp.ndarray,
        rngs: dict = None,
    ) -> TrainState:

        if cfg.model.scale_rel_ws:
            combined_mask = wt_mask + probe_mask
            targets = scale_rel_ws(graphs, targets, combined_mask)

        if dropout_active:

            def loss_fn(params):
                prediction = train_state.apply_fn(
                    params,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                    rngs=rngs,
                )
                loss = jnp.mean((targets - prediction) ** 2)
                return loss, prediction

        else:

            def loss_fn(params):
                prediction = train_state.apply_fn(
                    params,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                )
                loss = jnp.mean((targets - prediction) ** 2)
                return loss, prediction

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, prediction), grads = grad_fn(train_state.params)

        if cfg.model.scale_rel_ws:
            prediction = inverse_scale_rel_ws(graphs, prediction, combined_mask)

        new_train_state = train_state.apply_gradients(grads=grads)
        return loss, new_train_state, prediction

    ## Create a prediction method to be used in the validation loop
    @jax.jit
    def val_errors_fn(
        train_state: TrainState,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask,
        probe_mask,
        targets: jnp.ndarray,
    ) -> Tuple[float, float, int]:
        """This function assumes the graphs are padded"""
        predictions = prediction_fn(
            train_state.params,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
        )
        if cfg.model.scale_rel_ws:
            combined_mask = wt_mask + probe_mask
            predictions_scaled_rel_ws = predictions
            targets_scaled_rel_ws = scale_rel_ws(graphs, targets, combined_mask)
            errors_scaled_rel_ws = targets_scaled_rel_ws - predictions_scaled_rel_ws
            sq_errors_scaled_rel_ws = errors_scaled_rel_ws**2
            abs_errors_scaled_rel_ws = jnp.abs(errors_scaled_rel_ws)
            predictions = inverse_scale_rel_ws(graphs, predictions, combined_mask)

        errors = targets - predictions
        sq_errors = errors**2
        abs_errors = jnp.abs(errors)

        # Calculating the samples is based on the padding mask implemented in DeepOGraphNet see model for explanation
        n_samples = jnp.sum(wt_mask) + jnp.sum(probe_mask)
        # proto_mask = jnp.sum(jnp.abs(probe_graphs.nodes), axis=-1)
        # proto_mask = proto_mask + jnp.sum(proto_mask, axis=-1).reshape(-1, 1)
        # n_samples = jnp.sum(jnp.bool(proto_mask))
        if cfg.model.scale_rel_ws:
            return (
                jnp.sum(errors_scaled_rel_ws),
                jnp.sum(sq_errors_scaled_rel_ws),
                jnp.sum(abs_errors_scaled_rel_ws),
                jnp.sum(errors),
                jnp.sum(sq_errors),
                jnp.sum(abs_errors),
                n_samples,
            )
        else:
            return (
                jnp.sum(errors),
                jnp.sum(sq_errors),
                jnp.sum(abs_errors),
                n_samples,
            )

    # Open the pickle file to check the contents
    with open(os.path.join(cfg.data.main_path, "plot_graph_components.pkl"), "rb") as f:
        loaded_components = pickle.load(f)
    (
        plot_jraph_graph,
        plot_probe_graph,
        plot_node_positions,
        plot_probe_targets,
        plot_wt_mask,
        plot_probe_mask,
        plot_y_grid,
    ) = loaded_components.values()
    plot_probe_targets = (
        plot_probe_targets * cfg.data.scale_stats["velocity"]["range"][0]
    )
    plot_U = np.array(
        plot_jraph_graph.globals[0, 0].squeeze()
        * cfg.data.scale_stats["velocity"]["range"][0]
    )
    plot_TI = np.array(
        plot_jraph_graph.globals[0, 1].squeeze()
        * cfg.data.scale_stats["ti"]["range"][0]
    )

    # ## Create a checkpoint manager for saving the best model
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_dir = os.path.join(cfg.model_save_path, "checkpoints")
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )
    best_metric = float("inf")  # Replace with -inf for metrics where higher is better

    logger.info("Running main loop.")
    # Run the training loop
    loss_hist = []
    val_hist = []
    val_epochs = []
    pbar = tqdm(total=cfg.optimizer.n_epochs)
    pbar.set_description("Training")
    for epoch in range(cfg.optimizer.n_epochs):
        train_loss = 0
        # Shuffle and restart the iterator
        train_iterator = get_refreshed_train_fn()
        for i, (graphs, probe_graphs, node_array_tuple) in enumerate(train_iterator):
            targets, wt_mask, probe_mask = node_array_tuple

            if dropout_active:
                rng_key, params_key, dropout_key = jax.random.split(rng_key, 3)
                batch_loss, train_state, predictions = train_step_fn(
                    train_state,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                    targets,
                    rngs={"params": params_key, "dropout": dropout_key},
                )
            else:
                batch_loss, train_state, predictions = train_step_fn(
                    train_state,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                    targets,
                )
            logger.debug(f"Prediction: {predictions}", f"\nTrue output: {targets}")
            train_loss += batch_loss  # .item()

        loss_hist.append(train_loss / (i + 1))
        metrics = {"loss": loss_hist}

        if cfg.wandb.use:
            wandb_run.log({"train/loss": loss_hist[-1]}, step=epoch)

        # Validation loop
        if epoch % cfg.optimizer.validation.rate_of_validation == 0:
            val_iterator = get_refreshed_val_fn()

            errors = np.float64(0)
            sq_errors = np.float64(0)
            abs_errors = np.float64(0)
            n_samples = np.int64(0)
            if cfg.model.scale_rel_ws:
                errors_scaled_rel_ws = np.float64(0)
                sq_errors_scaled_rel_ws = np.float64(0)
                abs_errors_scaled_rel_ws = np.float64(0)

            for i, (graphs, probe_graphs, node_array_tuple) in enumerate(val_iterator):
                targets, wt_mask, probe_mask = node_array_tuple

                val_err_output = val_errors_fn(
                    train_state,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                    targets,
                )
                if cfg.model.scale_rel_ws:
                    (
                        err_sum_scaled_rel_ws_,
                        sq_err_sum_scaled_rel_ws_,
                        abs_err_sum_scaled_rel_ws_,
                        err_sum_,
                        sq_err_sum_,
                        abs_err_sum_,
                        n_samples_,
                    ) = val_err_output

                    errors_scaled_rel_ws += np.float64(err_sum_scaled_rel_ws_)
                    sq_errors_scaled_rel_ws += np.float64(sq_err_sum_scaled_rel_ws_)
                    abs_errors_scaled_rel_ws += np.float64(abs_err_sum_scaled_rel_ws_)
                else:
                    (
                        err_sum_,
                        sq_err_sum_,
                        abs_err_sum_,
                        n_samples_,
                    ) = val_err_output

                errors += np.float64(err_sum_)
                sq_errors += np.float64(sq_err_sum_)
                abs_errors += np.float64(abs_err_sum_)
                n_samples += np.int64(n_samples_)

            metrics["val_mse"] = sq_errors / n_samples
            metrics["val_mae"] = abs_errors / n_samples
            metrics["val_RMSE"] = jnp.sqrt(metrics["val_mse"])
            val_hist.append(metrics["val_mse"])
            val_epochs.append(epoch)
            metrics["val_loss"] = val_hist
            metrics["val_epochs"] = val_epochs

            if cfg.model.scale_rel_ws:
                metrics["val_mse_scaled_rel_ws"] = sq_errors_scaled_rel_ws / n_samples
                metrics["val_mae_scaled_rel_ws"] = abs_errors_scaled_rel_ws / n_samples
                metrics["val_rmse_scaled_rel_ws"] = jnp.sqrt(
                    metrics["val_mse_scaled_rel_ws"]
                )
                validation_metric = metrics["val_mse_scaled_rel_ws"]
            else:
                validation_metric = metrics["val_mse"]

            if cfg.wandb.use:
                wandb_val_dict = {
                    "val/loss(mse)": metrics["val_mse"],
                    "val/mae": metrics["val_mae"],
                    "val/rmse": metrics["val_RMSE"],
                }
                if cfg.model.scale_rel_ws:
                    wandb_val_dict.update(
                        {
                            "val/loss(mse)_scaled_rel_ws": metrics[
                                "val_mse_scaled_rel_ws"
                            ],
                            "val/mae_scaled_rel_ws": metrics["val_mae_scaled_rel_ws"],
                            "val/rmse_scaled_rel_ws": metrics["val_rmse_scaled_rel_ws"],
                        }
                    )

                plot_probe_predictions = model.apply(
                    train_state.params,
                    plot_jraph_graph,
                    plot_probe_graph,
                    plot_wt_mask,
                    plot_probe_mask,
                    train=False,
                )
                # Remove non probe nodes from predictions
                plot_probe_predictions = np.where(
                    plot_probe_mask == 1, plot_probe_predictions, np.nan
                )
                plot_probe_predictions = plot_probe_predictions[
                    ~np.isnan(plot_probe_predictions)
                ]
                plot_probe_predictions = (
                    plot_probe_predictions
                    * cfg.data.scale_stats["velocity"]["range"][0]
                )
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                plot_probe_graph_fn(
                    plot_jraph_graph,
                    plot_probe_graph,
                    plot_node_positions,
                    include_probe_edges=False,
                    ax=axes[0],
                )

                plot_crossstream_predictions(
                    np.array(plot_probe_predictions),
                    np.array(plot_probe_targets),
                    np.array(plot_y_grid),
                    ax=axes[1],
                )
                axes[1].set_xlim(plot_U - 1, plot_U + 1)
                # add a text box on the left side of the plot
                axes[1].set_title(
                    f"min prediction: {np.min(plot_probe_predictions):.2f}\t, max prediction: {np.max(plot_probe_predictions):.2f}, \t, mean prediction: {np.mean(plot_probe_predictions):.2f}",
                )
                plt.suptitle(
                    f"Epoch: {epoch}, probe predictions flowcase [U and TI]{np.round(plot_U,2)} {np.round(plot_TI,2)}",
                )
                wandb_run.log(
                    {"plot": wandb.Image(fig)},
                    step=epoch,
                )
                plt.close(fig)
                wandb_run.log(
                    wandb_val_dict,
                    step=epoch,
                )

            pbar_dict = {
                "Train Loss": f"{loss_hist[-1]:.8f}",
                "Val Loss": f"{validation_metric:.8f}",
            }
            if cfg.model.scale_rel_ws:
                pbar_dict.update(
                    {
                        "Val Loss (scaled rel ws)": f"{metrics['val_mse_scaled_rel_ws']:.8f}",
                    }
                )

            pbar.set_postfix(pbar_dict)
            if epoch >= cfg.optimizer.early_stop.start_epoch:
                # Save the model if it is the best so far
                if validation_metric < best_metric:
                    best_metric = validation_metric
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_manager.save(
                        epoch, ckpt
                    )  # Save only when performance improves
                    print(
                        f"New best model found at epoch {epoch} with metric {best_metric}. Checkpoint saved."
                    )

                ## Early stopping
                if "early_stop" in cfg.optimizer:
                    early_stop = early_stop.update(
                        validation_metric
                    )  # TODO, Could add something about training loss here as well
                    if early_stop.should_stop:
                        logger.info(
                            f"Met early stopping criteria, breaking at epoch {epoch}"
                        )
                        break
        else:
            pbar.set_postfix({"Train Loss": f"{loss_hist[-1]:.8f}"})
        pbar.update(1)

    pbar.close()

    cfg = add_to_hydra_cfg(
        cfg, "final_model_path", os.path.join(cfg.model_save_path, f"final_e_{epoch}")
    )
    ## Save final model
    ckpt = {
        "train_state": train_state,
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
    }
    orbax_checkpointer.save(
        os.path.join(cfg.model_save_path, f"final_e_{epoch}"),
        ckpt,
        force=True,  # force overwrites exsiting files
    )
    return cfg


if __name__ == "__main__":

    import warnings

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    import wandb

    warnings.simplefilter(action="ignore", category=FutureWarning)

    config_path = os.path.relpath(
        os.path.join(os.path.dirname(__file__), "configurations")
    )
    config_name = "test_GNO_probe"
    output_dir = os.path.dirname(os.path.abspath(config_path))
    with initialize(version_base="1.3", config_path=config_path):
        cfg_raw = compose(config_name=config_name, return_hydra_config=True)
        hydra_cfg = cfg_raw["hydra"]
        cfg = compose(config_name=config_name, return_hydra_config=False)

    output_dir = hydra_cfg["run"]["dir"]
    cfg = add_to_hydra_cfg(
        cfg, "model_save_path", os.path.join(os.path.abspath(output_dir), "model")
    )
    if cfg.wandb.use:
        run_name = output_dir
        wandb_run = wandb.init(project=cfg.wandb.project, name=run_name)
        wandb_run.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
    else:
        wandb_run = None

    train_GNO_probe(cfg, wandb_run=wandb_run)
