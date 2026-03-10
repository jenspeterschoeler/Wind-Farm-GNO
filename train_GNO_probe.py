"""Training pipeline for the GNO wind farm flow prediction model."""

import logging
import os
import warnings

# Configure JAX backend BEFORE importing JAX
# "cuda,cpu" means: try CUDA first, fall back to CPU if unavailable
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax

# Log which platform was actually selected
_gpu_available = any(d.platform == "gpu" for d in jax.devices())
if _gpu_available:
    print(f"GPU available: {jax.devices()}")
else:
    print(f"CPU only: {jax.devices()}")

import torch  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

import wandb  # noqa: E402
from utils import add_to_hydra_cfg  # noqa: E402
from utils.training_utils import (  # noqa: E402
    compute_hybrid_metric,
    create_prediction_fn,
    create_train_step_fn,
    create_val_errors_fn,
    create_wandb_plot,
    initialize_model_and_params,
    load_plot_components,
    load_resume_checkpoint,
    log_model_parameters,
    restore_early_stop_state,
    run_training_epoch,
    run_validation,
    save_final_model,
    save_resume_checkpoint_multi_metric,
    setup_checkpointing,
    setup_data_loaders,
    setup_multi_metric_checkpointing,
    setup_training_components,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


def train_GNO_probe(cfg: DictConfig, wandb_run=None, pretrained_checkpoint_path=None) -> DictConfig:
    """
    Train GNO probe model.

    Args:
        cfg: Configuration object
        wandb_run: Optional W&B run for logging
        pretrained_checkpoint_path: Optional path to pretrained checkpoint for transfer learning
    """
    # Setup data loaders
    train_dataset, get_refreshed_train_fn, get_refreshed_val_fn, cfg = setup_data_loaders(cfg)

    # Initialize model and parameters (with optional pretrained weights)
    first_batch = next(get_refreshed_train_fn())
    model, params, dropout_active, rng_key, cfg = initialize_model_and_params(
        cfg, first_batch, pretrained_checkpoint_path=pretrained_checkpoint_path
    )

    # Store pretrained params for fine-tuning (deep copy)
    pretrained_params = None
    if pretrained_checkpoint_path is not None:
        import copy

        pretrained_params = copy.deepcopy(params)
        logger.info("Pretrained params stored for fine-tuning")

    # Note: LoRA is now configured directly in model via use_lora_embedder/processor/decoder
    # parameters in the config (see utils/model_tools.py)

    # Log model parameters
    log_model_parameters(params, wandb_run)

    # Setup training components (pass pretrained_params for fine-tuning)
    train_state, optimizer, early_stop = setup_training_components(
        cfg, model, params, dropout_active, pretrained_params=pretrained_params
    )

    # Create JIT-compiled functions
    prediction_fn = create_prediction_fn(model)
    train_step_fn = create_train_step_fn(cfg, dropout_active)
    val_errors_fn = create_val_errors_fn(cfg, prediction_fn)

    # Load plot components for visualization (from pickle or validation data)
    plot_components = load_plot_components(cfg, get_refreshed_val_fn)

    # Setup checkpointing (multi-metric: MSE, MAE, hybrid)
    checkpoint_managers, orbax_checkpointer = setup_multi_metric_checkpointing(cfg)
    best_mse = float("inf")
    best_mae = float("inf")
    best_hybrid = float("inf")
    mse_baseline = None  # Set on first validation
    mae_baseline = None

    logger.info("Running main loop.")
    # Run the training loop
    loss_hist = []
    val_hist = []
    val_epochs = []
    pbar = tqdm(total=cfg.optimizer.n_epochs)
    pbar.set_description("Training")

    for epoch in range(cfg.optimizer.n_epochs):
        # Run training epoch
        train_loss, train_state, rng_key = run_training_epoch(
            train_state, get_refreshed_train_fn, train_step_fn, dropout_active, rng_key
        )
        loss_hist.append(train_loss)
        metrics = {"loss": loss_hist}

        if cfg.wandb.use and wandb_run is not None:
            wandb_run.log({"train/loss": loss_hist[-1]}, step=epoch)

        # Validation loop
        if epoch % cfg.optimizer.validation.rate_of_validation == 0:
            # Run validation
            val_metrics = run_validation(cfg, train_state, get_refreshed_val_fn, val_errors_fn)

            # Update metrics history
            metrics.update(val_metrics)
            val_hist.append(val_metrics["val_mse"])
            val_epochs.append(epoch)
            metrics["val_loss"] = val_hist
            metrics["val_epochs"] = val_epochs

            # Determine validation metric for checkpointing
            if cfg.model.scale_rel_ws:
                validation_metric = val_metrics["val_mse_scaled_rel_ws"]
            else:
                validation_metric = val_metrics["val_mse"]

            # W&B logging and plotting
            if cfg.wandb.use and wandb_run is not None:
                wandb_val_dict = {
                    "val/loss(mse)": val_metrics["val_mse"],
                    "val/mae": val_metrics["val_mae"],
                    "val/rmse": val_metrics["val_RMSE"],
                }
                if cfg.model.scale_rel_ws:
                    wandb_val_dict.update(
                        {
                            "val/loss(mse)_scaled_rel_ws": val_metrics["val_mse_scaled_rel_ws"],
                            "val/mae_scaled_rel_ws": val_metrics["val_mae_scaled_rel_ws"],
                            "val/rmse_scaled_rel_ws": val_metrics["val_rmse_scaled_rel_ws"],
                        }
                    )

                # Create and log visualization if plot components available
                if plot_components is not None:
                    fig = create_wandb_plot(cfg, model, train_state, plot_components)
                    plt.suptitle(f"Epoch: {epoch} - Predictions vs Targets")
                    wandb_run.log({"plot": wandb.Image(fig)}, step=epoch)
                    plt.close(fig)
                wandb_run.log(wandb_val_dict, step=epoch)

            # Update progress bar
            pbar_dict = {
                "Train Loss": f"{loss_hist[-1]:.8f}",
                "Val Loss": f"{validation_metric:.8f}",
            }
            if cfg.model.scale_rel_ws:
                pbar_dict.update(
                    {"Val Loss (scaled rel ws)": f"{val_metrics['val_mse_scaled_rel_ws']:.8f}"}
                )
            pbar.set_postfix(pbar_dict)

            # Checkpointing and early stopping
            if epoch >= cfg.optimizer.early_stop.start_epoch:
                # Get current metrics (use consistent scaling for MSE and MAE)
                if cfg.model.scale_rel_ws:
                    current_mse = val_metrics["val_mse_scaled_rel_ws"]
                    current_mae = val_metrics["val_mae_scaled_rel_ws"]
                else:
                    current_mse = val_metrics["val_mse"]
                    current_mae = val_metrics["val_mae"]

                # Initialize baselines on first validation after start_epoch
                if mse_baseline is None:
                    mse_baseline = current_mse
                if mae_baseline is None:
                    mae_baseline = current_mae

                # Compute hybrid metric (geometric mean of normalized MSE and MAE)
                current_hybrid = compute_hybrid_metric(
                    current_mse, current_mae, mse_baseline, mae_baseline
                )

                # Check and save best MSE model
                if current_mse < best_mse:
                    best_mse = current_mse
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_mse"].save(epoch, ckpt)
                    print(f"New best MSE model: {best_mse:.8f} at epoch {epoch}")

                # Check and save best MAE model
                if current_mae < best_mae:
                    best_mae = current_mae
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_mae"].save(epoch, ckpt)
                    print(f"New best MAE model: {best_mae:.8f} at epoch {epoch}")

                # Check and save best hybrid model
                if current_hybrid < best_hybrid:
                    best_hybrid = current_hybrid
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_hybrid"].save(epoch, ckpt)
                    print(
                        f"New best hybrid model: {best_hybrid:.4f} at epoch {epoch} "
                        f"(MSE={current_mse:.8f}, MAE={current_mae:.8f})"
                    )

                # Early stopping (still based on MSE for consistency)
                if "early_stop" in cfg.optimizer:
                    early_stop = early_stop.update(validation_metric)
                    if early_stop.should_stop:
                        logger.info(f"Met early stopping criteria, breaking at epoch {epoch}")
                        break
        else:
            pbar.set_postfix({"Train Loss": f"{loss_hist[-1]:.8f}"})

        pbar.update(1)

    pbar.close()

    # Save final model
    cfg = save_final_model(cfg, train_state, metrics, epoch, orbax_checkpointer)
    return cfg


def train_GNO_probe_resumable(
    cfg: DictConfig,
    wandb_run=None,
    pretrained_checkpoint_path=None,
    preemption_checker=None,
) -> DictConfig:
    """
    Train GNO probe model with resume capability for auto-resubmission.

    This function extends train_GNO_probe with automatic checkpoint resume
    and preemption handling for SLURM job timeout scenarios.

    Args:
        cfg: Configuration object
        wandb_run: Optional W&B run for logging
        pretrained_checkpoint_path: Optional path to pretrained checkpoint
        preemption_checker: Callable that returns True if preemption requested
            (e.g., SIGUSR1 received indicating job timeout imminent)

    Returns:
        Updated configuration after training
    """
    # Check for existing checkpoint to resume from
    checkpoint_path = os.path.join(cfg.model_save_path, "checkpoints")
    resume_data = load_resume_checkpoint(checkpoint_path)

    # Setup data loaders
    train_dataset, get_refreshed_train_fn, get_refreshed_val_fn, cfg = setup_data_loaders(cfg)

    # Determine starting state based on resume checkpoint
    if resume_data is not None and "resume_state" in resume_data:
        # ===== RESUME MODE =====
        logger.info("=" * 80)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("=" * 80)

        resume_state = resume_data["resume_state"]
        start_epoch = resume_state["epoch"] + 1  # Start from next epoch
        rng_key = jax.numpy.array(resume_state["rng_key"])

        # Restore multi-metric state (with backward compatibility)
        best_mse = resume_state.get("best_mse", resume_state.get("best_metric", float("inf")))
        best_mae = resume_state.get("best_mae", float("inf"))
        best_hybrid = resume_state.get("best_hybrid", float("inf"))
        mse_baseline = resume_state.get("mse_baseline", float("inf"))
        mae_baseline = resume_state.get("mae_baseline", float("inf"))

        loss_hist = resume_state["loss_hist"]
        val_hist = resume_state["val_hist"]
        val_epochs = resume_state["val_epochs"]

        # Restore config from checkpoint (maintains consistency)
        saved_cfg = DictConfig(resume_data["config"])
        # Keep model_save_path from current cfg (output directory)
        model_save_path = cfg.model_save_path
        cfg = saved_cfg
        cfg = add_to_hydra_cfg(cfg, "model_save_path", model_save_path)

        # Restore early stopping with preserved counter
        early_stop = restore_early_stop_state(resume_state.get("early_stop_state"), cfg)

        # Re-initialize model for apply_fn (need fresh batch for shape info)
        first_batch = next(get_refreshed_train_fn())
        model, _, dropout_active, _, cfg = initialize_model_and_params(
            cfg, first_batch, pretrained_checkpoint_path=pretrained_checkpoint_path
        )

        # Restore train_state from checkpoint
        from utils import setup_optimizer

        optimizer = setup_optimizer(cfg)

        if dropout_active:

            def apply_fn(params, graphs, probe_graphs, wt_mask, probe_mask, rngs):
                return model.apply(
                    params, graphs, probe_graphs, wt_mask, probe_mask, train=True, rngs=rngs
                )
        else:

            def apply_fn(params, graphs, probe_graphs, wt_mask, probe_mask):
                return model.apply(params, graphs, probe_graphs, wt_mask, probe_mask, train=True)

        from flax.training.train_state import TrainState

        # Extract params and opt_state from saved train_state
        saved_train_state = resume_data["train_state"]
        train_state = TrainState.create(
            apply_fn=apply_fn,
            params=saved_train_state.params,
            tx=optimizer,
        )
        # Restore optimizer state
        train_state = train_state.replace(opt_state=saved_train_state.opt_state)

        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best MSE so far: {best_mse:.8f}")
        logger.info(f"Best MAE so far: {best_mae:.8f}")
        logger.info(f"Best Hybrid so far: {best_hybrid:.4f}")
        logger.info(f"Training history: {len(loss_hist)} epochs completed")
        if early_stop is not None:
            logger.info(f"Early stop patience count: {early_stop.patience_count}")
        logger.info("=" * 80)

    else:
        # ===== FRESH START =====
        logger.info("Starting fresh training (no checkpoint found)")

        # Initialize model and parameters
        first_batch = next(get_refreshed_train_fn())
        model, params, dropout_active, rng_key, cfg = initialize_model_and_params(
            cfg, first_batch, pretrained_checkpoint_path=pretrained_checkpoint_path
        )

        # Store pretrained params for fine-tuning (deep copy)
        pretrained_params = None
        if pretrained_checkpoint_path is not None:
            import copy

            pretrained_params = copy.deepcopy(params)
            logger.info("Pretrained params stored for fine-tuning (weight anchoring)")

        # Log model parameters
        log_model_parameters(params, wandb_run)

        # Setup training components
        train_state, optimizer, early_stop = setup_training_components(
            cfg, model, params, dropout_active, pretrained_params=pretrained_params
        )

        start_epoch = 0
        best_mse = float("inf")
        best_mae = float("inf")
        best_hybrid = float("inf")
        mse_baseline = None
        mae_baseline = None
        loss_hist = []
        val_hist = []
        val_epochs = []

    # Create JIT-compiled functions
    prediction_fn = create_prediction_fn(model)
    train_step_fn = create_train_step_fn(cfg, dropout_active)
    val_errors_fn = create_val_errors_fn(cfg, prediction_fn)

    # Load plot components for visualization
    plot_components = load_plot_components(cfg, get_refreshed_val_fn)

    # Setup multi-metric checkpointing (MSE, MAE, hybrid)
    checkpoint_managers, orbax_checkpointer = setup_multi_metric_checkpointing(cfg)

    # Setup resume checkpointing (for preemption handling)
    resume_checkpoint_manager, _ = setup_checkpointing(cfg)

    logger.info("Running main loop.")
    pbar = tqdm(total=cfg.optimizer.n_epochs, initial=start_epoch)
    pbar.set_description("Training")

    # Main training loop
    for epoch in range(start_epoch, cfg.optimizer.n_epochs):
        # Check for preemption signal (SLURM timeout imminent)
        if preemption_checker is not None and preemption_checker():
            logger.warning(f"Preemption requested at epoch {epoch}, saving checkpoint...")
            metrics = {"loss": loss_hist, "val_loss": val_hist, "val_epochs": val_epochs}
            save_resume_checkpoint_multi_metric(
                cfg,
                resume_checkpoint_manager,
                train_state,
                metrics,
                epoch,
                rng_key,
                best_mse,
                best_mae,
                best_hybrid,
                mse_baseline,
                mae_baseline,
                early_stop,
                loss_hist,
                val_hist,
                val_epochs,
                is_preemption=True,
            )
            logger.info("Checkpoint saved, exiting for resubmission")
            pbar.close()
            raise SystemExit(0)  # Clean exit for submitit resubmission

        # Run training epoch
        train_loss, train_state, rng_key = run_training_epoch(
            train_state, get_refreshed_train_fn, train_step_fn, dropout_active, rng_key
        )
        loss_hist.append(train_loss)
        metrics = {"loss": loss_hist}

        if cfg.wandb.use and wandb_run is not None:
            wandb_run.log({"train/loss": loss_hist[-1]}, step=epoch)

        # Validation loop
        if epoch % cfg.optimizer.validation.rate_of_validation == 0:
            # Run validation
            val_metrics = run_validation(cfg, train_state, get_refreshed_val_fn, val_errors_fn)

            # Update metrics history
            metrics.update(val_metrics)
            val_hist.append(val_metrics["val_mse"])
            val_epochs.append(epoch)
            metrics["val_loss"] = val_hist
            metrics["val_epochs"] = val_epochs

            # Determine validation metric for checkpointing
            if cfg.model.scale_rel_ws:
                validation_metric = val_metrics["val_mse_scaled_rel_ws"]
            else:
                validation_metric = val_metrics["val_mse"]

            # W&B logging and plotting
            if cfg.wandb.use and wandb_run is not None:
                wandb_val_dict = {
                    "val/loss(mse)": val_metrics["val_mse"],
                    "val/mae": val_metrics["val_mae"],
                    "val/rmse": val_metrics["val_RMSE"],
                }
                if cfg.model.scale_rel_ws:
                    wandb_val_dict.update(
                        {
                            "val/loss(mse)_scaled_rel_ws": val_metrics["val_mse_scaled_rel_ws"],
                            "val/mae_scaled_rel_ws": val_metrics["val_mae_scaled_rel_ws"],
                            "val/rmse_scaled_rel_ws": val_metrics["val_rmse_scaled_rel_ws"],
                        }
                    )

                # Create and log visualization if plot components available
                if plot_components is not None:
                    fig = create_wandb_plot(cfg, model, train_state, plot_components)
                    plt.suptitle(f"Epoch: {epoch} - Predictions vs Targets")
                    wandb_run.log({"plot": wandb.Image(fig)}, step=epoch)
                    plt.close(fig)
                wandb_run.log(wandb_val_dict, step=epoch)

            # Update progress bar
            pbar_dict = {
                "Train Loss": f"{loss_hist[-1]:.8f}",
                "Val Loss": f"{validation_metric:.8f}",
            }
            if cfg.model.scale_rel_ws:
                pbar_dict.update(
                    {"Val Loss (scaled rel ws)": f"{val_metrics['val_mse_scaled_rel_ws']:.8f}"}
                )
            pbar.set_postfix(pbar_dict)

            # Checkpointing and early stopping
            if epoch >= cfg.optimizer.early_stop.start_epoch:
                # Get current metrics (use consistent scaling for MSE and MAE)
                if cfg.model.scale_rel_ws:
                    current_mse = val_metrics["val_mse_scaled_rel_ws"]
                    current_mae = val_metrics["val_mae_scaled_rel_ws"]
                else:
                    current_mse = val_metrics["val_mse"]
                    current_mae = val_metrics["val_mae"]

                # Initialize baselines on first validation after start_epoch
                if mse_baseline is None:
                    mse_baseline = current_mse
                if mae_baseline is None:
                    mae_baseline = current_mae

                # Compute hybrid metric
                current_hybrid = compute_hybrid_metric(
                    current_mse, current_mae, mse_baseline, mae_baseline
                )

                # Check and save best MSE model
                if current_mse < best_mse:
                    best_mse = current_mse
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_mse"].save(epoch, ckpt)
                    print(f"New best MSE model: {best_mse:.8f} at epoch {epoch}")

                # Check and save best MAE model
                if current_mae < best_mae:
                    best_mae = current_mae
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_mae"].save(epoch, ckpt)
                    print(f"New best MAE model: {best_mae:.8f} at epoch {epoch}")

                # Check and save best hybrid model
                if current_hybrid < best_hybrid:
                    best_hybrid = current_hybrid
                    ckpt = {
                        "train_state": train_state,
                        "config": OmegaConf.to_container(cfg),
                        "metrics": metrics,
                    }
                    checkpoint_managers["best_hybrid"].save(epoch, ckpt)
                    print(
                        f"New best hybrid model: {best_hybrid:.4f} at epoch {epoch} "
                        f"(MSE={current_mse:.8f}, MAE={current_mae:.8f})"
                    )

                # Save resume checkpoint periodically (for preemption recovery)
                # Save every time we improve any metric
                if (
                    current_mse <= best_mse
                    or current_mae <= best_mae
                    or current_hybrid <= best_hybrid
                ):
                    save_resume_checkpoint_multi_metric(
                        cfg,
                        resume_checkpoint_manager,
                        train_state,
                        metrics,
                        epoch,
                        rng_key,
                        best_mse,
                        best_mae,
                        best_hybrid,
                        mse_baseline,
                        mae_baseline,
                        early_stop,
                        loss_hist,
                        val_hist,
                        val_epochs,
                        is_preemption=False,
                    )

                # Early stopping (still based on MSE for consistency)
                if "early_stop" in cfg.optimizer and early_stop is not None:
                    early_stop = early_stop.update(validation_metric)
                    if early_stop.should_stop:
                        logger.info(f"Met early stopping criteria, breaking at epoch {epoch}")
                        break
        else:
            pbar.set_postfix({"Train Loss": f"{loss_hist[-1]:.8f}"})

        pbar.update(1)

    pbar.close()

    # Save final model
    cfg = save_final_model(cfg, train_state, metrics, epoch, orbax_checkpointer)
    return cfg


if __name__ == "__main__":
    import warnings

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    import wandb

    warnings.simplefilter(action="ignore", category=FutureWarning)

    config_path = os.path.relpath(os.path.join(os.path.dirname(__file__), "configurations"))
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
        wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    else:
        wandb_run = None

    # Check for pretrained checkpoint path in config (for transfer learning)
    pretrained_checkpoint_path = cfg.get("pretrained_checkpoint_path", None)

    train_GNO_probe(cfg, wandb_run=wandb_run, pretrained_checkpoint_path=pretrained_checkpoint_path)
