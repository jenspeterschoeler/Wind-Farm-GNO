"""
Main Training Entry Point

This unified entry point supports both standard training and resumable training
with checkpoint-based auto-resubmission for SLURM clusters.

The training mode is controlled by the `optimizer.resumable` config flag:
- resumable: false (default) - Standard training without auto-resume
- resumable: true - Checkpoint-based training with SIGUSR1 signal handling

Usage:
    # Standard training (default configs have resumable: false)
    python main.py --config-name test_GNO_probe

    # Resumable training for SLURM (awf_train has resumable: true)
    python main.py --config-name awf_train

    # Local testing of resumable training
    python main.py --config-name awf_train_local data=mini_graphs_nodes optimizer.n_epochs=5

    # Override resumable flag on command line
    python main.py --config-name test_GNO_probe optimizer.resumable=true
"""

import logging
import os
import warnings

import hydra
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf

import wandb
from train_GNO_probe import train_GNO_probe
from utils import add_to_hydra_cfg, get_run_info

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="configurations",
    config_name="test_GNO_probe",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point supporting both standard and resumable training.

    Training mode is controlled by cfg.optimizer.resumable:
    - False: Standard training using train_GNO_probe()
    - True: Resumable training using ResumableGNOTraining wrapper
    """
    # Append run info to the config
    cfg = add_to_hydra_cfg(cfg, "run_info", get_run_info())
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    cfg = add_to_hydra_cfg(cfg, "model_save_path", os.path.join(output_dir, "model"))

    # Check for pretrained checkpoint path (for transfer learning)
    pretrained_checkpoint_path = cfg.get("pretrained_checkpoint_path", None)

    # Check if resumable training is enabled
    resumable = cfg.optimizer.get("resumable", False)

    if resumable:
        # === RESUMABLE TRAINING MODE ===
        # Import here to avoid loading submitit when not needed
        from utils.resumable_training import ResumableGNOTraining

        logger.info("Using resumable training mode (optimizer.resumable=true)")

        # Initialize W&B with resume support
        if cfg.wandb.use:
            run_name = output_dir
            stored_run_id = cfg.get("wandb_run_id", None)

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                resume="allow",  # Allow resuming existing run
                id=stored_run_id,  # Use stored ID if available
            )
            wandb_run.config.update(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                allow_val_change=True,  # Allow updating config on resume
            )

            # Store run ID in config for future resubmissions
            cfg = add_to_hydra_cfg(cfg, "wandb_run_id", wandb_run.id)

            logger.info(f"W&B run initialized: {wandb_run.name} (id={wandb_run.id})")
            if stored_run_id:
                logger.info("Resuming W&B run from previous submission")
        else:
            wandb_run = None

        # Create resumable training wrapper and execute
        training_callable = ResumableGNOTraining()
        cfg_out = training_callable(
            cfg,
            wandb_run=wandb_run,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
        )

    else:
        # === STANDARD TRAINING MODE ===
        logger.info("Using standard training mode (optimizer.resumable=false)")

        # Initialize W&B (standard mode without resume support)
        if cfg.wandb.use:
            run_name = output_dir
            wandb_run = wandb.init(project=cfg.wandb.project, name=run_name)
            wandb_run.config.update(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        else:
            wandb_run = None

        # Select training function based on data.io.type
        if cfg.data.io.type == "GNO_probe":
            train_fn = train_GNO_probe
        else:
            raise ValueError(f"Invalid data.io.type: {cfg.data.io.type}")

        # Execute training
        cfg_out = train_fn(cfg, wandb_run, pretrained_checkpoint_path=pretrained_checkpoint_path)

    assert cfg_out is not None, "Training function must return a config"
    logger.info("Training completed successfully")

    # Return None for Hydra multirun (required)
    return None


if __name__ == "__main__":
    main()
