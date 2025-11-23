import logging
import os
import random
import time
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from post_process_GNO_probe import post_process_GNO_probe
from train_GNO_probe import train_GNO_probe
from utils import add_to_hydra_cfg, get_run_info

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:

    ### Random sleep for running multiple jobs on a shared cluster
    # # Generate a random sleep time (e.g., between 0 and 50 seconds)
    # sleep_time = random.uniform(0, 5) * 10

    # print(f"Sleeping for {sleep_time:.2f} seconds before loading data...")
    # time.sleep(sleep_time)

    #### Append run info to the config
    cfg = add_to_hydra_cfg(cfg, "run_info", get_run_info())
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    cfg = add_to_hydra_cfg(cfg, "model_save_path", os.path.join(output_dir, "model"))

    if cfg.wandb.use:
        run_name = output_dir
        wandb_run = wandb.init(project=cfg.wandb.project, name=run_name)
        wandb_run.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
    else:
        wandb_run = None

    #### Select the training and post-processing functions based on the data.io.type
    if cfg.data.io.type == "GNO_probe":
        train_fn = train_GNO_probe
        post_process_fn = post_process_GNO_probe
    else:
        raise ValueError(f"Invalid data.io.type: {cfg.data.io.type}")

    #### Train the model
    logger.info("Training the model")
    cfg_out = train_fn(cfg, wandb_run)

    #### Post process the results
    logger.info("Applying the model to the test set")
    post_process_fn(cfg_out, wandb_run)


def start_main(config_path: str, config_name: str, version_base="1.3"):
    # A wrapper function to set Hydra's config path and name dynamically
    @hydra.main(
        config_path=config_path, config_name=config_name, version_base=version_base
    )
    def hydra_wrapped_main(cfg: DictConfig) -> None:
        main(cfg)

    hydra_wrapped_main()


if __name__ == "__main__":
    from hydra import compose, initialize

    config_path = os.path.relpath(
        os.path.join(os.path.dirname(__file__), "configurations")
    )
    # config_name = "GNO_probe_grid_model_var"
    config_name = "test_GNO_probe"
    # output_dir = os.path.dirname(os.path.abspath(config_path))

    # with initialize(version_base="1.3", config_path=config_path):
    #     cfg = compose(config_name=config_name)

    logger.info(f"Starting main with config_name: {config_name}")
    logger.info(f"Starting main with config_path: {config_path}")
    start_main(config_name=config_name, config_path=config_path)
