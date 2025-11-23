import logging
import os
import platform
import subprocess
from datetime import datetime

import jax
import numpy as np
import optax
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_run_info():
    """Get the run info for the experiment"""
    run_info = {
        "TimeInitiatedTraining": datetime.now().replace(microsecond=0).isoformat(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "available_devices": str(jax.devices()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip(),
    }
    return run_info


def add_to_hydra_cfg(cfg, key, value):
    """Function to add a key-value pair to a hydra config object, do not use for replacement of existing keys (unless not part of a struct)"""
    OmegaConf.set_struct(cfg, False)
    # keys = key.split(".")  # Split the key by dot to handle nested keys

    # if len(keys) > 1:
    #     # If the key is nested, create the nested structure
    #     for k in keys[:-1]:
    #         if k not in cfg:
    #             cfg[k] = OmegaConf.create({})
    #         cfg = cfg[k]
    #     key = keys[-1]

    cfg.update(OmegaConf.create({key: value}))
    OmegaConf.set_struct(cfg, True)
    return cfg


def setup_optimizer(cfg):
    """Setup the optimizer for the model"""
    if cfg.optimizer.algorithm == "adam":
        if "lr_schedule" not in cfg.optimizer:
            assert "learning_rate" in cfg.optimizer, "learning_rate is not defined"
            lr_schedule = optax.constant_schedule(cfg.optimizer.learning_rate)
            logging.info(
                f"No learning rate schedule is defined, using default constant learning rate"
            )

        elif cfg.optimizer.lr_schedule.type == "constant":
            assert (
                "learning_rate" in cfg.optimizer.lr_schedule
            ), "learning_rate is not defined"
            lr_schedule = optax.constant_schedule(
                cfg.optimizer.lr_schedule.learning_rate
            )

        elif cfg.optimizer.lr_schedule.type == "piecewise_constant":
            assert (
                "init_learning_rate" in cfg.optimizer.lr_schedule
            ), "init_learning_rate is not defined"

            bounds_and_scales = {
                bound: value
                for bound, value in zip(
                    cfg.optimizer.lr_schedule.boundaries,
                    cfg.optimizer.lr_schedule.scales,
                )
            }

            lr_schedule = optax.piecewise_constant_schedule(
                init_value=cfg.optimizer.lr_schedule.init_learning_rate,
                boundaries_and_scales=bounds_and_scales,
            )

        else:
            raise NotImplementedError(
                "The schedule is not implemented, chose from 'constant', 'piecewise_constant'"
            )
        print(lr_schedule)
        optimizer = optax.adam(lr_schedule)

    else:
        raise NotImplementedError("The optimizer is not implemented, chose from 'adam'")
    return optimizer


if __name__ == "__main__":
    from jax import numpy as jnp
    from matplotlib import pyplot as plt

    # Initial learning rate
    init_lr = jnp.float32(0.1)

    # Define the schedule: divide by 10 at step 100
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=init_lr,
        boundaries_and_scales={
            100: jnp.float32(0.1),
            140: jnp.float32(0.1),
        },  # At step 100, multiply by 0.1
    )

    steps = jnp.arange(0, 200, 10)
    lrs = [lr_schedule(step) for step in steps]

    plt.figure()
    plt.semilogy(steps, lrs)
    plt.xlabel("Steps")
    plt.ylabel("Learning rate")
    plt.title("Piecewise constant schedule")
    plt.show()


def convert_to_wandb_format(nested_dict, parent_key="", separator="/"):
    """
    Convert nested dictionary to W&B compatible flat format with slash-separated keys
    Args:
        nested_dict: Input dictionary (possibly nested)
        parent_key: Internal use for recursion
        separator: Character to use for path separation (default '/')
    Returns:
        dict: Flat dictionary with slash-separated keys
    """
    items = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(convert_to_wandb_format(v, new_key, separator))
        else:
            items[new_key] = v
    return items


def get_model_save_paths(model_path):
    list_dir = os.listdir(model_path)
    final_model_path = None
    checkpoint_dir = None
    for i, dir_ in enumerate(list_dir):
        if "plots" in dir_:
            continue
        elif "final" in dir_:
            final_model_path = os.path.join(model_path, dir_)
        elif "checkpoints" in dir_:
            checkpoint_dir = os.path.join(model_path, dir_)

    if final_model_path is None and checkpoint_dir is None:
        raise ValueError("No final model or checkpoints found in model_save")

    list_checkpoints = os.listdir(checkpoint_dir)
    best_dir = -999
    for i, dir_ in enumerate(list_checkpoints):
        if int(dir_) > best_dir:
            best_dir = int(dir_)
            best_dir_str = dir_
    best_model_path = os.path.join(checkpoint_dir, best_dir_str)

    paths = [final_model_path, best_model_path]
    return paths


def get_model_paths(cfg):
    """Get the model paths from the config file"""
    model_path = cfg.model_save_path
    if model_path is None:
        raise ValueError("No model save path found in config file")

    if not os.path.exists(model_path):
        raise ValueError(f"Model save path {model_path} does not exist")

    # Get the model save paths
    paths = get_model_save_paths(model_path)

    return paths


# Convert ndarray to list if necessary
def convert_ndarray(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj
