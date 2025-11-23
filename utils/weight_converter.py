"""Standalone tool to convert weights to a portable format"""

import json
import os
from pathlib import Path

import flax.serialization
import jax
from omegaconf import DictConfig, OmegaConf

from .data_tools import setup_refresh_iterator, setup_train_dataset
from .GNO_probe import initialize_GNO_probe
from .misc import get_model_save_paths
from .model_tools import load_model, setup_model


def save_portable_model(model_parent_dir):
    """
    Save only the weights of the model to a file.
    """

    model_path = os.path.join(model_parent_dir, "model")
    paths = get_model_save_paths(model_path)
    params_paths = []
    org_params = []

    for sub_model_path, model_type_str in zip(paths, ["final", "best"]):
        # Load model
        if sub_model_path is None:
            continue

        parent_path = os.path.dirname(sub_model_path)
        if model_type_str == "best":
            parent_path = os.path.dirname(parent_path)

        model, params, metrics, cfg_model = load_model(sub_model_path)
        org_params.append(params)

        # Save
        params_path = os.path.join(model_parent_dir, f"{model_type_str}_params.msgpack")
        bytes_data = flax.serialization.to_bytes(params)
        Path(params_path).write_bytes(bytes_data)
        print(f"Model parameters saved to {params_path}")
        params_paths.append(params_path)

    model_cfg_path = os.path.join(model_parent_dir, "model_config.json")
    with open(model_cfg_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg_model), f, indent=4)
    print(f"Model config saved to {model_cfg_path}")

    return params_paths, model_cfg_path, org_params


def load_portable_model(params_path, model_cfg_path, dataset=None, inputs=None):
    """
    Load the model from a portable file.
    """
    with open(model_cfg_path, "r") as f:
        nested_dict = json.load(f)
        restored_cfg_model = DictConfig(nested_dict)
    model = setup_model(restored_cfg_model)

    if dataset is not None:
        assert inputs is None, "Cannot provide both dataset and inputs."

        if dataset is None and inputs is None:
            dataset, _ = setup_train_dataset(restored_cfg_model, in_mem=False)
            # dataset = dataset[0:1] # doesn't seem to work

        get_refreshed_train_fn, unpadded_iterator = setup_refresh_iterator(
            restored_cfg_model, dataset
        )
        train_iterator = get_refreshed_train_fn()
        graphs, probe_graphs, node_array_tuple = next(train_iterator)
        targets, wt_mask, probe_mask = node_array_tuple
    elif inputs is not None:
        assert dataset is None, "Cannot provide both dataset and inputs."
        graphs, probe_graphs, node_array_tuple = inputs
        targets, wt_mask, probe_mask = node_array_tuple

    # Read the saved weights
    bytes_data = Path(params_path).read_bytes()

    rng_key = jax.random.PRNGKey(0)
    placeholder_params, dropout_active = initialize_GNO_probe(
        restored_cfg_model,
        model,
        rng_key,
        graphs,
        probe_graphs,
        wt_mask,
        probe_mask,
    )
    restored_params = flax.serialization.from_bytes(placeholder_params, bytes_data)

    model = setup_model(restored_cfg_model)
    return restored_params, restored_cfg_model, model, dropout_active
