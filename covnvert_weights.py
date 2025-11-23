"""This script converts and verifies model weights saved in a specific format.
It enables a format that can bi ported for use with different resources e.g. CPU and GPU. This is necessary for models trained on the cluster.
It is only an example script and may need to be adapted for different models and use cases.
"""

import json
import os

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from utils.data_tools import setup_refresh_iterator, setup_train_dataset
from utils.torch_loader import Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model, save_portable_model

run_locally = True
cluster_model_name = (
    "SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1/"
)
local_model_name = "best_model_Vj8"

# BASE_LOCAL_WORK_ROOT = "/home/jpsch/Documents/Sophia_work"
BASE_LOCAL_WORK_ROOT = "./assets"
BASE_REMOTE_WORK_ROOT = "/work/users/jpsch"
DATA_LOCAL_ROOT = "/home/jpsch/code/spo-operator-tests/data"
TRAIN_REL_PATH = "medium_graphs_nodes/train_pre_processed"


LOCAL_MODEL_ROOT = os.path.join(BASE_LOCAL_WORK_ROOT, local_model_name)
REMOTE_MODEL_ROOT = os.path.join(BASE_REMOTE_WORK_ROOT, cluster_model_name)

LOCAL_CONFIG_PATH = os.path.relpath(LOCAL_MODEL_ROOT + "/.hydra")
REMOTE_CONFIG_PATH = os.path.relpath(REMOTE_MODEL_ROOT + "/.hydra")

LOCAL_PARAMS_PATHS = [LOCAL_MODEL_ROOT + "/best_params.msgpack"]
LOCAL_MODEL_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(LOCAL_CONFIG_PATH)), "model_config.json"
)

TRAIN_DATA_PATH_LOCAL = os.path.join(DATA_LOCAL_ROOT, TRAIN_REL_PATH)

# %% Save model if it saved on the cluster
if run_locally:  # Run the protable model locally on CPU
    config_path = LOCAL_CONFIG_PATH
    model_cfg_path = LOCAL_MODEL_CFG_PATH
    train_path = TRAIN_DATA_PATH_LOCAL
    dataset = Torch_Geomtric_Dataset(train_path)
    params_paths = LOCAL_PARAMS_PATHS
else:  # Save GPU model on the cluster and save as portable model
    config_path = REMOTE_CONFIG_PATH
    output_dir = os.path.dirname(os.path.abspath(config_path))
    params_paths, model_cfg_path, org_params = save_portable_model(output_dir)


# Load model config
with open(model_cfg_path, "r") as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)

if not run_locally:
    dataset, _ = setup_train_dataset(restored_cfg_model, in_mem=False)

get_refreshed_train_fn, _ = setup_refresh_iterator(restored_cfg_model, dataset)
iterator = get_refreshed_train_fn()
graphs, probe_graphs, node_array_tuple = next(iterator)
targets, wt_mask, probe_mask = node_array_tuple

model_idx = 0
restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
    params_paths[model_idx], model_cfg_path, dataset
)


# %% Test predictions with restored model
prediction_restored_params = model.apply(
    restored_params,
    graphs,
    probe_graphs,
    wt_mask,
    probe_mask,
)

if not run_locally:
    prediction_org_params = model.apply(
        org_params[model_idx],
        graphs,
        probe_graphs,
        wt_mask,
        probe_mask,
    )
    print("Original params:", prediction_org_params)
    print("Restored params:", prediction_restored_params)
    print(type(prediction_org_params), prediction_org_params.shape)
    diff = jnp.abs(prediction_org_params - prediction_restored_params)
    print("Difference (max):", diff.max())
    print("Difference (mean):", diff.mean())
    print("Difference (std):", diff.std())
    print("Difference (min):", diff.min())
    plt.figure()
    plt.plot(np.array(prediction_org_params))
    plt.plot(np.array(prediction_restored_params), alpha=0.5)
    plt.show()
