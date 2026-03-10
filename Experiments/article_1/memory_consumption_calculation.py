"""This script is intended to be run from the command line to measure memory consumption
from the basedir of the repository."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import psutil
from jax import numpy as jnp
from omegaconf import DictConfig

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from pympler import asizeof
from tqdm import tqdm

from utils.data_tools import retrieve_dataset_stats, setup_unscaler
from utils.plotting import matplotlib_set_rcparams
from utils.run_pywake import construct_on_the_fly_probe_graph, simulate_farm
from utils.torch_loader import JraphDataLoader, Torch_Geomtric_Dataset
from utils.weight_converter import load_portable_model

process = psutil.Process(os.getpid())

matplotlib_set_rcparams("paper")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--timing_experiment_id",
    type=int,
    default=1,
    help="ID of the timing experiment to run (for job arrays)",
)

experiment_id = parser.parse_args().timing_experiment_id
print(f"Running timing experiment ID: {experiment_id}")

if experiment_id == 1:
    n_probes_list = [1, 10]
    grid_configs = [[1, 1], [5, 2]]
elif experiment_id == 2:
    n_probes_list = [100]
    grid_configs = [[20, 5]]
elif experiment_id == 3:
    n_probes_list = [1000]
    grid_configs = [[100, 10]]
elif experiment_id == 4:
    n_probes_list = [2000]
    grid_configs = [[100, 20]]
elif experiment_id == 5:
    n_probes_list = [3000]
    grid_configs = [[200, 15]]
elif experiment_id == 6:
    n_probes_list = [4000]
    grid_configs = [[200, 20]]
elif experiment_id == 7:
    n_probes_list = [5000]
    grid_configs = [[300, 15]]
elif experiment_id == 8:
    n_probes_list = [6000]
    grid_configs = [[300, 20]]
elif experiment_id == 9:
    n_probes_list = [7000]
    grid_configs = [[350, 20]]
elif experiment_id == 10:
    n_probes_list = [8000]
    grid_configs = [[400, 20]]
elif experiment_id == 11:
    n_probes_list = [9000]
    grid_configs = [[450, 20]]
elif experiment_id == 12:
    n_probes_list = [10000]
    grid_configs = [[500, 20]]
else:
    raise ValueError(f"Invalid timing_experiment_id: {experiment_id}")

time.sleep(
    np.random.randint(10, 120)
)  # sleep random time to avoid all jobs starting at the same time for data loading


def expand_dims_node_array_tuple(node_array_tuple):
    new_tuple = ()
    for element in node_array_tuple:
        if len(element.shape) == 1:
            element = jnp.expand_dims(element, axis=-1)

        new_tuple += (element,)
    return new_tuple


def get_memory_mb():
    """Return memory usage in MB."""
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Resident Set Size (RAM usage)


# main_path = os.path.abspath(
#     "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
# )  # sophia
main_path = "./assets/best_model_Vj8"  # local

if "assets" in main_path:
    running_locally = True
else:
    running_locally = False

cfg_path = os.path.join(main_path, ".hydra/config.yaml")
model_cfg_path = os.path.abspath(os.path.join(main_path, "model_config.json"))
params_paths = os.path.join(main_path, "best_params.msgpack")
if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"
fig_folder_path = os.path.join(main_path, "model/figures_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)

### Load
with open(model_cfg_path, "r") as f:
    nested_dict = json.load(f)
    restored_cfg_model = DictConfig(nested_dict)


if running_locally:
    test_data_path = os.path.abspath(
        "./data/zenodo_graphs/test_pre_processed"
    )  # Download dataset and place in data directory: https://doi.org/10.5281/zenodo.17671257
else:
    test_data_path = restored_cfg_model.data.test_path  # Sophia server
    test_data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2/test_pre_processed"
    )
dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

input_node_features = [3, 4]
target_node_features = [0]


stats, scale_stats = retrieve_dataset_stats(dataset)
unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)


restored_params, restored_cfg_model, model, dropout_active = load_portable_model(
    params_paths, model_cfg_path, dataset
)


def model_prediction_fn(
    input_graphs,
    input_probe_graphs,
    input_wt_mask,
    input_probe_mask,
) -> jnp.ndarray:
    """This function assumes the graphs are padded"""

    prediction = model.apply(
        restored_params,
        input_graphs,
        input_probe_graphs,
        input_wt_mask,
        input_probe_mask,
    )

    return prediction


pred_fn = jax.jit(model_prediction_fn)


@jax.jit
def embed_fn(graphs, probe_graphs):
    pre_processed_graphs, pre_processed_probe_graphs = model.apply(
        restored_params, graphs, probe_graphs, train=False, method=model.embedder
    )
    return pre_processed_graphs, pre_processed_probe_graphs


@jax.jit
def wt_process_fn(
    pre_processed_graphs,
    pre_processed_probe_graphs,
    wt_mask,
    probe_mask,
):
    pre_processed_probe_graphs, latentspace_nodes = model.apply(
        restored_params,
        pre_processed_graphs,
        pre_processed_probe_graphs,
        wt_mask,
        probe_mask,
        train=False,
        method=model.wt_processor,
    )
    return pre_processed_probe_graphs, latentspace_nodes


@jax.jit
def probe_process_fn(pre_processed_probe_graphs):
    processed_probe_nodes = model.apply(
        restored_params,
        pre_processed_probe_graphs,
        train=False,
        method=model.probe_processor,
    )
    return processed_probe_nodes


@jax.jit
def decode_fn(
    latentspace_nodes,
    processed_probe_nodes,
    wt_mask,
    probe_mask,
    probe_graphs,
):
    prediction = model.apply(
        restored_params,
        latentspace_nodes,
        processed_probe_nodes,
        wt_mask,
        probe_mask,
        probe_graphs,
        method=model.decoder,
    )
    return prediction


def time_and_mem_function(fn_to_measure, size_of_input, *args):
    time.sleep(0.01)
    mem_start = get_memory_mb()
    start_time = time.time()
    _ = fn_to_measure(*args)
    end_time = time.time()
    mem_end = get_memory_mb()

    time_taken = end_time - start_time
    mem_taken = mem_end - mem_start + size_of_input
    return time_taken, mem_taken


def time_parts_of_model(graphs, probe_graphs, wt_mask, probe_mask, size_of_input):
    time_dict = {}
    mem_dict = {}

    time_taken, mem_taken = time_and_mem_function(
        embed_fn, size_of_input, graphs, probe_graphs
    )
    time_dict["embedding_time[s]"] = time_taken
    mem_dict["embedding_memory[MB]"] = mem_taken

    pre_processed_graphs, pre_processed_probe_graphs = embed_fn(graphs, probe_graphs)

    time_taken, mem_taken = time_and_mem_function(
        wt_process_fn,
        size_of_input,
        pre_processed_graphs,
        pre_processed_probe_graphs,
        wt_mask,
        probe_mask,
    )
    time_dict["wt_processing_time[s]"] = time_taken
    mem_dict["wt_processing_memory[MB]"] = mem_taken

    pre_processed_probe_graphs, latentspace_nodes = wt_process_fn(
        pre_processed_graphs,
        pre_processed_probe_graphs,
        wt_mask,
        probe_mask,
    )

    time_taken, mem_taken = time_and_mem_function(
        probe_process_fn,
        size_of_input,
        pre_processed_probe_graphs,
    )
    time_dict["probe_processing_time[s]"] = time_taken
    mem_dict["probe_processing_memory[MB]"] = mem_taken

    processed_probe_nodes = probe_process_fn(pre_processed_probe_graphs)

    time_taken, mem_taken = time_and_mem_function(
        decode_fn,
        size_of_input,
        latentspace_nodes,
        processed_probe_nodes,
        wt_mask,
        probe_mask,
        probe_graphs,
    )
    time_dict["decoding_time[s]"] = time_taken
    mem_dict["decoding_memory[MB]"] = mem_taken

    _ = decode_fn(
        latentspace_nodes,
        processed_probe_nodes,
        wt_mask,
        probe_mask,
        probe_graphs,
    )

    time_dict["total_model_time[s]"] = sum(time_dict.values())
    mem_dict["max_model_memory[MB]"] = max(mem_dict.values())

    return time_dict, mem_dict


df_idx_list = []
jax_cache_clean_counter = 0
D = 178.3  # rotor diameter in meters

for idxs_per_sample, grid_config in zip(n_probes_list, grid_configs):
    df_idx = pd.DataFrame()

    data_loader = JraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        idxs_per_sample=idxs_per_sample,
        input_node_feature_idxs=input_node_features,
        target_node_feature_idxs=target_node_features,
        add_pos_to_nodes=True,
        add_pos_to_edges=False,
        sample_stepsize=1,  # uses every sample (meant for validation to be done with less than whole dataframe)
        probe_graphs=True,
        return_positions=True,
        return_layout_info=True,
    )

    for i, ((graphs, probe_graphs, node_array_tuple), layout_info, wt_sep) in tqdm(
        enumerate(data_loader), desc=f"Timing model for {idxs_per_sample} probes"
    ):

        # (graphs, probe_graphs, node_array_tuple), layout_info, wt_sep = next(
        #     iter(data_loader)
        # )
        node_array_tuple = expand_dims_node_array_tuple(node_array_tuple)
        targets, wt_mask, probe_mask, positions = node_array_tuple

        inflow_dict = {
            "u": graphs.globals[0, 0:1],
            "ti": graphs.globals[1, 1:2],
        }

        wt_indexes = np.where(wt_mask == 1)[0]

        wt_positions = positions[wt_indexes] * scale_stats["distance"]["range"]

        xmin = wt_positions[:, 0].min() * scale_stats["distance"]["range"] / D
        xmax = wt_positions[:, 0].max() * scale_stats["distance"]["range"] / D
        x_range = [
            xmin - 10,
            xmax + 100,
        ]  # ranges are created to match original dataset

        x = np.linspace(x_range[0], x_range[1], grid_config[0]) * D

        ymin = wt_positions[:, 1].min() * scale_stats["distance"]["range"] / D
        ymax = wt_positions[:, 1].max() * scale_stats["distance"]["range"] / D
        y_range = [ymin - 5, ymax + 5]
        y = np.linspace(y_range[0], y_range[1], grid_config[1]) * D

        mem_start = get_memory_mb()
        start_time = time.time()
        grid = HorizontalGrid(x=x, y=y)
        simulate_farm(
            inflow_dict=inflow_dict,
            grid=grid,
            positions=wt_positions,
            convert_to_graph=False,
        )
        end_time = time.time()
        mem_end = get_memory_mb()

        time_pywake = end_time - start_time
        mem_pywake = mem_end - mem_start

        memory_size_of_inputs = asizeof.asizeof(
            graphs, probe_graphs, wt_mask, probe_mask
        ) / (1024**2)

        mem_start = get_memory_mb()
        start_time = time.time()
        prediction = model.apply(
            restored_params, graphs, probe_graphs, wt_mask, probe_mask
        )
        end_time = time.time()
        mem_end = get_memory_mb()

        time_no_jit = end_time - start_time
        mem_no_jit = mem_end - mem_start + memory_size_of_inputs

        time.sleep(0.1)

        mem_start = get_memory_mb()
        start_time = time.time()
        pre_pred = pred_fn(graphs, probe_graphs, wt_mask, probe_mask)
        end_time = time.time()
        mem_end = get_memory_mb()

        time_jitting = end_time - start_time
        mem_jitting = mem_end - mem_start + memory_size_of_inputs

        time.sleep(0.1)
        mem_start = get_memory_mb()
        start_time = time.time()
        pred = pred_fn(graphs, probe_graphs, wt_mask, probe_mask)
        end_time = time.time()
        mem_end = get_memory_mb()

        time_jitted = end_time - start_time
        mem_jitted = mem_end - mem_start + memory_size_of_inputs

        time_dict_jitting, mem_dict_jitting = time_parts_of_model(
            graphs, probe_graphs, wt_mask, probe_mask, memory_size_of_inputs
        )

        time_dict_jitted, mem_dict_jitted = time_parts_of_model(
            graphs, probe_graphs, wt_mask, probe_mask, memory_size_of_inputs
        )

        # rename keys to indicate jitting status
        time_dict_jitting = {
            key + "_jitting": value for key, value in time_dict_jitting.items()
        }
        mem_dict_jitting = {
            key + "_jitting": value for key, value in mem_dict_jitting.items()
        }
        time_dict_jitted = {
            key + "_jitted": value for key, value in time_dict_jitted.items()
        }
        mem_dict_jitted = {
            key + "_jitted": value for key, value in mem_dict_jitted.items()
        }

        size_of_graph = graphs.n_node * graphs.nodes.shape[-1] + (
            graphs.n_edge + probe_graphs.n_edge
        ) * (1 + graphs.edges.shape[-1])

        unscaled_graph = unscaler.inverse_scale_graph(graphs)

        U_freestream = unscaled_graph.globals[0, 0]
        TI_ambient = unscaled_graph.globals[0, 1]
        n_wt = int(jnp.sum(wt_mask))
        local_row = {
            "size_of_graph": int(size_of_graph[0]),
            "U_freestream": float(U_freestream),
            "TI_ambient": float(TI_ambient),
            "n_edges": int(graphs.n_edge[0]),
            "n_probe_edges": int(probe_graphs.n_edge[0]),
            "n_wt": int(n_wt),
            "n_probes": int(idxs_per_sample),
            "layout_type": str(layout_info[0]),
            "wt_spacing": wt_sep[0].numpy(),
            "total_pred_time_no_jit[s]": float(time_no_jit),
            "total_pred_time_jitting[s]": float(time_jitting),
            "total_pred_time_jitted[s]": float(time_jitted),
            "total_pred_memory_no_jit[MB]": float(mem_no_jit),
            "total_pred_memory_jitting[MB]": float(mem_jitting),
            "total_pred_memory_jitted[MB]": float(mem_jitted),
            "pywake_simulation_time[s]": float(time_pywake),
            "pywake_simulation_memory[MB]": float(mem_pywake),
            **time_dict_jitting,
            **mem_dict_jitting,
            **time_dict_jitted,
            **mem_dict_jitted,
        }
        df_idx = pd.concat(
            [df_idx, pd.DataFrame(local_row, index=[0])], ignore_index=True
        )
        jax_cache_clean_counter += 1
        if jax_cache_clean_counter == 10:
            jax.clear_caches()  # <--- clears old compiled executables
            jax_cache_clean_counter = 0
    df_idx_list.append(df_idx)

df = pd.concat(df_idx_list, ignore_index=True)
# save dataframe
df_path = os.path.join(fig_folder_path, f"memory_and_timing_{experiment_id}.csv")
df.to_csv(df_path, index=False)
