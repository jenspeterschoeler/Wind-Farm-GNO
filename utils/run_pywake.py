import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from py_wake import HorizontalGrid
from py_wake.deficit_models import NiayifarGaussianDeficit, SelfSimilarityDeficit2020
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.turbulence_models import CrespoHernandez

# LinearSum
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph import torch_pyg_to_jraph
from utils.to_graph import append_globals_to_nodes, min_max_scale, to_graph

wt = DTU10MW()
D_10_MW = wt.diameter()


def simulate_farm(
    inflow_dict: Dict,
    positions: np.ndarray,
    grid: HorizontalGrid,
    convert_to_graph=False,
    to_graph_kws: Dict = None,
):
    """Function to simulate the power and loads of a wind farm given the inflow conditions and the
    wind turbine positions using PyWake. The function will return the simulated power and loads
    for each turbine.

    args:
    inflow_df: pd.DataFrame, the inflow conditions for the wind farm
    positions: np.ndarray, the wind turbine positions
    loads_method: str, the kind of load model to use: either OneWT or TwoWT
    """
    site = UniformSite()

    ws = inflow_dict["u"]
    wd = np.ones(len(ws)) * 270
    ti = inflow_dict["ti"]
    x = positions[:, 0]
    y = positions[:, 1]

    wt = DTU10MW()

    wf_model = All2AllIterative(
        site,
        wt,
        wake_deficitModel=NiayifarGaussianDeficit(),
        blockage_deficitModel=SelfSimilarityDeficit2020(),
        superpositionModel=LinearSum(),
        turbulenceModel=CrespoHernandez(),
    )

    farm_sims = []
    flow_maps = []
    for ws_, wd_, ti_ in zip(ws, wd, ti):
        farm_sim = wf_model(
            x,
            y,  # wind turbine positions
            wd=wd_,  # Wind direction
            ws=ws_,  # Wind speed
            TI=ti_,  # Turbulence intensity
        )

        flow_map = farm_sim.flow_map(grid=grid, wd=wd_, ws=ws_)
        farm_sims.append(farm_sim)
        flow_maps.append(flow_map)

    xx, yy = np.meshgrid(grid.x, grid.y)
    x_grid = xx.flatten()
    y_grid = yy.flatten()
    xy_grid = np.array([x_grid, y_grid]).T

    farm_sim["nwt_final"] = len(x)
    if convert_to_graph:
        graph_list = []
        for farm_sim, flowmap in zip(farm_sims, flow_maps):
            wt_ws = farm_sim.WS_eff.values.copy().squeeze()
            # TODO Add these as optionals e.g. extra stuff in the graph, consider backwards compatability?
            wt_ti = farm_sim.TI_eff.values.copy().squeeze()
            wt_CTs = farm_sim.CT.values.copy().squeeze()
            wts_info = np.array([wt_ws, wt_ti, wt_CTs]).T
            u_output = flowmap.WS_eff.values.copy().squeeze()
            original_trunk_shape = u_output.shape
            u_output = u_output.flatten()
            output_features = u_output.reshape(-1, 1)
            global_features = np.array([farm_sim["WS"], farm_sim["TI"]]).copy()
            node_features = wts_info  # wt_CTs.reshape(-1, 1)
            points = np.array([farm_sim["x"], farm_sim["y"]]).T.copy()
            graph = to_graph(
                points=points,
                node_features=node_features,
                global_features=global_features,
                trunk_inputs=xy_grid,
                output_features=output_features,
                **to_graph_kws,
            )
            graph_list.append(graph)
        return graph_list, original_trunk_shape
    else:
        return flow_maps, farm_sims


def construct_test_graph(positions, U, TI, grid):
    inflow_dict = {
        "u": U,
        "ti": TI,
    }
    # Simulate the farm
    to_graph_kws = {
        "connectivity": "delaunay",
        "add_edge": "cartesian",
        "rel_wd": None,
    }

    graph, _ = simulate_farm(
        inflow_dict=inflow_dict,
        positions=positions,
        grid=grid,
        convert_to_graph=True,
        to_graph_kws=to_graph_kws,
    )
    graph = graph[
        0
    ]  # [0] is because the function initally runs in parralel and therefore outputs to a list, no parralism here.

    # add batch dimension to trunk inputs/outputs with torch
    graph.output_features = graph.output_features.unsqueeze(0)
    graph.n_edge = graph.n_edge.unsqueeze(0)
    graph.trunk_inputs = graph.trunk_inputs.unsqueeze(0)
    graph.n_node = graph.n_node.unsqueeze(0)
    graph.global_features = graph.global_features.unsqueeze(0)
    return graph


def get_crosstream_probe_graph(
    scale_stats,
    dataset,
    data_loader,
    pos_behind_last_turbine=5,
    data_idx=0,
    D=D_10_MW,
):

    assert (
        scale_stats["scaling_type"] == "min_max"
    ), "Hardcoding: only min-max scaling is supported"
    assert (
        scale_stats["scaling_method"] == "run4"
    ), "Hardcoding: only run4 type is supported"

    org_graph = dataset[data_idx]

    U = org_graph.global_features[0:1] * scale_stats["velocity"]["range"][0]
    TI = org_graph.global_features[1:] * scale_stats["ti"]["range"][0]
    org_positions = org_graph.pos
    positions = org_positions * scale_stats["distance"]["range"]

    distance_behind_farm = pos_behind_last_turbine * D
    x = positions[:, 0].max() + distance_behind_farm

    y_minmax = np.abs(positions[:, 1]).max()
    y_min = -y_minmax - 5 * D
    y_max = y_minmax + 5 * D
    y_grid = np.linspace(y_min, y_max, 100)

    xx, yy = np.meshgrid(x, y_grid)
    probe_positions = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    node_positions = np.concatenate((positions, probe_positions), axis=0)

    grid = HorizontalGrid(
        x=[x],
        y=y_grid,
    )

    graph = construct_test_graph(org_positions, U, TI, grid)
    scaled_graph = min_max_scale(
        graph, scale_stats, scaling_method=scale_stats["scaling_method"]
    )
    scaled_graph = append_globals_to_nodes(scaled_graph)

    io_tuple_probe_graph = data_loader.torch_pyg_to_jraph(graph)
    jraph_graph, probe_graph, array_tuple = io_tuple_probe_graph
    probe_targets, wt_mask, probe_mask = array_tuple
    wt_mask = wt_mask.reshape(
        -1, 1
    )  # This empty dimension is normally added by the batcher
    probe_mask = probe_mask.reshape(-1, 1)

    return (
        y_grid,
        node_positions,
        jraph_graph,
        probe_graph,
        (probe_targets, wt_mask, probe_mask),
    )


def construct_on_the_fly_probe_graph(
    positions,
    U,
    TI,
    grid,
    scale_stats,
    return_positions: bool = False,
):
    inflow_dict = {
        "u": U,
        "ti": TI,
    }
    # Simulate the farm
    to_graph_kws = {
        "connectivity": "delaunay",
        "add_edge": "cartesian",
        "rel_wd": None,
    }

    pyg_graph, _ = simulate_farm(
        inflow_dict=inflow_dict,
        positions=positions,
        grid=grid,
        convert_to_graph=True,
        to_graph_kws=to_graph_kws,
    )
    pyg_graph = pyg_graph[
        0
    ]  # [0] is because the function initally runs in parralel and therefore outputs to a list, no parralism here.

    # add batch dimension to trunk inputs/outputs with torch
    pyg_graph.output_features = pyg_graph.output_features.unsqueeze(0)
    pyg_graph.n_edge = pyg_graph.n_edge.unsqueeze(0)
    pyg_graph.trunk_inputs = pyg_graph.trunk_inputs.unsqueeze(0)
    pyg_graph.n_node = pyg_graph.n_node.unsqueeze(0)
    pyg_graph.global_features = pyg_graph.global_features.unsqueeze(0)

    pyg_graph = min_max_scale(pyg_graph, scale_stats=scale_stats, scaling_method="run4")
    pyg_graph = append_globals_to_nodes(pyg_graph)

    jraph_graph, jraph_probe_graphs, node_array_tuple = torch_pyg_to_jraph(
        pyg_graph,
        graphs_only=False,
        probe_graphs=True,
        input_node_feature_idxs=[3, 4],
        target_node_feature_idxs=[0],
        add_pos_to_nodes=True,
        add_pos_to_edges=False,
        return_positions=return_positions,
        return_idxs=False,
    )

    return jraph_graph, jraph_probe_graphs, node_array_tuple


if __name__ == "__main__":
    import os
    import pickle

    from matplotlib import pyplot as plt

    from utils import JraphDataLoader, Torch_Geomtric_Dataset
    from utils.plotting import plot_crossstream_predictions, plot_probe_graph_fn
    from utils.to_graph import get_node_indexes
    from utils.weight_converter import load_portable_model

    #### Shared Parameters ####
    # test_data_path = "/home/jpsch/code/spo-operator-tests/data/medium_graphs_nodes/test_pre_processed"  # Local path

    test_data_path = "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2/test_pre_processed"  # HPC path

    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)

    wt = DTU10MW()
    D = wt.diameter()

    to_graph_kws = {
        "connectivity": "delaunay",
        "add_edge": "cartesian",
        "rel_wd": None,
    }

    ##### Cross-stream probe graph example using existing graph from dataset #####

    # main_path = "../2025-06-13/21-40-03"
    main_path = "../15-16-02/3"
    cfg_path = os.path.join(main_path, ".hydra/config.yaml")
    model_cfg_path = os.path.join(main_path, "model_config.json")
    model_path = os.path.join(main_path, "best_params.msgpack")

    restored_params, restored_cfg_model, model, _ = load_portable_model(
        model_path, model_cfg_path, dataset=dataset
    )

    input_node_idxs = get_node_indexes(restored_cfg_model.data.io.input_node_features)
    target_node_ixds = get_node_indexes(restored_cfg_model.data.io.target_node_features)

    data_loader = JraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        add_pos_to_nodes=True,
        probe_graphs=True,
        input_node_feature_idxs=input_node_idxs,
        target_node_feature_idxs=target_node_ixds,
    )

    # Get the probe graph
    (
        y_grid,
        node_positions,
        jraph_graph,
        probe_graph,
        (probe_targets, wt_mask, probe_mask),
    ) = get_crosstream_probe_graph(
        restored_cfg_model.data.scale_stats,
        dataset,
        data_loader,
        pos_behind_last_turbine=5,
        # data_idx=0, # single string
        # data_idx=11, # cluster
        data_idx=40,  # multi-string
    )

    probe_predictions = model.apply(
        restored_params,
        jraph_graph,
        probe_graph,
        wt_mask,
        probe_mask,
    )

    # create graphs for plotting
    save_graphs = False
    # pos_behind_list = [5, 20, 50]
    # graph_idxs = [0, 11, 40]  # single string, cluster, multi-string
    pos_behind_list = [1]  # Only testing with two positions behind
    graph_idxs = [0]  # single string, cluster
    graphs = []
    i = 0
    if save_graphs:
        graph_save_path = os.path.join(test_data_path, "plot_graphs")
        os.makedirs(graph_save_path, exist_ok=True)
    for pos_behind in pos_behind_list:
        for graph_idx in graph_idxs:
            (
                y_grid,
                node_positions,
                jraph_graph,
                probe_graph,
                (probe_targets, wt_mask, probe_mask),
            ) = get_crosstream_probe_graph(
                restored_cfg_model.data.scale_stats,
                dataset,
                data_loader,
                pos_behind_last_turbine=pos_behind,
                data_idx=graph_idx,
            )

            plot_graph_components = {
                "jraph_graph": jraph_graph,
                "probe_graph": probe_graph,
                "node_positions": node_positions,
                "probe_targets": probe_targets,
                "wt_mask": wt_mask,
                "probe_mask": probe_mask,
                "y_grid": np.asanyarray(y_grid),
            }
            probe_predictions = model.apply(
                restored_params,
                jraph_graph,
                probe_graph,
                wt_mask,
                probe_mask,
            )

            if save_graphs:
                # Save the plot graph components to a pickle file
                with open(
                    os.path.join(graph_save_path, f"plot_graph_components_{i}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(plot_graph_components, f)

            i += 1

    probe_targets = probe_targets[probe_mask.squeeze() > 0]
    probe_predictions = probe_predictions[probe_mask.squeeze() > 0]

    plot_crossstream_predictions(
        np.array(probe_predictions), np.array(probe_targets), np.array(y_grid)
    )
    plt.savefig(
        os.path.join(main_path, "crossstream_predictions.png"),
        bbox_inches="tight",
        dpi=300,
    )

    plot_probe_graph_fn(
        jraph_graph,
        probe_graph,
        node_positions,
        include_wt_nodes=True,
        include_wt_edges=True,
        include_probe_nodes=True,
        include_probe_edges=False,
    )
    plt.savefig(
        os.path.join(main_path, "probe_graph.png"),
        bbox_inches="tight",
        dpi=300,
    )
