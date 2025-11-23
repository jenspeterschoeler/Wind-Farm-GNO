import math
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.transforms import Cartesian, Delaunay, Distance, FaceToEdge, Polar


def get_node_indexes(vars: List[str]) -> List[int]:
    if vars == ["U"]:
        node_indexes = [3]
    elif vars == ["U", "TI"]:
        node_indexes = [3, 4]
    elif vars == ["U", "TI", "CT"]:
        node_indexes = [2, 3, 4]
    elif vars == ["u"]:
        #! Currently only possible as output because it is the only stored flowmap
        node_indexes = [0]
    else:
        raise ValueError(f"Node features {vars} not recognized.")
    return node_indexes


def append_globals_to_nodes(data):
    """
    Append global features to each node in the graph
    """
    assert data.n_node.shape[0] == 1
    n_nodes = int(data.n_node[0])
    # n_globals = data.n_edge[0]
    data.node_features = torch.cat(
        [data.node_features, data.global_features.repeat(n_nodes, 1)], dim=1
    )
    return data


def min_max_scale(data, scale_stats, scaling_method="run2"):
    """Min-max scale the dataset"""
    velocity_min = torch.tensor(scale_stats["velocity"]["min"])
    velocity_range = torch.tensor(scale_stats["velocity"]["range"])
    distance_min = torch.tensor(scale_stats["distance"]["min"])
    distance_range = torch.tensor(scale_stats["distance"]["range"])
    ti_min = torch.tensor(scale_stats["ti"]["min"])
    ti_range = torch.tensor(scale_stats["ti"]["range"])
    ct_min = torch.tensor(scale_stats["ct"]["min"])
    ct_range = torch.tensor(scale_stats["ct"]["range"])

    if scaling_method == "run3":
        # HACK! zeros to enable res-net if we want
        velocity_min = torch.tensor([0])  #
        ti_min = torch.tensor([0])
        ct_min = torch.tensor([0])
    elif scaling_method == "run4":
        # HACK! zeros to enable res-net if we want, also required for constructing the connections on the fly e.g. for probe setup
        distance_min = torch.tensor([0])
        velocity_min = torch.tensor([0])  #
        ti_min = torch.tensor([0])
        ct_min = torch.tensor([0])

    node_min = torch.cat([velocity_min, ti_min, ct_min], dim=0)
    node_range = torch.cat([velocity_range, ti_range, ct_range], dim=0)
    global_min = torch.cat([velocity_min, ti_min], dim=0)
    global_range = torch.cat([velocity_range, ti_range], dim=0)

    data.output_features = (data.output_features - velocity_min) / velocity_range
    data.node_features = (
        data.node_features - node_min
    ) / node_range  #! NOTE THIS MIGHT ALSO HAVE GLOBALS IF UNSCALING

    data.edge_attr = (data.edge_attr - distance_min) / distance_range
    data.global_features = (data.global_features - global_min) / global_range
    data.trunk_inputs = (data.trunk_inputs - distance_min) / distance_range
    data.pos = (data.pos - distance_min) / distance_range
    return data


class PyGTupleData(PyGData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "global_features":
            return None
        if key == "n_node":
            return None
        if key == "n_edge":
            return None
        if key == "trunk_inputs":
            return None
        if key == "output_features":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def to_graph(
    points: np.ndarray,
    connectivity: str = "delaunay",
    add_edge: str = "polar",
    node_features: np.ndarray = None,
    global_features: np.ndarray = None,
    rel_wd: float = 270,
    trunk_inputs=None,
    output_features=None,
) -> PyGTupleData:
    """
    Converts np.array to torch_geometric.data.data.Data object with the specified connectivity and edge feature type.
    """
    assert connectivity in [
        "delaunay",
    ]
    assert points.shape[1] == 2

    points_ = torch.Tensor(points)
    raw_graph_data = PyGTupleData(pos=points_)
    if connectivity.casefold() == "delaunay":
        delaunay_fn = Delaunay()
        edge_fn = FaceToEdge()
        graph = edge_fn(delaunay_fn(raw_graph_data))

    else:
        raise ValueError(
            "Please define the connectivity scheme (available types: : 'delaunay')"
        )

    if add_edge == "polar".casefold():
        polar_fn = Polar(norm=False)
        graph = polar_fn(graph)
        if rel_wd is not None:
            edge_rel_wd = math.radians(270) - graph.edge_attr[:, 1]
            graph.edge_attr = torch.cat(
                (graph.edge_attr, edge_rel_wd.unsqueeze(1)), dim=1
            )
    elif add_edge == "cartesian".casefold():
        cartesian_fn = Cartesian(norm=False)
        distance_fn = Distance(norm=False)
        graph = cartesian_fn(graph)
        graph = distance_fn(graph)

    else:
        raise ValueError(
            "Please select a coordinate system that is supported (available types: : 'polar')"
        )

    graph.n_node = torch.Tensor([node_features.shape[0]])
    graph.n_edge = torch.Tensor([graph.edge_attr.shape[0]])

    if node_features is not None:
        graph.node_features = torch.Tensor(node_features)

    if global_features is not None:
        graph.global_features = torch.Tensor(global_features)

    if trunk_inputs is not None:
        graph.trunk_inputs = torch.Tensor(trunk_inputs)

    if output_features is not None:
        graph.output_features = torch.Tensor(output_features)

    graph = PyGTupleData(
        **graph
    )  # Convert at the end in case altering __cat_dim__ breaks something
    return graph
