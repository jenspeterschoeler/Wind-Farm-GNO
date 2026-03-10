"""Graph construction and manipulation utilities."""

import logging

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import torch
from jraph._src import utils as jraph_utils
from omegaconf import DictConfig

# from utils.run_pywake import simulate_farm #!commented to avoid loading pywake, issue with monkeypatch of asarray
from utils.torch_loader import sum_by_parts_torch

logger = logging.getLogger(__name__)


def print_shapes(graph: jraph.GraphsTuple) -> None:
    graph_shape_msg = f"Shape of graph: {jax.tree.map(lambda x: jnp.shape(x), graph)}"
    print(graph_shape_msg)


def replace_nodes_with_globals(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """
    Replace node features with global features in a jraph GraphsTuple.

    Args:
    - graph: A jraph.GraphsTuple representing the batched graph.

    Returns:
    - A modified jraph.GraphsTuple where each node's features are replaced
    by the global features for its respective graph.
    """
    # Repeat the global features for each node in each graph
    repeated_globals = jnp.repeat(graph.globals, graph.n_node, axis=0)  # type: ignore[arg-type]

    # Replace the nodes with the repeated global features
    updated_graph = graph._replace(nodes=repeated_globals)

    return updated_graph


def get_padded_sizes(
    cfg: DictConfig,
) -> tuple[int, int, int]:
    assert cfg.optimizer.batching.type == "dynamic_graph_batching"
    max_nodes = cfg.data.stats.graph_size["max_n_nodes"]
    max_edges = cfg.data.stats.graph_size["max_n_edges"]
    batch_cfg = cfg.optimizer.batching
    padded_max_graphs = batch_cfg.max_graph_size
    max_node_ratio = batch_cfg.max_node_ratio
    max_edge_ratio = batch_cfg.max_edge_ratio

    if padded_max_graphs < 2:
        # For the padding function to work the max graphs has to be more than 1
        padded_max_graphs = 2

    padded_max_nodes = int(jnp.ceil(max_nodes * max_node_ratio * padded_max_graphs))
    padded_max_edges = int(jnp.ceil(max_edges * max_edge_ratio * padded_max_graphs))

    return padded_max_nodes, padded_max_edges, padded_max_graphs


def get_dynamic_batchning_max_sizes(iterator):
    """Estimate the maximum sizes of graph components in the dataset for dynamic batching."""
    max_nodes = 0
    max_edges = 0
    for graph in iterator:
        max_nodes = max(max_nodes, graph.n_node)
        max_edges = max(max_edges, graph.n_edge)
    return max_nodes, max_edges


def get_test_graph(type: str = "double_graph"):
    """A simple function to get some test graphs for testing purposes.
    Implemented as a function to avoid code duplication and clutter in scripts."""
    if type == "single_graph":
        test_graph = jraph.GraphsTuple(
            n_node=jnp.array([4]),
            n_edge=jnp.array([5]),
            nodes=jnp.array([[1.0, 2.0], [1.0, 0], [0, 2.0], [0, 0]]),
            edges=jnp.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
            globals=jnp.array([[1.0]]),
            senders=jnp.array([0, 1, 2, 1, 3]),
            receivers=jnp.array([2, 2, 3, 0, 1]),
        )

    elif type == "empty_graph":
        test_graph = jraph.GraphsTuple(
            nodes=jnp.zeros((10, 64)),
            edges=jnp.zeros((10, 64)),
            globals=jnp.zeros((1, 64)),
            senders=jnp.zeros((10,), dtype=jnp.int32),
            receivers=jnp.zeros((10,), dtype=jnp.int32),
            n_node=jnp.array([10]),
            n_edge=jnp.array([10]),
        )

    elif type == "graph_padded_graphs":
        test_graph = get_test_graph(type="single_graph")
        padded_test_graph = jraph_utils.pad_with_graphs(
            test_graph,
            n_graph=2,
            n_node=(test_graph.n_node * 1.75).astype(jnp.int32)[0],
            n_edge=(test_graph.n_edge * 2)[0],
        )
        test_graph = padded_test_graph

    elif type == "double_graph":
        # Multifeatured graph with 2 graphs
        test_graph = jraph.GraphsTuple(
            n_node=jnp.array([4, 4]),
            n_edge=jnp.array([5, 5]),
            nodes=jnp.array(
                [
                    [1.0, 2.0],
                    [1.0, 0],
                    [0, 2.0],
                    [0, 0],
                    [1.0, 2.0],
                    [1.0, 0],
                    [0, 2.0],
                    [0, 0],
                ]
            ),
            edges=jnp.array(
                [  # edges for the first graph
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    # edges for the second graph
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
            globals=jnp.array([[0.0, 1.0], [20.0, 10.0]]),
            senders=jnp.array([0, 1, 2, 1, 3, 4, 5, 6, 5, 7]),
            receivers=jnp.array([2, 2, 3, 0, 3, 5, 5, 6, 4, 5]),
        )

    elif type == "double_clone_graph":
        # Multifeatured graph with 2 graphs
        test_graph = jraph.GraphsTuple(
            n_node=jnp.array([4, 4]),
            n_edge=jnp.array([5, 5]),
            nodes=jnp.array(
                [
                    [1.0, 2.0],
                    [1.0, 0],
                    [0, 2.0],
                    [0, 0],
                    [1.0, 2.0],
                    [1.0, 0],
                    [0, 2.0],
                    [0, 0],
                ]
            ),
            edges=jnp.array(
                [  # edges for the first graph
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    # edges for the second graph
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ),
            globals=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            senders=jnp.array([0, 1, 2, 1, 3, 4, 5, 6, 5, 7]),
            receivers=jnp.array([2, 2, 3, 0, 3, 6, 6, 7, 4, 7]),
        )

    elif type == "double_padded_graph":
        test_graph = get_test_graph(type="double_graph")
        padded_test_graph = jraph_utils.pad_with_graphs(
            test_graph,
            n_graph=3,
            n_node=int((jnp.sum(test_graph.n_node) * 1.25).astype(jnp.int32).item()),
            n_edge=int((jnp.sum(test_graph.n_edge) * 1.5).astype(jnp.int32).item()),
        )
        test_graph = padded_test_graph

    else:
        raise ValueError(
            f"Unknown type of test graph: {type}, available types are: single_graph, empty_graph, double_graph"
        )
    return test_graph


def get_test_graph_operator_pair(type: str = "empty"):
    if type == "empty":
        test_graph = get_test_graph(type="empty_graph")
        test_trunk_input = jnp.zeros((1, 64))

    elif type == "padded":
        test_graph = get_test_graph(type="graph_padded_graphs")
        assert test_graph.n_node.shape[0] == 2
        test_trunk_input = jnp.ones((1, 3))
        test_trunk_input = jnp.concatenate([test_trunk_input, jnp.zeros((1, 3))], axis=0)

    elif type == "double":
        test_graph = get_test_graph(type="double_graph")
        test_trunk_input = jax.random.randint(
            jax.random.key(123),
            shape=(test_graph.n_node.shape[0], 3),
            minval=0,
            maxval=10,
        )

    elif type == "double_padded":
        test_graph = get_test_graph(type="double_padded_graph")
        assert test_graph.n_node.shape[0] == 3
        test_trunk_input = jnp.ones((2, 3))
        test_trunk_input = jnp.concatenate([test_trunk_input, jnp.zeros((1, 3))], axis=0)

    elif type == "double_padded_clone":
        test_graph = get_test_graph(type="double_clone_graph")
        padded_test_graph = jraph_utils.pad_with_graphs(
            test_graph,
            n_graph=3,
            n_node=int((jnp.sum(test_graph.n_node) * 1.25).astype(jnp.int32).item()),
            n_edge=int((jnp.sum(test_graph.n_edge) * 1.5).astype(jnp.int32).item()),
        )
        test_graph = padded_test_graph
        assert test_graph.n_node.shape[0] == 3
        test_trunk_input = jnp.ones((2, 3))
        test_trunk_input = jnp.concatenate([test_trunk_input, jnp.zeros((1, 3))], axis=0)

    else:
        raise ValueError(
            f"Unknown type of test graph operator pair: {type}, available types are: empty"
        )
    return test_graph, test_trunk_input


def min_max_scale(data, scale_stats, scaling_method="run4"):
    """Min-max scale the dataset"""
    logger.info(
        "This function uses some hardcoded values for the global values i.e location of u and ti it is assumed u is the first global feature and ti is the second also node features only containt CT, and positions are cartesian"
    )
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


def construct_probe_graphs(
    graphs,
    probe_positions,
    probe_targets,
    input_node_feature_idxs,
    target_node_feature_idxs,
    return_positions: bool = False,
):
    """A function to construct a probe graph from the 'trunk data'."""
    n_nodes = graphs.n_node
    n_probes = probe_positions.shape[1]
    node_pos = graphs.nodes[:, -2:]  # Assumes pos has been appended at the end

    ### Senders
    # Below was designed for cases with torch_batch > 1, but that is currently not used
    node_count = np.concatenate(
        [
            np.array(
                [
                    0,
                ]
            ),
            np.cumsum(n_nodes)[:-1],
        ]
    )
    senders = np.concat(
        [np.tile(np.arange(n) + n_old, n_probes) for n, n_old in zip(n_nodes, node_count)]
    )

    ### Receivers
    # Local probes for edges (receivers)
    probes_local_offsets = np.arange(len(n_nodes)) * n_probes
    probe_index_locals = np.concatenate(
        [
            np.array(np.repeat(np.arange(n_probes) + probe_offset, n_node))
            for n_node, probe_offset in zip(n_nodes, probes_local_offsets)
        ]
    )

    # Global probes for graph (receivers)
    probe_receivers = probe_index_locals + np.sum(n_nodes)

    ### Edges
    probe_positions_flat = probe_positions.reshape(-1, 2)
    wt_to_probe_edges = node_pos[senders, :] - probe_positions_flat[probe_index_locals, :]
    # append distances to the probe positions
    distances = np.sqrt(np.sum(wt_to_probe_edges**2, axis=1)).reshape(-1, 1)
    wt_to_probe_edges = np.concatenate([wt_to_probe_edges, distances], axis=1)

    n_edges = graphs.n_node * n_probes

    ## Nodes
    assert len(input_node_feature_idxs) == graphs.globals.shape[-1], (
        "The input_node_feature_idxs must match the number of node features in the globals."
    )  # TODO Improve this to mean they are the same dimension exactly not just shape
    probe_node_features = np.concatenate(
        [
            graphs.nodes[
                ..., input_node_feature_idxs
            ],  # These are replaced inside model, but it is helpful for debugging that they have values
            np.repeat(graphs.globals, n_probes, axis=0),
            # np.ones((n_probes * n_graphs, len(input_node_feature_idxs))),
        ]
    )

    # Node targets
    node_targets = np.concatenate(
        [
            graphs.nodes[:, target_node_feature_idxs],
            probe_targets.reshape(-1, 1),
        ]
    )

    n_node_total = graphs.n_node + n_probes

    nodes = probe_node_features  # We now use the same intially

    graphs = graphs._replace(nodes=nodes, n_node=n_node_total)

    # Node masks
    node_mask = np.zeros(len(nodes))
    probe_node_mask = node_mask
    probe_node_idx = np.unique(probe_receivers)
    probe_node_mask[probe_node_idx] = 1

    wt_node_mask = np.ones(len(nodes))
    wt_node_mask[probe_node_idx] = 0

    probe_graphs = jraph.GraphsTuple(
        nodes=probe_node_features,
        edges=wt_to_probe_edges,
        senders=senders,  # type: ignore[arg-type]
        receivers=probe_receivers,  # type: ignore[arg-type]
        globals=graphs.globals,
        n_node=n_node_total,
        n_edge=n_edges,
    )
    array_tuple = (
        node_targets,
        wt_node_mask,
        probe_node_mask,
    )

    if return_positions:
        # Append the positions trunk arrays
        node_positions = np.concatenate(
            [
                node_pos,
                probe_positions_flat,
            ]
        )
        array_tuple += (node_positions,)

    return (
        graphs,
        probe_graphs,
        array_tuple,
    )


def torch_pyg_to_jraph(
    pyg_batch,
    graphs_only: bool = False,
    probe_graphs: bool = True,
    input_node_feature_idxs=None,
    target_node_feature_idxs=None,
    add_pos_to_nodes: bool = True,
    add_pos_to_edges: bool = True,
    return_idxs: bool = False,
    return_positions: bool = False,
):
    # Extract node features, edge indices, and edge features
    if target_node_feature_idxs is None:
        target_node_feature_idxs = [0]
    if input_node_feature_idxs is None:
        input_node_feature_idxs = [2, 3]
    node_features = pyg_batch.node_features
    edge_indices = pyg_batch.edge_index
    edge_features = pyg_batch.edge_attr if pyg_batch.edge_attr is not None else None

    if add_pos_to_nodes:
        feature_index = np.concatenate([input_node_feature_idxs, [-2, -1]])
        node_features = torch.cat([node_features, pyg_batch.pos], dim=1)
    else:
        feature_index = input_node_feature_idxs

    if add_pos_to_edges:
        # find the position of receivers, receivers because the edge_attr is the position of the sender from the reciever
        receiver_pos = pyg_batch.pos[pyg_batch.edge_index[0, :]]
        edge_features = torch.cat([edge_features, receiver_pos], dim=1)  # type: ignore[call-overload]

    # Extract graph-level features if available
    graph_features = pyg_batch.global_features if hasattr(pyg_batch, "global_features") else None

    # Compute number of nodes and edges per graph
    if pyg_batch.batch is not None:
        n_node = torch.bincount(pyg_batch.batch)
        bincount_edges = torch.bincount(pyg_batch.edge_index[0, :])
        n_edge = sum_by_parts_torch(bincount_edges, torch.bincount(pyg_batch.batch))
    else:
        n_node = torch.tensor([pyg_batch.num_nodes])
        n_edge = torch.tensor([pyg_batch.num_edges])

    # Convert to jax.numpy arrays
    node_features = jnp.array(node_features.numpy())
    edge_indices = jnp.array(edge_indices.numpy())
    edge_features = jnp.array(edge_features.numpy()) if edge_features is not None else None

    graph_features = jnp.array(graph_features.numpy()) if graph_features is not None else None
    n_node = jnp.array(n_node.numpy())
    n_edge = jnp.array(n_edge)

    # Pop the trunk_inputs and the output_features
    trunk_inputs = pyg_batch.trunk_inputs
    output_features = pyg_batch.output_features
    # take a sample from these

    # Create jraph.GraphsTuple

    if graphs_only:
        jraph_graph = jraph.GraphsTuple(
            nodes=node_features[..., feature_index],
            edges=edge_features,
            senders=edge_indices[0, :],
            receivers=edge_indices[1, :],
            globals=graph_features,
            n_node=n_node,
            n_edge=n_edge,
        )

        return jraph_graph
    else:
        trunk_inputs = jnp.array(trunk_inputs.numpy())
        output_features = jnp.array(output_features.numpy())
        if probe_graphs:
            jraph_graph = jraph.GraphsTuple(
                nodes=node_features,
                edges=edge_features,
                senders=edge_indices[0, :],
                receivers=edge_indices[1, :],
                globals=graph_features,
                n_node=n_node,
                n_edge=n_edge,
            )

            graphs, probe_graphs, array_tuple = construct_probe_graphs(
                jraph_graph,
                trunk_inputs,
                output_features,
                input_node_feature_idxs,
                target_node_feature_idxs,
                return_positions=return_positions,
            )
            if return_idxs:
                # Get idxs from pyg_batch if available, otherwise use all indices
                if hasattr(pyg_batch, "idxs"):
                    idxs = jnp.array(pyg_batch.idxs)
                else:
                    idxs = jnp.arange(trunk_inputs.shape[1])
                array_tuple += (idxs,)

            return (
                graphs,
                probe_graphs,
                array_tuple,
            )
        else:
            jraph_graph = jraph.GraphsTuple(
                nodes=node_features[..., feature_index],
                edges=edge_features,
                senders=edge_indices[0, :],
                receivers=edge_indices[1, :],
                globals=graph_features,
                n_node=n_node,
                n_edge=n_edge,
            )
    return (
        jraph_graph,
        trunk_inputs,
        output_features,
    )


if __name__ == "__main__":  # type: ignore
    # test pad single graph
    graph = jraph.GraphsTuple(
        n_node=np.array([2], dtype=np.int32),  # type: ignore[arg-type]
        n_edge=np.array([1], dtype=np.int32),  # type: ignore[arg-type]
        nodes=np.array([[1.0], [2.0]]),
        edges=np.array([[1.0]]),
        senders=np.array([0], dtype=np.int32),  # type: ignore[arg-type]
        receivers=np.array([1], dtype=np.int32),  # type: ignore[arg-type]
        globals=np.array([[1.0]]),
    )
