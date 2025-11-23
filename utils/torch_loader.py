import io
import os
import warnings
from typing import Any, Generator, Iterator, List, Tuple, Union
from zipfile import ZipFile

import jax
import jraph
import numpy as np
import torch
from jax import numpy as jnp
from jraph._src import graph as gn_graph
from jraph._src import utils as jraph_utils
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

warnings.simplefilter(action="ignore", category=FutureWarning)

NestedListStr = Union[str, List["NestedListStr"]]
import logging

logger = logging.getLogger(__name__)


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


class Torch_Geomtric_Dataset(Dataset):
    """Dataset class for the GraphFarmsOperator dataset.
    Loads data from disk into a format that can be used with the Graph Operator DataLoader.

    If in_mem is True, the data is pre-loaded into memory so that subsequent shuffling or iteration
    does not require re-reading the files from disk.
    """

    def __init__(self, root_path: str, in_mem: bool = True) -> None:
        super().__init__()
        self._root_path = root_path
        self.in_mem = in_mem

        # Build the zip matrix and corresponding indexes
        self.zip_matrix = self._create_zip_matrix()
        all_zip_paths_repeated, all_zip_item_names = self._create_indexes()
        self.zip_paths_repeated = all_zip_paths_repeated
        self.zip_contents = all_zip_item_names

        # If data should be kept in memory, load it once
        if self.in_mem:
            self._cache = [
                self._open_single_content_in_zip((zip_path, zip_item))
                for zip_path, zip_item in zip(
                    self.zip_paths_repeated, self.zip_contents
                )
            ]

    def _open_zip(self, zip_construct: List[str]) -> Tuple[Any, ...]:
        """Open a zip file and return a tuple of loaded items."""
        zip_path = zip_construct[0]
        zip_items = zip_construct[1:]
        content = ()
        with ZipFile(zip_path) as zf:
            for item in zip_items:
                with zf.open(item) as f:
                    stream = io.BytesIO(f.read())
                    data = torch.load(stream, weights_only=False)
                    content += (data,)
        return content

    def _open_single_content_in_zip(self, zip_construct: Tuple[str, str]) -> Any:
        """Open a single item from a zip file."""
        zip_path, zip_item = zip_construct
        with ZipFile(zip_path) as zf:
            with zf.open(zip_item) as f:
                stream = io.BytesIO(f.read())
                data = torch.load(stream, weights_only=False)
        return data

    def _create_zip_matrix(self) -> NestedListStr:
        """Create a matrix of zip file paths and their contained items."""
        # Get a list of all zip file paths in the root directory.
        zip_list = [
            os.path.join(path, name)
            for path, _, files in os.walk(self._root_path)
            for name in files
        ]
        zip_list = [zip_file for zip_file in zip_list if zip_file.endswith(".zip")]
        self.zip_list = zip_list

        zip_matrix = []
        # For each zip file, get the list of items inside it.
        for zip_path in zip_list:
            with ZipFile(zip_path, "r") as zip_ref:
                zip_items = zip_ref.namelist()
            # Store the zip path along with its items (unpack so each becomes a tuple element)
            zip_matrix.append((zip_path, *zip_items))
        return zip_matrix

    def _create_indexes(self) -> Tuple[List[str], List[str]]:
        """Create lists of zip paths repeated and corresponding zip item names."""
        all_zip_paths_repeated = []
        all_zip_item_names = []
        for zip_construct in self.zip_matrix:
            zip_path, *zip_items = zip_construct
            repeated_zip_path = [zip_path] * len(zip_items)
            all_zip_paths_repeated += repeated_zip_path
            all_zip_item_names += zip_items
        return all_zip_paths_repeated, all_zip_item_names

    def __len__(self) -> int:
        return len(self.zip_paths_repeated)

    def __getitem__(self, index: int) -> Any:
        if self.in_mem:
            # Return the preloaded item from cache.
            return self._cache[index]
        else:
            # Read from disk on demand.
            zip_path = self.zip_paths_repeated[index]
            zip_item = self.zip_contents[index]
            return self._open_single_content_in_zip((zip_path, zip_item))


def sum_by_parts(numbers, lengths):
    result = []
    start = 0
    for length in lengths:
        end = start + length
        result.append(sum(numbers[start:end]))
        start = end
    return result


def sum_by_parts_torch(numbers, lengths):
    result = []
    start = 0
    for length in lengths:
        end = start + length
        result.append(torch.sum(numbers[start:end]))
        start = end
    return result


def load_sample_probabilities(probabilities_path: str) -> torch.Tensor:
    """Load the sample probabilities from a .npz file."""

    with np.load(probabilities_path) as data:
        sample_weights = torch.tensor(data["probabilities"], dtype=torch.float32)
        org_shape = torch.tensor(data["org_shape"], dtype=torch.int64)
    return sample_weights, org_shape


class JraphDataLoader:
    def __init__(
        self,
        dataset,
        stats=None,
        batch_size=2,
        shuffle=True,
        idxs_per_sample=None,
        n_cuts=None,
        sample_probabilities_path=None,
        input_node_feature_idxs=None,
        target_node_feature_idxs=None,
        add_pos_to_nodes=False,
        add_pos_to_edges=False,
        sample_stepsize=1,  # uses every sample (meant for validation to be done with less than whole dataframe)
        graphs_only=False,
        probe_graphs=False,
        return_positions=False,
        trunk_sample_strategy=None,
        return_idxs=False,
        return_layout_info=False,
        **kwargs,
    ):
        self.pyg_loader = PyGDataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )
        self.stats = stats

        self.idxs_per_sample = idxs_per_sample
        self.sample_probabilities_path = sample_probabilities_path
        self.sample_stepsize = sample_stepsize

        if input_node_feature_idxs is not None:
            print(f"Using node feature indexes {input_node_feature_idxs}")
            self.input_node_feature_idxs = input_node_feature_idxs
        else:
            self.input_node_feature_idxs = np.arange(
                self.pyg_loader.dataset[0].node_features.shape[-1]
            )
        if target_node_feature_idxs is not None:
            print(f"Using node feature indexes {target_node_feature_idxs}")
            self.target_node_feature_idxs = target_node_feature_idxs
        else:
            self.target_node_feature_idxs = np.arange(
                self.pyg_loader.dataset[0].node_features.shape[-1]
            )

        self.add_pos_to_nodes = add_pos_to_nodes
        self.add_pos_to_edges = add_pos_to_edges
        self.graphs_only = graphs_only
        self.probe_graphs = probe_graphs
        self.return_positions = return_positions
        self.return_idxs = return_idxs
        self.trunk_sample_strategy = trunk_sample_strategy
        if self.idxs_per_sample is not None and self.trunk_sample_strategy is None:
            print(
                "Using random sampling, to revert to default behaviour when using idxs_per_sample, and not setting trunk_sample_strategy"
            )
            self.trunk_sample_strategy = "random_sample"
        self.n_cuts = n_cuts

        if return_positions:
            assert self.probe_graphs, "return_positions requires probe_graphs"

        if self.probe_graphs:
            assert (
                self.add_pos_to_nodes == 1
            ), "add_pos_to_nodes must be True for probe_graphs"
        if self.idxs_per_sample is not None:
            self.random_generator = torch.Generator()
            self.random_generator.manual_seed(123)

        if self.sample_probabilities_path is not None:
            assert (
                idxs_per_sample is not None
            ), "sample_weights_path requires idxs_per_sample"
            self.sample_probabilities, self.org_shape = load_sample_probabilities(
                self.sample_probabilities_path
            )

        if self.sample_stepsize > 1:
            assert (
                idxs_per_sample is None
            ), "idxs_per_sample and sample_stepsize cannot be used together"

        self.return_layout_info = return_layout_info

    def __iter__(self):
        for batch in self.pyg_loader:
            if self.return_layout_info:
                # pop the layout info from the batch
                layout_type = batch.layout_type
                wt_spacing = batch.wt_spacing

                yield (
                    self.torch_pyg_to_jraph(batch),
                    layout_type,
                    wt_spacing,
                )

            else:
                yield self.torch_pyg_to_jraph(batch)

    def construct_probe_graphs(
        self,
        graphs,
        probe_positions,
        probe_targets,
        input_node_feature_idxs,
        target_node_feature_idxs,
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
        senders = np.concatenate(
            [
                np.tile(np.arange(n) + n_old, n_probes)
                for n, n_old in zip(n_nodes, node_count)
            ]
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
        wt_to_probe_edges = (
            node_pos[senders, :] - probe_positions_flat[probe_index_locals, :]
        )
        # append distances to the probe positions
        distances = np.sqrt(np.sum(wt_to_probe_edges**2, axis=1)).reshape(-1, 1)
        wt_to_probe_edges = np.concatenate([wt_to_probe_edges, distances], axis=1)

        n_edges = graphs.n_node * n_probes

        ## Nodes
        assert (
            len(input_node_feature_idxs) == graphs.globals.shape[-1]
        ), "The input_node_feature_idxs must match the number of node features in the globals."  # TODO Improve this to mean they are the same dimension exactly not just shape
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
            senders=senders,
            receivers=probe_receivers,
            globals=graphs.globals,
            n_node=n_node_total,
            n_edge=n_edges,
        )
        array_tuple = (
            node_targets,
            wt_node_mask,
            probe_node_mask,
        )

        if self.return_positions:
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

    def random_idxs(self, array):
        if self.sample_probabilities_path is None:
            idxs = torch.randint(
                low=0,
                high=array.shape[1],  # shape[1] is the indexes, shape[0] is batch_size
                size=(self.idxs_per_sample,),
                generator=self.random_generator,
            )
        else:
            idxs = torch.multinomial(
                self.sample_probabilities,
                num_samples=self.idxs_per_sample,
                replacement=False,
                generator=self.random_generator,
            )
        return idxs

    def torch_pyg_to_jraph(self, pyg_batch):
        # Extract node features, edge indices, and edge features
        node_features = pyg_batch.node_features
        edge_indices = pyg_batch.edge_index
        edge_features = pyg_batch.edge_attr if pyg_batch.edge_attr is not None else None

        if self.add_pos_to_nodes:
            feature_index = np.concatenate([self.input_node_feature_idxs, [-2, -1]])
            node_features = torch.cat(
                [node_features, pyg_batch.pos], dim=1
            )  # TODO Currently there are different beahviours depending on model, make decsions when GNO is re-introduced
        else:
            feature_index = self.input_node_feature_idxs

        if self.add_pos_to_edges:
            # find the position of receivers, receivers because the edge_attr is the position of the sender from the reciever
            receiver_pos = pyg_batch.pos[pyg_batch.edge_index[0, :]]
            edge_features = torch.cat([edge_features, receiver_pos], dim=1)

        # Extract graph-level features if available
        graph_features = (
            pyg_batch.global_features if hasattr(pyg_batch, "global_features") else None
        )

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
        edge_features = (
            jnp.array(edge_features.numpy()) if edge_features is not None else None
        )

        graph_features = (
            jnp.array(graph_features.numpy()) if graph_features is not None else None
        )
        n_node = jnp.array(n_node.numpy())
        n_edge = jnp.array(n_edge)

        # Pop the trunk_inputs and the output_features
        trunk_inputs = pyg_batch.trunk_inputs
        output_features = pyg_batch.output_features
        # take a sample from these

        # Create jraph.GraphsTuple

        if self.graphs_only:

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
            if self.trunk_sample_strategy == "random_sample":
                if self.idxs_per_sample is not None:
                    idxs = self.random_idxs(trunk_inputs)

                else:
                    raise ValueError("idxs_per_sample must be set for random sampling")

            elif self.trunk_sample_strategy == None:
                idxs = np.arange(trunk_inputs.shape[1])
            elif self.trunk_sample_strategy == "take_all":
                idxs = slice(None)
            elif self.trunk_sample_strategy == "evenly_distributed":
                assert self.idxs_per_sample is not None
                idxs = np.linspace(
                    0, output_features.shape[1] - 1, num=self.idxs_per_sample, dtype=int
                )
            else:
                raise NotImplementedError(
                    "trunk_sample_strategy must be either random_sample, deterministic or None"
                )

            trunk_inputs = trunk_inputs[:, idxs, ...]
            output_features = output_features[:, idxs, ...]

            trunk_inputs = jnp.array(
                trunk_inputs[:, :: self.sample_stepsize, ...].numpy()
            )
            output_features = jnp.array(
                output_features[:, :: self.sample_stepsize, ...].numpy()
            )
            if self.probe_graphs:
                jraph_graph = jraph.GraphsTuple(
                    nodes=node_features,
                    edges=edge_features,
                    senders=edge_indices[0, :],
                    receivers=edge_indices[1, :],
                    globals=graph_features,
                    n_node=n_node,
                    n_edge=n_edge,
                )

                graphs, probe_graphs, array_tuple = self.construct_probe_graphs(
                    jraph_graph,
                    trunk_inputs,
                    output_features,
                    self.input_node_feature_idxs,
                    self.target_node_feature_idxs,
                )
                if self.return_idxs:
                    idxs = jnp.array(idxs)
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


def pad_array(array: np.ndarray, n_graph: int) -> np.ndarray:
    missing_graphs = n_graph - array.shape[0]
    if missing_graphs > 0:
        print(f"missing graphs: {missing_graphs}")
        padding = np.zeros((missing_graphs,) + array.shape[1:])
        array = np.concatenate([array, padding], axis=0)
    return array


def pad_trunk_and_output(
    trunk_input: np.ndarray, output_features: np.ndarray, n_graph: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the trunk_and_output array with zeros to match the number of graphs."""
    missing_graphs = n_graph - trunk_input.shape[0]
    if missing_graphs > 0:
        logger.debug(f"missing graphs: {missing_graphs}")
        padding_trunk = np.zeros((missing_graphs,) + trunk_input.shape[1:])
        padding_output = np.zeros((missing_graphs,) + output_features.shape[1:])
        logger.debug(
            f"padding shape, trunk: {padding_trunk.shape}\t, output: {padding_output.shape}"
        )
        trunk_input = np.concatenate([trunk_input, padding_trunk], axis=0)
        output_features = np.concatenate([output_features, padding_output], axis=0)
        logger.debug(
            f"trunk shape: {trunk_input.shape}\t, output: {output_features.shape}"
        )
    return trunk_input, output_features


def pad_probe_targets_all_nodes_version(probe_targets, n_node_probes):
    """Pad the probe_targets array with zeros to match the number of nodes in the probe graph."""
    missing_nodes = n_node_probes - probe_targets.shape[0]
    if missing_nodes > 0:
        logger.debug(f"missing nodes: {missing_nodes}")
        padding = np.zeros((missing_nodes,) + probe_targets.shape[1:])
        probe_targets = np.concatenate([probe_targets, padding], axis=0)
    return probe_targets


def atleast_2d_last(arr):
    arr = np.asarray(arr)  # Ensure it's a NumPy array
    if arr.ndim == 0:  # Scalar case
        return arr.reshape(1, 1)
    elif arr.ndim == 1:  # 1D case
        return arr[:, np.newaxis]  # Add new axis at the last (-1)
    elif arr.ndim >= 2:  # Already 2D or more
        return arr  # No changes
    return arr


def dynamically_batch_graph_probe_operator(
    graph_operator_tuple_iterator: Iterator[
        Tuple[gn_graph.GraphsTuple, gn_graph.GraphsTuple, jnp.ndarray]
    ],
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_probes: int,
    return_layout_info: bool = False,
) -> Generator[
    Tuple[gn_graph.GraphsTuple, gn_graph.GraphsTuple, jnp.ndarray], None, None
]:
    """Dynamically batches trees with (`jraph.GraphsTuples`, `gn_graph.GraphsTuple`, `jnp.ndarray`) up to specified sizes. The function is based on the original `jraph._src.utils.dynamically_batch` function."""

    def append_to_array_tuple(array_tuple_1, array_tuple_2):
        """Appends all the arrays inside the two tuples. Has the advantage of being able to take different amounts of arrays in the tuples."""
        array_tuple = tuple(
            map(
                lambda arr1, arr2: np.concatenate([arr1, arr2], axis=0),
                array_tuple_1,
                array_tuple_2,
            )
        )
        return array_tuple

    def pad_array_tuple(array_tuple, n_node_probes):
        """Constructs and append appropirate padding sizes for the different sized arrays in the tuple."""
        n_pad = n_node_probes - array_tuple[0].shape[0]

        array_tuple_2d = tuple(map(atleast_2d_last, array_tuple))
        padding_zeros_tuple = tuple(
            map(lambda arr_comp: np.zeros((n_pad, arr_comp.shape[-1])), array_tuple_2d)
        )
        array_tuple_padded = append_to_array_tuple(array_tuple_2d, padding_zeros_tuple)
        # array_tuple_padded = tuple(
        #     map(lambda arr_comp: np.squeeze(arr_comp), array_tuple_padded)
        # )
        return array_tuple_padded

    def batch_and_pad_array_tuple(array_tuple_list, n_pad):
        combined_array_tuple = array_tuple_list[0]
        for array_tuple in array_tuple_list[1:]:
            combined_array_tuple = append_to_array_tuple(
                combined_array_tuple, array_tuple
            )
        return pad_array_tuple(combined_array_tuple, n_pad)

    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or "
            f"equal to `2` for padding with graphs, got {n_graph}."
        )
    #! Currently the probe graph sizes are not used to decide the batch size,
    #!   but the max size will still be fixed as they are bound to the normal graph size.
    n_node_org = n_node
    n_node_probes = n_node_org + n_probes * n_graph
    n_node = n_node_probes
    n_edge_probes = n_node * n_probes  # * n_graph

    # # Use the biggest max edge for encoding to be shared
    # n_edge = np.max([n_edge, n_edge_probes])
    # n_edge_probes = n_edge

    n_graph_probes = n_graph  # * 2

    valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
    accumulated_graphs = []
    accumulated_probe_graphs = []
    # accumulated_probe_targets = []
    accumulated_array_tuples = []
    num_accumulated_nodes = 0
    num_accumulated_edges = 0
    num_accumulated_graphs = 0
    for data_tuple in graph_operator_tuple_iterator:
        if not return_layout_info:
            graph_element, probe_graphs, array_tuple = data_tuple
        else:
            assert n_graph == 2, "When returning layout info, n_graph must be 2"
            (graph_element, probe_graphs, array_tuple), layout_type, wt_spacing = (
                data_tuple
            )

        graph_element_nodes, graph_element_edges, graph_element_graphs = (
            jraph_utils._get_graph_size(graph_element)
        )
        if jraph_utils._is_over_batch_size(graph_element, valid_batch_size):
            # First yield the batched graph so far if exists.
            if accumulated_graphs:
                batched_graph = jraph_utils.batch_np(accumulated_graphs)

                if not return_layout_info:
                    yield (
                        jraph_utils.pad_with_graphs(
                            batched_graph, n_node, n_edge, n_graph
                        ),
                        n_node,
                        n_edge,
                        n_graph,
                    )
                else:
                    yield (
                        jraph_utils.pad_with_graphs(
                            batched_graph, n_node, n_edge, n_graph
                        ),
                        n_node,
                        n_edge,
                        n_graph,
                        layout_type,
                        wt_spacing,
                    )

            # Then report the error.
            graph_size = graph_element_nodes, graph_element_edges, graph_element_graphs
            graph_size = {k: v for k, v in zip(jraph_utils._NUMBER_FIELDS, graph_size)}
            batch_size = {
                k: v for k, v in zip(jraph_utils._NUMBER_FIELDS, valid_batch_size)
            }
            raise RuntimeError(
                "Found graph bigger than batch size. Valid Batch "
                f"Size: {batch_size}, Graph Size: {graph_size}"
            )

        # If this is the first graph_element of the batch, set it and continue.
        # Otherwise check if there is space for the graph in the batch:
        #   if there is, add it to the batch
        #   if there isn't, return the old batch and start a new batch.
        if not accumulated_graphs:
            accumulated_graphs = [graph_element]
            accumulated_probe_graphs = [probe_graphs]
            accumulated_array_tuples = [array_tuple]
            num_accumulated_nodes = graph_element_nodes
            num_accumulated_edges = graph_element_edges
            num_accumulated_graphs = graph_element_graphs
            continue
        else:
            if (
                (num_accumulated_graphs + graph_element_graphs > n_graph - 1)
                or (num_accumulated_nodes + graph_element_nodes > n_node - 1)
                or (num_accumulated_edges + graph_element_edges > n_edge)
            ):
                batched_graph = jraph_utils.batch_np(accumulated_graphs)
                batched_probe_graphs = jraph_utils.batch_np(accumulated_probe_graphs)

                batched_and_padded_array_tuple = batch_and_pad_array_tuple(
                    accumulated_array_tuples, n_node_probes
                )

                if not return_layout_info:
                    yield (
                        jraph_utils.pad_with_graphs(
                            batched_graph, n_node, n_edge, n_graph
                        ),
                        jraph_utils.pad_with_graphs(
                            batched_probe_graphs,
                            n_node_probes,
                            n_edge_probes,
                            n_graph_probes,
                        ),
                        # padded_probe_targets,
                        batched_and_padded_array_tuple,
                    )
                else:
                    yield (
                        jraph_utils.pad_with_graphs(
                            batched_graph, n_node, n_edge, n_graph
                        ),
                        jraph_utils.pad_with_graphs(
                            batched_probe_graphs,
                            n_node_probes,
                            n_edge_probes,
                            n_graph_probes,
                        ),
                        # padded_probe_targets,
                        batched_and_padded_array_tuple,
                        layout_type,
                        wt_spacing,
                    )

                accumulated_graphs = [graph_element]
                accumulated_probe_graphs = [probe_graphs]
                # accumulated_probe_targets = [probe_targets]
                accumulated_array_tuples = [array_tuple]
                num_accumulated_nodes = graph_element_nodes
                num_accumulated_edges = graph_element_edges
                num_accumulated_graphs = graph_element_graphs
            else:
                accumulated_graphs.append(graph_element)
                accumulated_probe_graphs.append(probe_graphs)
                # accumulated_probe_targets.append(probe_targets)
                accumulated_array_tuples.append(array_tuple)

                num_accumulated_nodes += graph_element_nodes
                num_accumulated_edges += graph_element_edges
                num_accumulated_graphs += graph_element_graphs

    # We may still have data in batched graph.
    if accumulated_graphs:
        batched_graph = jraph_utils.batch_np(accumulated_graphs)
        batched_probe_graphs = jraph_utils.batch_np(accumulated_probe_graphs)
        # batched_probe_targets = np.asarray(batched_probe_targets)
        # padded_probe_targets = pad_array(batched_probe_targets, n_graph)
        # batched_probe_targets = np.concatenate(accumulated_probe_targets)
        # padded_probe_targets = pad_probe_targets_all_nodes_version(
        #     batched_probe_targets, n_node_probes
        # )
        # padded_probe_targets = jnp.asarray(padded_probe_targets)
        batched_and_padded_array_tuple = batch_and_pad_array_tuple(
            accumulated_array_tuples, n_node_probes
        )
        if not return_layout_info:
            yield (
                jraph_utils.pad_with_graphs(batched_graph, n_node, n_edge, n_graph),
                jraph_utils.pad_with_graphs(
                    batched_probe_graphs,
                    n_node_probes,
                    n_edge_probes,
                    n_graph_probes,
                ),
                # padded_probe_targets,
                batched_and_padded_array_tuple,
            )
        else:
            yield (
                jraph_utils.pad_with_graphs(batched_graph, n_node, n_edge, n_graph),
                jraph_utils.pad_with_graphs(
                    batched_probe_graphs,
                    n_node_probes,
                    n_edge_probes,
                    n_graph_probes,
                ),
                # padded_probe_targets,
                batched_and_padded_array_tuple,
                layout_type,
                wt_spacing,
            )


if __name__ == "__main__":
    import sys

    from graph import print_shapes
    from matplotlib import pyplot as plt

    sys.path.append(os.path.abspath("../"))

    from utils.data_tools import retrieve_dataset_stats

    data_path = os.path.abspath("../data/medium_graphs_nodes/train_pre_processed")
    # data_path = os.path.abspath("./data/medium_graphs_nodes/train_pre_processed")
    dataset = Torch_Geomtric_Dataset(data_path)

    batch_size = (
        1  #! Plot looks bestwith single graph otherwise nodes are tangled toghether
    )

    stats, scale_stats = retrieve_dataset_stats(
        dataset,
    )

    loader = JraphDataLoader(
        dataset,
        stats=stats,
        batch_size=batch_size,
        shuffle=True,
        idxs_per_sample=90000,
        add_pos_to_nodes=True,
    )

    for graphs, trunk_input, trunk_output in loader:
        print_shapes(graphs)
        print(trunk_input.shape)
        print(trunk_output)
        for xy, u in zip(
            np.split(trunk_input, batch_size, axis=0),
            np.split(trunk_output, batch_size, axis=0),
        ):
            xy = xy.squeeze()
            u = u.squeeze()
            wt_pos = graphs.nodes[:, -2:]
            plt.figure()
            plt.tricontourf(xy[:, 0], xy[:, 1], u)
            plt.scatter(wt_pos[:, 0], wt_pos[:, 1], c="r")
            plt.axis("equal")
            plt.show()
        break

    loader = JraphDataLoader(
        dataset,
        stats=stats,
        batch_size=batch_size,
        shuffle=True,
        idxs_per_sample=5,
        add_pos_to_nodes=True,
        trunk_sample_strategy="random_crossstream",
    )
    for graphs, trunk_input, trunk_output in loader:
        print_shapes(graphs)
        print(trunk_input.shape)
        print(trunk_output)
        for xy in np.split(trunk_input, batch_size, axis=0):
            xy = xy.squeeze()
            wt_pos = graphs.nodes[:, -2:]
            plt.figure()
            plt.scatter(xy[:, 0], xy[:, 1], c="b")
            plt.scatter(wt_pos[:, 0], wt_pos[:, 1], c="r")
            plt.axis("equal")
            plt.show()
        break

    loader = JraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        idxs_per_sample=2,
        add_pos_to_edges=False,
        sample_probabilities_path="../data/medium_graphs_nodes/probabilities_combined.npz",
    )

    max_graph = 50
    n_node = max_graph * 0.7
    n_edge = max_graph * 0.6

    dynamic_batcher = dynamically_batch_graph_operator(
        loader, n_node=1000, n_edge=3000, n_graph=max_graph
    )

    for graphs, trunk_input, trunk_output in dynamic_batcher:
        print_shapes(graphs)
        print(trunk_input.shape)
        print(trunk_output.shape)
        break

    n_probes = 2
    loader = JraphDataLoader(
        dataset,
        batch_size=1,  #! CAN ONLY BE 1 FOR "GNO_probe" MODEL TO WORK
        shuffle=True,
        idxs_per_sample=n_probes,
        add_pos_to_edges=False,
        add_pos_to_nodes=True,
        probe_graphs=True,
        input_node_feature_idxs=[3, 4],
        target_node_feature_idxs=[
            0,
        ],  # See function utils.data_tools.get_node_feature_idxs for explanation
    )

    for i, (graphs, probe_graphs, node_tuple_array) in enumerate(loader):
        probe_targets, wt_node_mask, probe_node_mask = node_tuple_array
        print_shapes(graphs)
        print_shapes(probe_graphs)
        print(probe_targets.shape)
        break

    dynamic_batcher = dynamically_batch_graph_probe_operator(
        loader, n_node=200, n_edge=500, n_graph=5, n_probes=n_probes
    )

    for padded_graphs, padded_probe_graphs, padded_node_tuple_array in dynamic_batcher:
        padded_probe_targets, padded_wt_node_mask, padded_probe_node_mask = (
            padded_node_tuple_array
        )
        print_shapes(padded_graphs)
        print_shapes(padded_probe_graphs)
        print(padded_probe_targets.shape)
        break

    def get_wt_and_probe_idxs(graphs, probe_graphs):
        unique_wt_senders_receivers = np.unique(
            np.concat([graphs.senders, graphs.receivers])
        )
        unique_probe_receivers = np.unique(probe_graphs.receivers)
        return unique_wt_senders_receivers, unique_probe_receivers

    unique_wt_senders_receivers, unique_probe_receivers = get_wt_and_probe_idxs(
        padded_graphs, padded_probe_graphs
    )
    overlapping_values = np.intersect1d(
        unique_wt_senders_receivers, unique_probe_receivers
    )
    print(overlapping_values)  # Should only be the padded node
    print(unique_wt_senders_receivers)
    print(unique_probe_receivers)

    n_probes = 200
    loader = JraphDataLoader(
        dataset,
        batch_size=1,  #! CAN ONLY BE 1 FOR "GNO_probe" MODEL TO WORK
        shuffle=True,
        idxs_per_sample=n_probes,
        add_pos_to_edges=False,
        add_pos_to_nodes=True,
        probe_graphs=True,
        input_node_feature_idxs=[3, 4],
        target_node_feature_idxs=[
            0,
        ],
        return_positions=True,
    )

    for graphs, probe_graphs, node_tuple_array in loader:
        probe_targets, wt_node_mask, probe_node_mask, node_positions = node_tuple_array

        print_shapes(graphs)
        print_shapes(probe_graphs)
        print(probe_targets.shape)
        break

    unique_wt_senders_receivers, unique_probe_receivers = get_wt_and_probe_idxs(
        graphs, probe_graphs
    )
    overlapping_values = np.intersect1d(
        unique_wt_senders_receivers, unique_probe_receivers
    )
    print(overlapping_values)  # Should only be the padded node
    print(unique_wt_senders_receivers)
    print(unique_probe_receivers)

    wt_positions = node_positions[unique_wt_senders_receivers, :]
    probe_positions = node_positions[unique_probe_receivers, :]

    # The two options below are equivalent and are only here as an example
    probe_edge_coordinates = [
        node_positions[probe_graphs.receivers, :] + probe_graphs.edges[:, :-1],
        node_positions[probe_graphs.receivers, :],
    ]  # ]
    probe_edge_coordinates = [
        node_positions[probe_graphs.senders, :],
        node_positions[probe_graphs.receivers, :],
    ]

    wf_edge_coordinates = [
        node_positions[graphs.receivers, :] + graphs.edges[:, :-1],
        node_positions[graphs.receivers, :],
    ]
    # wf_edge_coordinates = [node_positions[graphs.senders, :], node_positions[graphs.receivers, :]]

    fig = plt.figure()
    plt.scatter(
        wt_positions[:, 0],
        wt_positions[:, 1],
        c="b",
        marker="2",
        s=100,
        label="WT nodes",
    )
    plt.scatter(
        probe_positions[:, 0],
        probe_positions[:, 1],
        c="r",
        marker="o",
        s=20,
        label="Probe nodes",
    )
    for i, (sender, receiver) in enumerate(zip(*probe_edge_coordinates)):
        plt.plot(
            [sender[0], receiver[0]],
            [sender[1], receiver[1]],
            c="k",
            alpha=0.5,
            ls="-",
            linewidth=0.5,
            label="Probe edges" if i == 0 else "",
        )

    for i, (sender, receiver) in enumerate(zip(*wf_edge_coordinates)):
        plt.plot(
            [sender[0], receiver[0]],
            [sender[1], receiver[1]],
            c="g",
            alpha=1,
            ls="-",
            linewidth=0.5,
            label="WT edges" if i == 0 else "",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.show()
