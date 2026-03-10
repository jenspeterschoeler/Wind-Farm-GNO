"""Data loading and preprocessing utilities for GNO training."""

import json
import os

import jraph
import numpy as np
from jax import numpy as jnp
from jraph._src import utils as jraph_utils
from omegaconf import DictConfig

from .misc import add_to_hydra_cfg
from .torch_loader import (
    JraphDataLoader,
    Torch_Geomtric_Dataset,
    dynamically_batch_graph_probe_operator,
)

NestedListStr = str | list["NestedListStr"]

import logging  # noqa: E402

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def get_node_indexes(vars: list[str]) -> list[int]:
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


def setup_train_dataset(cfg: DictConfig, in_mem=True):
    data_type = cfg.data.type
    if data_type == "GraphFarmsOperatorDataset":
        if cfg.data.encoding == "torch":
            if cfg.data.pre_processed:
                dataset = Torch_Geomtric_Dataset(
                    root_path=os.path.abspath(cfg.data.train_path),
                    in_mem=in_mem,
                )
                (
                    dataset_stats,
                    dataset_scale_stats,
                ) = retrieve_dataset_stats(dataset)
            else:
                raise NotImplementedError(
                    "Only pre-processed datasets are supported for now, with the Torch based dataset."
                )

        else:
            raise NotImplementedError(f"Encoding {cfg.data.encoding} not implemented.")

    else:
        raise ValueError(f"Data type {data_type} not recognized.")
    cfg.data = add_to_hydra_cfg(cfg.data, "stats", dataset_stats)
    cfg.data = add_to_hydra_cfg(cfg.data, "scale_stats", dataset_scale_stats)
    return dataset, cfg


def setup_test_val_iterator(
    cfg: DictConfig,
    type_str: str,
    trunk_sample_strategy: str | None = None,
    idxs_per_sample: int | None = None,
    n_cuts: int | None = None,
    return_idxs: bool = False,
    return_positions: bool = False,
    cache: bool = True,
    path=None,
    return_layout_info: bool = False,
    num_workers: int | None = None,
    dataset: Torch_Geomtric_Dataset | None = None,
):
    data_type = cfg.data.type
    if path is None:
        if type_str == "val":
            path = cfg.data.val_path
        elif type_str == "test":
            path = cfg.data.test_path
        else:
            raise ValueError(f"Type {type_str} not recognized.")
    if trunk_sample_strategy != "evenly_distributed":
        assert idxs_per_sample is None or (trunk_sample_strategy is None), (
            "If idxs_per_sample is provided, trunk_sample_strategy cannot be provided."
        )

    if num_workers is None:
        num_workers = cfg.optimizer.validation.num_workers
    if num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = cfg.optimizer.validation.prefetch_factor

    if data_type == "GraphFarmsOperatorDataset":
        if cfg.data.encoding == "torch":
            logger.info(f"Loading data, with path: {path}")
            if dataset is None:
                dataset = Torch_Geomtric_Dataset(root_path=os.path.abspath(path), in_mem=cache)
            stats, scale_stats = retrieve_dataset_stats(dataset)

            data_iterator_kwargs = {
                "shuffle": False,
                "idxs_per_sample": idxs_per_sample,
                "batch_size": cfg.optimizer.validation.torch_batch,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "add_pos_to_nodes": cfg.data.io.add_pos_to_nodes,
                "add_pos_to_edges": cfg.data.io.add_pos_to_edges,
                "sample_stepsize": 1,  # default value, overwritten for GNO
                "return_layout_info": return_layout_info,
            }

            if cfg.data.io.type == "GNO_probe":
                data_iterator_kwargs["graphs_only"] = False
                data_iterator_kwargs["probe_graphs"] = True
                data_iterator_kwargs["trunk_sample_strategy"] = trunk_sample_strategy

                if trunk_sample_strategy == "random" or (
                    trunk_sample_strategy is None and idxs_per_sample is None
                ):
                    data_iterator_kwargs["idxs_per_sample"] = cfg.optimizer.batching.n_probe

                assert cfg.data.io.add_pos_to_nodes is True, "Probe edges require node positions."
                if cfg.data.io.add_pos_to_edges is True:
                    raise NotImplementedError(
                        "GNO_probe does not support appending positions to edges."
                    )
                data_iterator_kwargs["input_node_feature_idxs"] = get_node_indexes(
                    cfg.data.io.input_node_features
                )
                data_iterator_kwargs["target_node_feature_idxs"] = get_node_indexes(
                    cfg.data.io.target_node_features
                )

            elif cfg.data.io.type == "GNN":
                data_iterator_kwargs["graphs_only"] = True
                data_iterator_kwargs["node_feature_idxs"] = (
                    None  # HACK The data_iterator is split later and therefore all of the node features are used for GNN
                )

            data_iterator = JraphDataLoader(
                dataset,
                stats=stats,
                return_idxs=return_idxs,
                return_positions=return_positions,
                **data_iterator_kwargs,
            )
            max_nodes = (
                cfg.data.stats.graph_size.max_n_nodes + 1
            )  # +1 to account for 0 vs 1 indexing
            max_edges = cfg.data.stats.graph_size.max_n_edges + 1

            if cfg.data.io.type == "GNN":

                def get_refreshed_iterator():
                    padded_loader = jraph_utils.dynamically_batch(
                        iter(data_iterator),
                        n_node=max_nodes,
                        n_edge=max_edges,
                        n_graph=2,
                    )
                    return padded_loader

            elif cfg.data.io.type == "GNO_probe":
                n_graph = 2
                max_nodes = int(
                    cfg.optimizer.batching.max_node_ratio
                    * cfg.data.stats.graph_size.max_n_nodes
                    * n_graph
                )
                max_edges = int(
                    cfg.optimizer.batching.max_edge_ratio
                    * cfg.data.stats.graph_size.max_n_edges
                    * n_graph
                )

                def get_refreshed_iterator():
                    assert data_iterator.idxs_per_sample is not None, "idxs_per_sample must be set"
                    padded_loader = dynamically_batch_graph_probe_operator(
                        iter(data_iterator),
                        n_node=max_nodes,
                        n_edge=max_edges,
                        n_graph=n_graph,
                        n_probes=data_iterator.idxs_per_sample,
                        return_layout_info=return_layout_info,
                    )
                    return padded_loader

            else:
                raise NotImplementedError(
                    f"IO type {cfg.data.io.type} not implemented for batching."
                )

    else:
        raise ValueError(f"Data type {data_type} not recognized.")
    return get_refreshed_iterator, dataset, stats, scale_stats


def setup_refresh_iterator(cfg: DictConfig, train_dataset):
    if cfg.optimizer.batching.type == "dynamic_graph_batching":
        if cfg.data.type == "GraphFarmsOperatorDataset" and cfg.data.encoding == "torch":
            max_graphs = cfg.optimizer.batching.max_graph_size
            max_nodes = int(
                cfg.optimizer.batching.max_node_ratio
                * cfg.data.stats.graph_size.max_n_nodes
                * max_graphs
            )
            max_edges = int(
                cfg.optimizer.batching.max_edge_ratio
                * cfg.data.stats.graph_size.max_n_edges
                * max_graphs
            )

            if cfg.data.io.type == "GNO":
                if cfg.model.trunk_info == "wf_distance":
                    add_pos_to_nodes = False
                    add_pos_to_edges = True

                elif cfg.model.trunk_info == "turbine_distances":
                    add_pos_to_nodes = True
                    add_pos_to_edges = True

                add_to_hydra_cfg(cfg.data.io, "add_pos_to_nodes", add_pos_to_nodes)
                add_to_hydra_cfg(cfg.data.io, "add_pos_to_edges", add_pos_to_edges)

                if cfg.optimizer.batching.sample_probabilities == "combined":
                    sample_probabilities_path = os.path.join(
                        f"{cfg.data.main_path}",
                        f"probabilities_{cfg.optimizer.batching.sample_probabilities}.npz",
                    )
                    logger.info(f"Using sample probabilities from: {sample_probabilities_path}")
                elif cfg.optimizer.batching.sample_probabilities.lower() == "none":
                    sample_probabilities_path = None
                    logger.info("No sample probabilities used.")
                else:
                    raise ValueError(
                        f"Sample probabilities {cfg.optimizer.batching.sample_probabilities} not recognized."
                    )
                idxs_per_sample = cfg.optimizer.batching.idxs_per_sample
                graphs_only = True
                probe_graphs = False

            elif cfg.data.io.type == "GNN" or cfg.data.io.type == "GNO_probe":
                add_pos_to_edges = cfg.data.io.add_pos_to_edges
                add_pos_to_nodes = cfg.data.io.add_pos_to_nodes
                sample_probabilities_path = None

                if cfg.data.io.type == "GNN":
                    graphs_only = True
                    probe_graphs = False
                    idxs_per_sample = None  # No probe sampling for GNN type
                elif cfg.data.io.type == "GNO_probe":
                    graphs_only = False
                    probe_graphs = True
                    idxs_per_sample = cfg.optimizer.batching.n_probe
                    input_node_feature_idxs = get_node_indexes(cfg.data.io.input_node_features)
                    target_node_feature_idxs = get_node_indexes(cfg.data.io.target_node_features)

            else:
                raise NotImplementedError(
                    f"IO type {cfg.data.io.type} not implemented for batching."
                )

            train_iterator = JraphDataLoader(
                train_dataset,
                shuffle=True,
                batch_size=cfg.optimizer.batching.torch_batch,
                num_workers=cfg.optimizer.batching.num_workers,
                prefetch_factor=(
                    cfg.optimizer.batching.prefetch_factor
                    if cfg.optimizer.batching.num_workers > 0
                    else None
                ),
                idxs_per_sample=idxs_per_sample,
                sample_probabilities_path=sample_probabilities_path,
                add_pos_to_nodes=add_pos_to_nodes,
                add_pos_to_edges=add_pos_to_edges,
                graphs_only=graphs_only,
                probe_graphs=probe_graphs,
                input_node_feature_idxs=input_node_feature_idxs,
                target_node_feature_idxs=target_node_feature_idxs,
            )

            if cfg.data.io.type == "GNO_probe":

                def get_refreshed_iterator():
                    assert train_iterator.idxs_per_sample is not None, "idxs_per_sample must be set"
                    padded_loader = dynamically_batch_graph_probe_operator(
                        iter(train_iterator),
                        n_node=max_nodes,
                        n_edge=max_edges,
                        n_graph=max_graphs,
                        n_probes=train_iterator.idxs_per_sample,
                    )
                    return padded_loader

            elif cfg.data.io.type == "GNN":

                def get_refreshed_iterator():
                    padded_loader = jraph_utils.dynamically_batch(
                        iter(train_iterator),
                        n_node=max_nodes,
                        n_edge=max_edges,
                        n_graph=max_graphs,
                    )
                    return padded_loader

    else:
        raise ValueError(f"Batching type {cfg.optimizer.batching.type} not recognized.")
    return get_refreshed_iterator, train_iterator


def pad_graph_operator_triplet(
    graph: jraph.GraphsTuple,
    trunk_and_ouput: jnp.ndarray,
    max_nodes: int,
    max_edges: int,
    max_graphs: int,
    return_split_tao: bool = False,
) -> tuple[jraph.GraphsTuple, jnp.ndarray] | tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
    padded_graph = jraph_utils.pad_with_graphs(graph, max_nodes, max_edges, max_graphs)
    trunk_and_ouput = jnp.atleast_2d(trunk_and_ouput)
    padded_trunk_and_output = jnp.concatenate(
        [
            trunk_and_ouput,
            jnp.zeros((max_graphs - trunk_and_ouput.shape[0], trunk_and_ouput.shape[1])),
        ],
        axis=0,
    )
    if return_split_tao:
        padded_trunk = padded_trunk_and_output[:, 0:2]
        padded_output = padded_trunk_and_output[:, 2:]
        return padded_graph, padded_trunk, padded_output
    else:
        return padded_graph, padded_trunk_and_output


def unpad_output(output: jnp.ndarray, graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Unpad the output array to match the number of non paded graphs in the graphs tuple."""
    n_padding_graphs = jraph_utils.get_number_of_padding_with_graphs_graphs(graphs)
    return jnp.squeeze(output[:-n_padding_graphs])


class online_stats_alg:
    """Online algorithm to compute mean, variance/std, min and max
    For the calculation of mean and variance, the Welford's online algorithm is used [1].
        [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, target_shape: int) -> None:
        self.count = 0
        self.mean = jnp.zeros(target_shape)
        self.M2 = jnp.zeros(target_shape)
        self.min = jnp.ones((1, target_shape)) * 1e9
        self.max = jnp.ones((1, target_shape)) * -1e9

    # For a new value new_value, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, new_value: jnp.ndarray) -> None:
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

        self.min = jnp.min(
            jnp.concatenate([self.min, jnp.atleast_2d(new_value)], axis=0),
            axis=0,
            keepdims=True,
        )
        self.max = jnp.max(
            jnp.concatenate([self.max, jnp.atleast_2d(new_value)], axis=0),
            axis=0,
            keepdims=True,
        )

    # Retrieve the mean, variance and sample variance from an aggregate
    def finalize(self) -> dict | float:
        if self.count < 2:
            return float("nan")
        else:
            mean = self.mean
            variance = self.M2 / self.count
            std = jnp.sqrt(variance)
            stats_dict = {
                "mean": mean,
                "variance": variance,
                "std": std,
                "min": jnp.squeeze(self.min),
                "max": jnp.squeeze(self.max),
            }
            return stats_dict


def retrieve_dataset_stats(dataset: Torch_Geomtric_Dataset) -> tuple[dict, dict]:
    path = dataset._root_path  # Access the private attribute of the dataset
    if "train_pre_processed" in path or "test_pre_processed" in path or "val_pre_processed" in path:
        path = os.path.dirname(path)

    stats_path = os.path.join(path, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info(
            f"Dataset stats loaded from: {stats_path}, \nComputed on: {stats['date_ISO8601']}"
        )
    else:
        raise FileNotFoundError(f"Dataset stats not found at: {stats_path}")

    scale_stats_path = os.path.join(path, "scale_stats.json")
    if os.path.exists(stats_path):
        with open(scale_stats_path) as f:
            scale_stats = json.load(f)
        logger.info(
            f"Scale stats loaded from: {scale_stats_path}, \nComputed on: {stats['date_ISO8601']}"
        )
    else:
        raise FileNotFoundError(f"Scale stats not found at: {scale_stats_path}")
    return stats, scale_stats


class standard_unscaler:
    """Class to un-scale data scaled with a standard scaler."""

    def __init__(self, scale_stats, epsilon=1e-6) -> None:
        self.scale_stats = scale_stats
        self.epsilon = epsilon

        assert scale_stats["scaling_type"] == "standard", (
            "Only standard scaling is supported, also check if scaling_type is in scale_stats otherwise manually adding it could be an option."
        )
        if scale_stats["scaling_method"] == "run1":
            """This is a backwards compatible method that uses the mean and std of separate componets without considering what the scales relates to."""
            self.data_stats = scale_stats["data_stats"]

    def __call__(
        self, graph: jraph.GraphsTuple, trunk_input: jnp.ndarray, output: jnp.ndarray
    ) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
        graph = self.inverse_scale_graph(graph)
        trunk_input = self.inverse_scale_trunk_input(trunk_input)
        output = self.inverse_scale_output(output)
        return graph, trunk_input, output

    def inverse_scale_trunk_input(self, input: jnp.ndarray) -> jnp.ndarray:
        input_mean = jnp.array(self.data_stats["trunk"]["mean"])
        input_std = jnp.array(self.data_stats["trunk"]["std"])
        return input * input_std + input_mean

    def inverse_scale_output(self, output: jnp.ndarray) -> jnp.ndarray:
        output_mean = jnp.array(self.data_stats["output"]["mean"])
        output_std = jnp.array(self.data_stats["output"]["std"])
        return output * output_std + output_mean

    def inverse_scale_graph(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        assert graph.nodes is not None and graph.edges is not None and graph.globals is not None
        graph_global_mean = jnp.array(self.data_stats["global_features"]["mean"])
        graph_global_std = jnp.array(self.data_stats["global_features"]["std"])
        graph_nodes_mean = jnp.array(self.data_stats["node_features"]["mean"])
        graph_nodes_std = jnp.array(self.data_stats["node_features"]["std"])
        graph_edges_mean = jnp.array(self.data_stats["edge_features"]["mean"])
        graph_edges_std = jnp.array(self.data_stats["edge_features"]["std"])

        new_nodes = (graph.nodes * (graph_nodes_std + self.epsilon)) + graph_nodes_mean  # type: ignore[operator]
        new_edges = (graph.edges * (graph_edges_std + self.epsilon)) + graph_edges_mean  # type: ignore[operator]
        new_globals = (graph.globals * (graph_global_std + self.epsilon)) + graph_global_mean  # type: ignore[operator]

        graph = graph._replace(nodes=new_nodes, edges=new_edges, globals=new_globals)
        return graph


class minmax_unscaler:
    """Class to un-scale data scaled with a standard scaler."""

    def __init__(self, scale_stats, node_vars, epsilon=1e-6) -> None:
        self.scale_stats = scale_stats
        self.node_vars = node_vars
        self.epsilon = epsilon
        assert scale_stats["scaling_type"] == "min_max", "Only minmax scaling is supported."

        if (
            scale_stats["scaling_method"] == "run2"
            or scale_stats["scaling_method"] == "run3"
            or scale_stats["scaling_method"] == "run4"
        ):
            """This is a method that uses shared velocity, distance etc. values to scale to keep the scaling consistent between graph, input and output."""
            velocity_min = scale_stats["velocity"]["min"]
            velocity_range = scale_stats["velocity"]["range"]
            distance_min = scale_stats["distance"]["min"]
            distance_range = scale_stats["distance"]["range"]
            ti_min = scale_stats["ti"]["min"]
            ti_range = scale_stats["ti"]["range"]
            ct_min = scale_stats["ct"]["min"]
            ct_range = scale_stats["ct"]["range"]

            if scale_stats["scaling_method"] == "run3":
                # HACK! zeros to enable res-net if we want
                velocity_min = 0  #
                ti_min = 0
                ct_min = 0
            elif scale_stats["scaling_method"] == "run4":
                # HACK! zeros to enable res-net if we want, also required for constructing the connections on the fly e.g. for probe setup
                distance_min = 0
                velocity_min = 0  #
                ti_min = 0
                ct_min = 0

            self.global_min = np.array([velocity_min, ti_min]).squeeze()
            self.global_range = np.array([velocity_range, ti_range]).squeeze()

            self.edge_min = np.array([distance_min]).squeeze()
            self.edge_range = np.array([distance_range]).squeeze()

            self.trunk_min = np.array([distance_min]).squeeze()
            self.trunk_range = np.array([distance_range]).squeeze()
            self.output_min = np.array([velocity_min]).squeeze()
            self.output_range = np.array([velocity_range]).squeeze()

            if self.node_vars == ["u", "ti", "ct"]:
                self.node_min = np.array(
                    [velocity_min, ti_min, ct_min]
                ).squeeze()  # HACK zeros to enable res-net if we want
                self.node_range = np.array([velocity_range, ti_range, ct_range]).squeeze()
            elif self.node_vars == ["u", "ti"]:
                self.node_min = np.array([velocity_min, ti_min]).squeeze()
                self.node_range = np.array([velocity_range, ti_range]).squeeze()
            elif self.node_vars == ["u"]:
                self.node_min = np.array([velocity_min]).squeeze()
                self.node_range = np.array([velocity_range]).squeeze()
            else:
                raise ValueError(f"Node variables {self.node_vars} not recognized.")

        else:
            raise ValueError(
                f"{scale_stats['scaling_run']} is unrecognized check combination of run and type. Alternatively, the scaling method is not implemented."
            )

    def __call__(
        self, graph: jraph.GraphsTuple, trunk_input: jnp.ndarray, output: jnp.ndarray
    ) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
        graph = self.inverse_scale_graph(graph)
        trunk_input = self.inverse_scale_trunk_input(trunk_input)
        output = self.inverse_scale_output(output)
        return graph, trunk_input, output

    def inverse_scale_trunk_input(self, input: jnp.ndarray) -> jnp.ndarray:
        return input * self.trunk_range + self.trunk_min

    def inverse_scale_output(self, output: jnp.ndarray) -> jnp.ndarray:
        return output * self.output_range + self.output_min

    def inverse_nodes(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """Function separated for specific GNN case"""
        return nodes * self.node_range + self.node_min

    def inverse_scale_graph(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        graph_globals = graph.globals
        graph_nodes = graph.nodes
        graph_edges = graph.edges

        new_globals = (graph_globals * self.global_range) + self.global_min
        new_nodes = self.inverse_nodes(graph_nodes)  # type: ignore[arg-type]
        new_edges = (graph_edges * self.edge_range) + self.edge_min

        graph = graph._replace(globals=new_globals, nodes=new_nodes, edges=new_edges)
        return graph


def setup_unscaler(cfg: DictConfig, scale_stats: dict) -> standard_unscaler | minmax_unscaler:
    graph_component = cfg.data.io.graph_components_target
    if (
        cfg.data.io.type == "GNO_probe"
    ):  # HACK the keywords should be more similar for this to not be necesary
        target_vars = [var.lower() for var in cfg.data.io.target_node_features]
    else:
        target_vars = [var.lower() for var in cfg.data.io.targets]

    if scale_stats["scaling_type"] == "standard":
        unscaler = standard_unscaler(scale_stats)
    elif scale_stats["scaling_type"] == "min_max":
        if cfg.data.io.graph_components_target == "nodes":
            unscaler = minmax_unscaler(scale_stats, target_vars)
        else:
            raise NotImplementedError(
                f"Graph component {graph_component} not implemented for min_max scaling."
            )
    else:
        raise ValueError(f"Scaling type {scale_stats['scaling_type']} not recognized.")
    return unscaler
