"""
Implementaion of the GEnerailised aggregation Network (GEN) processing block from [1].
Curretly message normalization is not implemented, but could easily be included.
[1] Li, G., Xiong, C., Thabet, A., & Ghanem, B. (2020). DeeperGCN: All You Need to Train Deeper GCNs. http://arxiv.org/abs/2006.07739
"""

from typing import Any, Callable, Sequence, Iterable, Mapping, Union

import jax
from flax import linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils as jraph_utils
from jraph._src.models import NodeFeatures, ArrayTree, AggregateEdgesToNodesFn


Agg_Messages = ArrayTree
GENUpdateNodeFn = Callable[[NodeFeatures, Agg_Messages], NodeFeatures]


def softmax_aggregation(
    messages: ArrayTree, receivers: jnp.ndarray, sum_n_node: int
) -> ArrayTree:
    """Softmax aggregation of messages as defined in [1]"""
    message_softmax_attention = jax.tree.map(
        lambda m: jraph_utils.segment_softmax(m, receivers, num_segments=sum_n_node),
        messages,
    )
    softmaxed_messages = message_softmax_attention * messages
    aggregated_messages = jax.tree.map(
        lambda m: jraph_utils.segment_sum(m, receivers, num_segments=sum_n_node),
        softmaxed_messages,
    )
    return aggregated_messages


def GEN_block(
    node_update_fn: GENUpdateNodeFn,
    message_to_node_aggregation_fn: AggregateEdgesToNodesFn = softmax_aggregation,
    epsilon: float = 1e-6,
    message_norm: bool = False,
    message_norm_scale_param: Callable = None,
) -> Callable[[jraph.GraphsTuple], jraph.GraphsTuple]:
    """A GEN block as defined in [1] and [2]
    Implemented as a jraph block for easy integration with other jraph blocks
    Inspiration drawn from the GAT implementation (see jraph._src.models.GAT).

    Parameters
    ----------
    node_update_fn : GENUpdateNodeFn
        Function that updates the nodes in the graph.
    epsilon : float, optional
        Small value to avoid numerical instability, by default 1e-6

    Returns
    -------
    Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
        A function that applies a GEN layer.
    """

    def _ApplyGEN(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes, edges, receivers, senders, _, _, _ = graph

        assert (
            nodes.shape[-1] == edges.shape[-1]
        ), "Nodes and edges have to have the same feature dimension"
        assert len(senders) == len(
            receivers
        ), "Senders and receivers have to have the same length"
        assert (
            len(senders) == edges.shape[0]
        ), "Amount of senders and edges have to be the same"

        sum_n_node = jax.tree.leaves(nodes)[0].shape[0]

        # Gather node features corresponding to senders and receivers
        sent_attributes = jax.tree.map(lambda n: n[senders], nodes)
        # // received_attributes = jax.tree.map(lambda n: n[receivers], nodes)

        # Compute the messages
        messages = jax.tree.map(
            lambda e, s: nn.relu(e + s) + epsilon, edges, sent_attributes
        )

        aggregated_messages = message_to_node_aggregation_fn(
            messages, receivers, sum_n_node
        )

        if message_norm:
            """Message normalization as defined in [1] eq. (5):
            h' = \phi_h(h + s * ||h_i|| * m_i / ||m_i||), where
            s * ||h_i|| * m_i / ||m_i|| is the normalization term"""

            if message_norm_scale_param is None:
                raise ValueError(
                    "A message_norm_scale_param must be provided when message_norm is enabled."
                )

            agg_msg_norm = jnp.linalg.norm(aggregated_messages, axis=-1, keepdims=True)
            nodes_norm = jnp.linalg.norm(nodes, axis=-1, keepdims=True)
            aggregated_messages = (
                message_norm_scale_param
                * nodes_norm
                * aggregated_messages
                / (agg_msg_norm + epsilon)
            )

        # Update nodes using the aggregated messages
        updated_nodes = jax.tree.map(
            lambda n, agg_m: node_update_fn(n + agg_m), nodes, aggregated_messages
        )

        return graph._replace(nodes=updated_nodes)

    return _ApplyGEN


if __name__ == "__main__":
    import sys
    import os

    # add the 'src' directory as one where we can import modules
    src_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(src_dir)
    from utils.graph import get_test_graph
    from utils.plotting import draw_jraph_graph_structure
    from models.mlp import MLP

    graph = jraph.GraphsTuple(
        nodes=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        edges=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        receivers=jnp.array([0, 1, 2]),
        senders=jnp.array([1, 2, 0]),
        n_node=jnp.array([3]),
        n_edge=jnp.array([3]),
        globals=jnp.array([[1.0, 2.0]]),
    )

    # graph = get_test_graph(type="double_graph") #! Doesn't work curretnly because of edges with all 0's
    draw_jraph_graph_structure(graph)

    class GEN_test_model(nn.Module):
        latent_size: int
        num_mlp_layers: int
        num_trunk_layers: int
        trunk_latent_size: int
        epsilon: float = 1e-6
        message_norm: bool = False

        @nn.compact
        def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
            embedder = jraph.GraphMapFeatures(
                embed_node_fn=MLP(
                    name="embed_node",
                    feature_sizes=[self.latent_size] * self.num_mlp_layers,
                    output_size=self.latent_size,
                    activation=nn.relu,
                ),
                embed_edge_fn=MLP(
                    name="embed_edge",
                    feature_sizes=[self.latent_size] * self.num_mlp_layers,
                    output_size=self.latent_size,
                    activation=nn.relu,
                ),
            )

            processed_graph = embedder(graph)
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers

            node_update_fn = jraph.concatenated_args(
                MLP(
                    name=f"node_update",
                    feature_sizes=mlp_feature_sizes,
                    output_size=12,
                    activation=nn.relu,
                )
            )

            for i in range(1):
                if self.message_norm:
                    message_norm_scale = self.param(
                        f"message_norm_scale_{i}", nn.initializers.ones, (1,)
                    )
                else:
                    message_norm_scale = None
                GEN_layer = GEN_block(
                    node_update_fn=node_update_fn,
                    message_to_node_aggregation_fn=softmax_aggregation,
                    message_norm=self.message_norm,
                    message_norm_scale_param=message_norm_scale,
                    epsilon=self.epsilon,
                )
                processed_graph = GEN_layer(processed_graph)

            # Add global decoder
            return processed_graph

    # Define the model
    model = GEN_test_model(
        latent_size=12,
        num_mlp_layers=2,
        num_trunk_layers=2,
        trunk_latent_size=12,
        message_norm=True,
    )

    # Apply the model
    params = model.init(jax.random.PRNGKey(0), graph)
    prediction = model.apply(params, graph)
