"""
Implementaion of the GEnerailised aggregation Network (GEN) processing block from [1].
Curretly message normalization is not implemented, but could easily be included.
[1] Li, G., Xiong, C., Thabet, A., & Ghanem, B. (2020). DeeperGCN: All You Need to Train Deeper GCNs. http://arxiv.org/abs/2006.07739
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
from jraph._src import utils as jraph_utils
from jraph._src.models import AggregateEdgesToNodesFn, ArrayTree, NodeFeatures

Agg_Messages = ArrayTree
GENUpdateNodeFn = Callable[
    [NodeFeatures], NodeFeatures
]  # Takes concatenated (nodes + agg_messages)


def softmax_aggregation(messages: ArrayTree, receivers: jnp.ndarray, sum_n_node: int) -> ArrayTree:
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
    message_norm_scale_param: Callable | None = None,
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

        assert nodes is not None and edges is not None, "Nodes and edges must be present"
        assert senders is not None and receivers is not None, (
            "Senders and receivers must be present"
        )

        assert (
            nodes.shape[-1] == edges.shape[-1]  # type: ignore[possibly-missing-attribute]
        ), "Nodes and edges have to have the same feature dimension"
        assert len(senders) == len(receivers), "Senders and receivers have to have the same length"
        assert len(senders) == edges.shape[0], "Amount of senders and edges have to be the same"  # type: ignore[possibly-missing-attribute]

        sum_n_node = jax.tree.leaves(nodes)[0].shape[0]

        # Gather node features corresponding to senders and receivers
        sent_attributes = jax.tree.map(lambda n: n[senders], nodes)
        # // received_attributes = jax.tree.map(lambda n: n[receivers], nodes)

        # Compute the messages
        messages = jax.tree.map(lambda e, s: nn.relu(e + s) + epsilon, edges, sent_attributes)

        aggregated_messages = message_to_node_aggregation_fn(messages, receivers, sum_n_node)

        if message_norm:
            r"""Message normalization as defined in [1] eq. (5):
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
