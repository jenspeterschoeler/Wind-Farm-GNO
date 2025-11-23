"""
Windfarm_GNN.py:
=================

[1] Li, G., Xiong, C., Thabet, A., & Ghanem, B. (2020). DeeperGCN: All You Need to Train Deeper GCNs. http://arxiv.org/abs/2006.07739
"""

import os
import sys

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_path)

import logging
from typing import Any, Dict

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from models.GEN_block import GEN_block, softmax_aggregation
from models.mlp import MLP
from models.RBF_encoder import RBFEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Windfarm_GNN(nn.Module):
    name: str
    target_size: int
    latent_size: int
    hidden_layer_size: int
    num_mlp_layers: int
    message_passing_steps: int
    decoder_hidden_layer_size: int = None
    num_decoder_layers: int = None
    encoder_dropout_rate: float = 0.0
    processor_dropout_rate: float = 0.0
    layer_norm_encoder: bool = False
    layer_norm_processor: bool = False
    message_norm: bool = False
    layer_norm_decoder: bool = False  #! Should always be false
    res_net: bool = False
    RBF_encoder_kwargs: Dict[str, Any] = None
    encode: bool = True
    decode: bool = True  # If False returns node latent space intended for probe model
    epsilon: float = 1e-6

    def setup(self):
        super().setup()
        assert (
            self.message_norm * self.layer_norm_processor != 1
        ), "Cannot have both message norm and layer norm in the processor"

    @nn.compact
    def __call__(
        self,
        graphs: jraph.GraphsTuple,
        train: bool = False,  #! True during training, only releveant for dropout
    ) -> jnp.ndarray:

        # TODO Should this replace embedder (probably not)?
        if self.encode:
            if self.RBF_encoder_kwargs is not None:
                RBF_encoder = RBFEncoder(**self.RBF_encoder_kwargs)
                encoded_edges = RBF_encoder(
                    graphs.edges
                )  #! THIS ASSUMES THAT THE EDGES ARE ALL DISTANCES
                _encoded_edges_raveled = jnp.transpose(encoded_edges, (0, 2, 1))
                encoded_edges_raveled = _encoded_edges_raveled.reshape(
                    encoded_edges.shape[0],
                    encoded_edges.shape[1] * encoded_edges.shape[2],
                )
                graphs = graphs._replace(edges=encoded_edges_raveled)

            # Embed the nodes and edges
            embed_node_fn = MLP(
                name="embed_node",
                feature_sizes=[self.hidden_layer_size] * self.num_mlp_layers,
                output_size=self.latent_size,
                activation=nn.relu,
                dropout_rate=self.encoder_dropout_rate,
                layer_norm=self.layer_norm_encoder,
            )
            embed_edge_fn = MLP(
                name="embed_edge",
                feature_sizes=[self.hidden_layer_size] * self.num_mlp_layers,
                output_size=self.latent_size,
                activation=nn.relu,
                dropout_rate=self.encoder_dropout_rate,
                layer_norm=self.layer_norm_encoder,
            )

            embedder = jraph.GraphMapFeatures(
                embed_node_fn=lambda n, **kwargs: embed_node_fn(
                    n,
                    train=train,
                ),
                embed_edge_fn=lambda e, **kwargs: embed_edge_fn(
                    e,
                    train=train,
                ),
                embed_global_fn=None,
            )
            processed_graphs = embedder(graphs)
        else:
            processed_graphs = graphs

        # GNN block
        mlp_feature_sizes = [self.hidden_layer_size] * self.num_mlp_layers
        for i in range(self.message_passing_steps):

            if self.message_norm:
                message_norm_scale = self.param(
                    f"message_norm_scale_{i}", nn.initializers.ones, (1,)
                )
                node_update_layer_norm = False
            else:
                message_norm_scale = None
                node_update_layer_norm = self.layer_norm_processor

            original_node_update_fn = MLP(
                name=f"node_update_{i}",
                feature_sizes=mlp_feature_sizes,
                output_size=self.latent_size,
                activation=nn.relu,
                dropout_rate=self.processor_dropout_rate,
                layer_norm=node_update_layer_norm,  # Special case due to potential message norm
            )

            def node_update_fn(n, **kwargs):
                return original_node_update_fn(n, train=train)

            # node_update_fn = lambda n, **kwargs: original_node_update_fn(n, train=train)

            GEN_layer = GEN_block(
                node_update_fn=node_update_fn,
                message_to_node_aggregation_fn=softmax_aggregation,
                message_norm=self.message_norm,
                message_norm_scale_param=message_norm_scale,
                epsilon=self.epsilon,
            )
            processed_graphs = GEN_layer(processed_graphs)

        # Decoder
        if self.decode:
            if self.decoder_hidden_layer_size is None:
                decoder_hidden_layer_size = self.hidden_layer_size
            else:
                decoder_hidden_layer_size = self.decoder_hidden_layer_size
            if self.num_decoder_layers is None:
                num_decoder_layers = self.num_mlp_layers
            else:
                num_decoder_layers = self.num_decoder_layers

            decoder = MLP(
                name="decoder",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
            )

            if self.res_net:
                """This only works if the initial values are U and TI"""
                delta_nodes = decoder(processed_graphs.nodes)
                new_nodes = graphs.nodes + delta_nodes
            else:
                new_nodes = decoder(processed_graphs.nodes)
        else:
            new_nodes = processed_graphs.nodes

        proto_mask = jnp.sum(jnp.abs(graphs.nodes), axis=-1)
        mask = jnp.bool(proto_mask).reshape(-1, 1)
        new_nodes = new_nodes * mask

        return new_nodes


if __name__ == "__main__":
    import torch

    from utils import (
        JraphDataLoader,
        Torch_Geomtric_Dataset,
        dynamically_batch_graph_operator,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    mini_model = MLP(
        name="mini_model",
        feature_sizes=[64, 64],
        output_size=2,
        activation=nn.relu,
        dropout_rate=0.0,
        layer_norm=False,
    )

    mini_parms = mini_model.init(jax.random.PRNGKey(0), jnp.ones((5, 64)))
    mini_output = mini_model.apply(mini_parms, jnp.ones((5, 64)))

    case = 2
    if case == 1:
        # TODO this option does not work currently because the positional information is only added to the nodes
        add_pos_to_nodes = True
        add_pos_to_edges = False

    elif case == 2:
        add_pos_to_nodes = False
        add_pos_to_edges = True

    data_path = os.path.abspath("../data/medium_graphs_nodes/train_pre_processed")
    # data_path = os.path.abspath(
    #     "/work/users/jpsch/SPO_sophia_dir/data/medium_graphs_nodes/train_pre_processed"
    # )  # sophia version
    dataset = Torch_Geomtric_Dataset(data_path)
    loader = JraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        idxs_per_sample=1,
        add_pos_to_nodes=add_pos_to_nodes,
        add_pos_to_edges=False,
    )
    padded_loader = (
        dynamically_batch_graph_operator(  # TODO Change to default jraph loader
            loader, n_node=100, n_edge=300, n_graph=5
        )
    )
    for graphs, _, _ in padded_loader:
        # graphs_input = graphs
        # TODO Make assertion on the type of dataset that is loaded beacuase this will not work with all datasets
        graph_nodes_input = graphs.nodes[:, 3:]
        graphs_nodes_output = graphs.nodes[:, :2]
        graphs_input = graphs._replace(nodes=graph_nodes_input)
        graphs_output = graphs._replace(nodes=graphs_nodes_output)
        break

    # Define the model
    model = Windfarm_GNN(
        target_size=graphs_output.nodes.shape[-1],
        latent_size=64,
        hidden_layer_size=64,
        num_mlp_layers=2,
        message_passing_steps=2,
        decoder_hidden_layer_size=64,
        num_decoder_layers=2,
        encoder_dropout_rate=0.0,
        processor_dropout_rate=0.0,
        layer_norm_encoder=False,
        layer_norm_processor=False,
        message_norm=False,
        layer_norm_decoder=False,
        epsilon=1e-6,
    )

    params = model.init(jax.random.PRNGKey(0), graphs_input, train=False)
    output = model.apply(params, graphs_input)
    print(params)
    print(output.shape)

    @jax.jit
    def predict_fn(graphs):
        return model.apply(params, graphs)

    prediction = predict_fn(graphs_input)

    print(prediction.shape)
