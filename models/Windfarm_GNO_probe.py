"""
Windfarm_GNO_probe.py:
=========================

Graph Neural Operator (GNO) model for wind farm flow prediction with optional
global conditioning support.

A variation of the DeepGraphOperatorNetwork (DeepGraphONet) model by Sun et. al. [1].
Termed a Graph Neural Operator (GNO), it combines Graph Learning and Operator Learning
by integrating Message Passing Graph Networks with DeepOperatorNetwork (DeepONet)[4].

[1] Sun, Y., Moya, C., Lin, G., & Yue, M. (2022). DeepGraphONet: A Deep Graph Operator
    Network to Learn and Zero-shot Transfer the Dynamic Response of Networked Systems.
[2] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning
    on Large Graphs.
[3] Li, G., Xiong, C., Thabet, A., & Ghanem, B. (2020). DeeperGCN: All You Need to Train
    Deeper GCNs.
[4] Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear
    operators via DeepONet.
[5] Seidman, J. H., Kissas, G., Perdikaris, P., & Pappas, G. J. (2022). NOMAD: Nonlinear
    Manifold Decoders for Operator Learning.
[6] de Santos, F. N., Duthé, G., Abdallah, I., Réthoré, P.-É., Weijtjens, W., Chatzi, E.,
    & Devriendt, C. (n.d.). Multivariate prediction on wake-affected wind turbines using
    graph neural networks.
"""

import logging
from typing import Any

import jax.numpy as jnp
import jraph
from flax import linen as nn

from models.mlp import MLP
from models.RBF_encoder import RBFEncoder
from models.Windfarm_GNN import Windfarm_GNN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Windfarm_GNO_probe(nn.Module):
    target_size: int
    latent_size: int
    hidden_layer_size: int
    num_mlp_layers: int
    wt_message_passing_steps: int
    probe_message_passing_steps: int
    decoder_hidden_layer_size: int | None = None
    num_decoder_layers: int | None = None
    decoder_strategy: str = "shared"
    encoder_dropout_rate: float = 0.0
    processor_dropout_rate: float = 0.0
    layer_norm_encoder: bool = False
    layer_norm_processor: bool = False
    message_norm: bool = False
    layer_norm_decoder: bool = False
    res_net: bool = False
    RBF_encoder_kwargs: dict[str, Any] | None = None
    epsilon: float = 1e-6
    # Global conditioning parameter
    use_global_conditioning: bool = False
    # LoRA parameters - granular control per component
    use_lora_embedder: bool = False
    use_lora_processor: bool = False
    use_lora_decoder: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(
        self,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask: jnp.ndarray,
        probe_mask: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        ### Pre-process
        pre_processed_graphs, pre_processed_probe_graphs = self.embedder(
            graphs, probe_graphs, train
        )

        ### Get wind turbine state with the "classic" GNN
        pre_processed_probe_graphs, latentspace_nodes = self.wt_processor(
            pre_processed_graphs, pre_processed_probe_graphs, wt_mask, probe_mask, train
        )

        # Message pass to the probe positions
        processed_probe_nodes = self.probe_processor(pre_processed_probe_graphs, train)

        # Extract globals for conditioning if enabled
        globals_context = graphs.globals if self.use_global_conditioning else None

        ## Decoder
        new_nodes = self.decoder(
            latentspace_nodes,
            processed_probe_nodes,
            wt_mask,
            probe_mask,
            probe_graphs,
            globals_context=globals_context,  # type: ignore[invalid-argument-type]
        )
        return new_nodes

    @nn.compact
    def embedder(self, graphs, probe_graphs, train=False):
        ### Pre-process
        if self.RBF_encoder_kwargs is not None:
            RBF_encoder = RBFEncoder(**self.RBF_encoder_kwargs)

            def RBF_encode_distances(jraph_graphs):
                encoded_edges = RBF_encoder(jraph_graphs.edges)
                _encoded_edges_raveled = jnp.transpose(encoded_edges, (0, 2, 1))
                encoded_edges_raveled = _encoded_edges_raveled.reshape(
                    encoded_edges.shape[0],
                    encoded_edges.shape[1] * encoded_edges.shape[2],
                )
                jraph_graphs = jraph_graphs._replace(edges=encoded_edges_raveled)
                return jraph_graphs

            pre_processed_graphs = RBF_encode_distances(graphs)
            pre_processed_probe_graphs = RBF_encode_distances(probe_graphs)
        else:
            pre_processed_graphs = graphs
            pre_processed_probe_graphs = probe_graphs

        # Embed the nodes and edges
        embed_node_fn = MLP(
            name="embed_node",
            feature_sizes=[self.hidden_layer_size] * self.num_mlp_layers,
            output_size=self.latent_size,
            activation=nn.relu,
            dropout_rate=self.encoder_dropout_rate,
            layer_norm=self.layer_norm_encoder,
            use_lora=self.use_lora_embedder,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        embed_edge_fn = MLP(
            name="embed_edge",
            feature_sizes=[self.hidden_layer_size] * self.num_mlp_layers,
            output_size=self.latent_size,
            activation=nn.relu,
            dropout_rate=self.encoder_dropout_rate,
            layer_norm=self.layer_norm_encoder,
            use_lora=self.use_lora_embedder,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
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

        pre_processed_graphs = embedder(pre_processed_graphs)
        pre_processed_probe_graphs = embedder(pre_processed_probe_graphs)
        return pre_processed_graphs, pre_processed_probe_graphs

    @nn.compact
    def wt_processor(
        self,
        pre_processed_graphs,
        pre_processed_probe_graphs,
        wt_mask,
        probe_mask,
        train=False,
    ):
        # Encode the graph(windfarm) information into a latent space
        wt_GNN = Windfarm_GNN(
            name="Windfarm_GNN_0",
            target_size=self.target_size,
            latent_size=self.latent_size,
            hidden_layer_size=self.hidden_layer_size,
            num_mlp_layers=self.num_mlp_layers,
            message_passing_steps=self.wt_message_passing_steps,
            decoder_hidden_layer_size=self.decoder_hidden_layer_size,
            num_decoder_layers=self.num_decoder_layers,
            encoder_dropout_rate=self.encoder_dropout_rate,
            processor_dropout_rate=self.processor_dropout_rate,
            layer_norm_encoder=self.layer_norm_encoder,
            layer_norm_processor=self.layer_norm_processor,
            message_norm=self.message_norm,
            layer_norm_decoder=self.layer_norm_decoder,
            res_net=self.res_net,
            RBF_encoder_kwargs=self.RBF_encoder_kwargs,
            epsilon=self.epsilon,
            encode=False,
            decode=False,
            use_lora_embedder=False,
            use_lora_processor=self.use_lora_processor,
            use_lora_decoder=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        latentspace_nodes = wt_GNN(pre_processed_graphs, train=train)
        latentspace_nodes = latentspace_nodes * wt_mask + pre_processed_graphs.nodes * probe_mask

        pre_processed_probe_graphs = pre_processed_probe_graphs._replace(nodes=latentspace_nodes)
        return pre_processed_probe_graphs, latentspace_nodes

    @nn.compact
    def probe_processor(self, pre_processed_probe_graphs, train=False):
        # Message pass to the probe positions
        probe_GNN = Windfarm_GNN(
            name="Windfarm_GNN_1",
            target_size=self.target_size,
            latent_size=self.latent_size,
            hidden_layer_size=self.hidden_layer_size,
            num_mlp_layers=self.num_mlp_layers,
            message_passing_steps=self.probe_message_passing_steps,
            decoder_hidden_layer_size=self.decoder_hidden_layer_size,
            num_decoder_layers=self.num_decoder_layers,
            encoder_dropout_rate=self.encoder_dropout_rate,
            processor_dropout_rate=self.processor_dropout_rate,
            layer_norm_encoder=self.layer_norm_encoder,
            layer_norm_processor=self.layer_norm_processor,
            message_norm=self.message_norm,
            layer_norm_decoder=self.layer_norm_decoder,
            res_net=self.res_net,
            RBF_encoder_kwargs=self.RBF_encoder_kwargs,
            epsilon=self.epsilon,
            encode=False,
            decode=False,
            use_lora_embedder=False,
            use_lora_processor=self.use_lora_processor,
            use_lora_decoder=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        processed_probe_nodes = probe_GNN(pre_processed_probe_graphs, train=train)
        return processed_probe_nodes

    @nn.compact
    def decoder(
        self,
        latentspace_nodes,
        processed_probe_nodes,
        wt_mask,
        probe_mask,
        probe_graphs,
        globals_context: jnp.ndarray | None = None,
    ):
        ## Decoder
        if self.decoder_hidden_layer_size is None:
            decoder_hidden_layer_size = self.hidden_layer_size
        else:
            decoder_hidden_layer_size = self.decoder_hidden_layer_size
        if self.num_decoder_layers is None:
            num_decoder_layers = self.num_mlp_layers
        else:
            num_decoder_layers = self.num_decoder_layers

        # Process global conditioning if enabled
        global_embedding = None
        if self.use_global_conditioning and globals_context is not None:
            global_context_mlp = MLP(
                name="global_context_mlp",
                feature_sizes=[self.hidden_layer_size],
                output_size=self.latent_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=False,
            )
            # Project globals to latent space
            global_embedding = global_context_mlp(globals_context)  # [batch_size, latent_size]

            # Broadcast to all nodes using JIT-compatible searchsorted
            n_node = probe_graphs.n_node
            total_nodes = processed_probe_nodes.shape[0]
            sum_n_node = jnp.cumsum(n_node)
            node_indices = jnp.arange(total_nodes)
            node_graph_idx = jnp.searchsorted(sum_n_node, node_indices, side="right")
            global_broadcast = jnp.take(global_embedding, node_graph_idx, axis=0)

        if self.decoder_strategy == "shared":
            processed_probe_nodes = latentspace_nodes * wt_mask + processed_probe_nodes * probe_mask

            # Concatenate global embedding if available
            if global_embedding is not None:
                decoder_input = jnp.concatenate([processed_probe_nodes, global_broadcast], axis=-1)
            else:
                decoder_input = processed_probe_nodes

            decoder = MLP(
                name="decoder",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
                use_lora=self.use_lora_decoder,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            delta_nodes = decoder(decoder_input)

        elif self.decoder_strategy == "separate":
            wt_decoder_input = latentspace_nodes * wt_mask
            probe_decoder_input = processed_probe_nodes * probe_mask

            # Concatenate global embedding if available
            if global_embedding is not None:
                wt_decoder_input = jnp.concatenate([wt_decoder_input, global_broadcast], axis=-1)
                probe_decoder_input = jnp.concatenate(
                    [probe_decoder_input, global_broadcast], axis=-1
                )

            decoder_wt = MLP(
                name="decoder_wt",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
                use_lora=self.use_lora_decoder,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            decoder_probe = MLP(
                name="decoder_probe",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
                use_lora=self.use_lora_decoder,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            delta_nodes_wt = decoder_wt(wt_decoder_input)
            delta_nodes_probe = decoder_probe(probe_decoder_input)
            delta_nodes = delta_nodes_wt * wt_mask + delta_nodes_probe * probe_mask

        # Initialize new_nodes to prevent UnboundLocalError when res_net=False
        new_nodes = delta_nodes

        if self.res_net:
            """This only works if the initial values are U and TI"""
            new_nodes = probe_graphs.nodes[:, 0:1] + delta_nodes

        node_mask = wt_mask + probe_mask
        new_nodes = new_nodes * node_mask
        return new_nodes
