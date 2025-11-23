"""
Windfarm_GNO_probe.py:
=================
A variation of the DeepGraphOperatorNetwork (DeepGraphONet) model by Sun et. al. [1]. Termed a Graph Neural Operator (GNO),
The GNO combines the spaces of Graph Learning and Operator Learning by combining Message Passing Graph Networks (originally based on the GraphSAGE with Graph Convolutional Network [2] but here DeeperGCN is used [3]) and DeepOperatorNetwork (DeepONet)[4]. In addtition a non-linear decoder is used inspired bu the NOn-linear MAnifold Decoder (NOMAD) [5] architecture. The GEN-block is chosen beacuse it performed best in [6].

[1] Sun, Y., Moya, C., Lin, G., & Yue, M. (2022). DeepGraphONet: A Deep Graph Operator Network to Learn and Zero-shot Transfer the Dynamic Response of Networked Systems. http://arxiv.org/abs/2209.10622
[2] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. http://arxiv.org/abs/1706.02216
[3] Li, G., Xiong, C., Thabet, A., & Ghanem, B. (2020). DeeperGCN: All You Need to Train Deeper GCNs. http://arxiv.org/abs/2006.07739
[4] Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218–229. https://doi.org/10.1038/s42256-021-00302-5
[5] Seidman, J. H., Kissas, G., Perdikaris, P., & Pappas, G. J. (2022). NOMAD: Nonlinear Manifold Decoders for Operator Learning. http://arxiv.org/abs/2206.03551
[6] de Santos, F. N., Duthé, G., Abdallah, I., Réthoré, P.-É., Weijtjens, W., Chatzi, E., & Devriendt, C. (n.d.). Multivariate prediction on wake-affected wind
turbines using graph neural networks. https://doi.org/10.3929/ethz-b-000674010

"""

import logging
from typing import Any, Dict

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
    decoder_hidden_layer_size: int = None
    num_decoder_layers: int = None
    decoder_strategy: str = "shared"  #! shared or separate
    encoder_dropout_rate: float = 0.0
    processor_dropout_rate: float = 0.0
    layer_norm_encoder: bool = False
    layer_norm_processor: bool = False
    message_norm: bool = False
    layer_norm_decoder: bool = False  #! Should always be false
    res_net: bool = False
    RBF_encoder_kwargs: Dict[str, Any] = None
    epsilon: float = 1e-6

    def setup(self):
        super().setup()

    @nn.compact
    def __call__(
        self,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask: jnp.ndarray,
        probe_mask: jnp.ndarray,
        train: bool = False,  #! True during training, only releveant for dropout
    ) -> jnp.ndarray:

        ### Pre-process
        pre_processed_graphs, pre_processed_probe_graphs = self.embedder(
            graphs, probe_graphs, train
        )

        ### Get wind turbine state with the "classic" GNN, # ? could from here to decoder be put inside a loop?
        pre_processed_probe_graphs, latentspace_nodes = self.wt_procesor(
            pre_processed_graphs, pre_processed_probe_graphs, wt_mask, probe_mask, train
        )

        # Message pass to the probe positions
        processed_probe_nodes = self.probe_procesor(pre_processed_probe_graphs, train)

        ## Decoder
        new_nodes = self.decoder(
            latentspace_nodes,
            processed_probe_nodes,
            wt_mask,
            probe_mask,
            probe_graphs,
        )
        return new_nodes

    @nn.compact
    def embedder(self, graphs, probe_graphs, train=False):
        ### Pre-process
        if self.RBF_encoder_kwargs is not None:
            RBF_encoder = RBFEncoder(**self.RBF_encoder_kwargs)

            def RBF_encode_distances(jraph_graphs):
                encoded_edges = RBF_encoder(
                    jraph_graphs.edges
                )  #! THIS ASSUMES THAT THE EDGES ARE ALL DISTANCES
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

        pre_processed_graphs = embedder(pre_processed_graphs)
        pre_processed_probe_graphs = embedder(pre_processed_probe_graphs)
        return pre_processed_graphs, pre_processed_probe_graphs

    @nn.compact
    def wt_procesor(
        self,
        pre_processed_graphs,
        pre_processed_probe_graphs,
        wt_mask,
        probe_mask,
        train=False,
    ):
        # Encode the graph(windfarm) information into a latent space
        wt_GNN = Windfarm_GNN(
            name="Windfarm_GNN_0",  #! same as automatically generated name, but specified after training to run wt_procceser separately
            target_size=None,  # Inactive when decode=False
            latent_size=self.latent_size,
            hidden_layer_size=self.hidden_layer_size,
            num_mlp_layers=self.num_mlp_layers,
            message_passing_steps=self.wt_message_passing_steps,
            decoder_hidden_layer_size=None,  # Inactive when decode=False
            num_decoder_layers=None,  # Inactive when decode=False
            encoder_dropout_rate=None,  # Inactive when encode=False
            processor_dropout_rate=self.processor_dropout_rate,
            layer_norm_encoder=None,  # Inactive when encode=False
            layer_norm_processor=self.layer_norm_processor,
            message_norm=self.message_norm,
            layer_norm_decoder=None,  # Inactive when decode=False
            res_net=None,  # Inactive when decode=False
            RBF_encoder_kwargs=None,  # Inactive when encode=False
            epsilon=self.epsilon,
            encode=False,
            decode=False,  # Option added to obtain the latentspace representation rather than the node predictions
        )

        latentspace_nodes = wt_GNN(pre_processed_graphs, train=train)
        latentspace_nodes = (
            latentspace_nodes * wt_mask + pre_processed_graphs.nodes * probe_mask
        )

        pre_processed_probe_graphs = pre_processed_probe_graphs._replace(
            nodes=latentspace_nodes
        )
        return pre_processed_probe_graphs, latentspace_nodes

    @nn.compact
    def probe_procesor(self, pre_processed_probe_graphs, train=False):

        # Message pass to the probe positions
        probe_GNN = Windfarm_GNN(
            name="Windfarm_GNN_1",  #! same as automatically generated name, but specified after training to run probe_procceser separately
            target_size=None,  # Inactive when decode=False
            latent_size=self.latent_size,
            hidden_layer_size=self.hidden_layer_size,
            num_mlp_layers=self.num_mlp_layers,
            message_passing_steps=self.probe_message_passing_steps,
            decoder_hidden_layer_size=None,  # Inactive when decode=False
            num_decoder_layers=None,  # Inactive when decode=False
            encoder_dropout_rate=None,  # Inactive when encode=False
            processor_dropout_rate=self.processor_dropout_rate,
            layer_norm_encoder=None,  # Inactive when encode=False
            layer_norm_processor=self.layer_norm_processor,
            message_norm=self.message_norm,
            layer_norm_decoder=None,  # Inactive when decode=False
            res_net=None,  # Inactive when decode=False
            RBF_encoder_kwargs=None,  # Inactive when encode=False
            epsilon=self.epsilon,
            encode=False,
            decode=False,  #
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

        if self.decoder_strategy == "shared":
            processed_probe_nodes = (
                latentspace_nodes * wt_mask + processed_probe_nodes * probe_mask
            )

            decoder = MLP(
                name="decoder",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
            )
            delta_nodes = decoder(processed_probe_nodes)

        elif self.decoder_strategy == "separate":
            wt_decoder_input = latentspace_nodes * wt_mask
            probe_decoder_input = processed_probe_nodes * probe_mask

            decoder_wt = MLP(
                name="decoder_wt",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
            )
            decoder_probe = MLP(
                name="decoder_probe",
                feature_sizes=[decoder_hidden_layer_size] * num_decoder_layers,
                output_size=self.target_size,
                activation=nn.relu,
                dropout_rate=0.0,
                layer_norm=self.layer_norm_decoder,
            )
            delta_nodes_wt = decoder_wt(wt_decoder_input)
            delta_nodes_probe = decoder_probe(probe_decoder_input)
            delta_nodes = delta_nodes_wt * wt_mask + delta_nodes_probe * probe_mask

        if self.res_net:
            """This only works if the initial values are U and TI"""
            new_nodes = (
                probe_graphs.nodes[:, 0:1] + delta_nodes
            )  # Only U because added TI flow map does not exist currently, "0:1" makes sure it is 0 but leaves the extra dimension

        node_mask = wt_mask + probe_mask  # excludes padding nodes
        new_nodes = new_nodes * node_mask
        return new_nodes
