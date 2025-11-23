from typing import Dict

import jax
import orbax.checkpoint as ocp
from flax import linen as nn
from omegaconf import DictConfig

from models import Windfarm_GNN, Windfarm_GNO_probe


def get_RBF_kwargs(cfg: DictConfig):
    """Get the range for the RBF encoder"""
    if "RBF_dist_encoder" in cfg.model:
        if cfg.model.RBF_dist_encoder.type == "off":
            RBF_encoder_kwargs = None
        elif cfg.model.RBF_dist_encoder.type == "gaussian_cosine_cutoff":

            if cfg.model.RBF_dist_encoder.extrema_strategy == "pre_processed_ones":
                assert (
                    cfg.data.pre_processed == True
                ), "Data must be pre-processed, otherwise range has to be provided"
                d_min = -1
                d_max = 1
            else:
                raise NotImplementedError(
                    f"cfg.model.RBF_dist_encoder.extrema_strategy: {cfg.model.RBF_dist_encoder.extrema_strategy}, not implemented"
                )
            RBF_encoder_kwargs = {
                "num_kernels": cfg.model.RBF_dist_encoder.num_kernels,
                "d_min": d_min,
                "d_max": d_max,
                "learnable": cfg.model.RBF_dist_encoder.learnable,
            }
        else:
            raise NotImplementedError(
                f"RBF_dist_encoder: {cfg.model.RBF_dist_encoder.type}, not implemented"
            )

    return RBF_encoder_kwargs


def setup_WindfarmGNN(cfg: DictConfig) -> Windfarm_GNN:
    """Setup a WindfarmGNN model"""

    RBF_encoder_kwargs = get_RBF_kwargs(cfg)

    model = Windfarm_GNN(
        latent_size=cfg.model.latent_size,
        hidden_layer_size=cfg.model.hidden_layer_size,
        num_mlp_layers=cfg.model.num_mlp_layers,
        message_passing_steps=cfg.model.message_passing_steps,
        target_size=cfg.model.output_shape,
        decoder_hidden_layer_size=cfg.model.decoder_hidden_layer_size,
        num_decoder_layers=cfg.model.num_decoder_layers,
        encoder_dropout_rate=cfg.model.regularization.encoder_dropout_rate,
        processor_dropout_rate=cfg.model.regularization.processor_dropout_rate,
        layer_norm_encoder=cfg.model.regularization.layer_norm_encoder,
        layer_norm_processor=cfg.model.regularization.layer_norm_processor,
        message_norm=cfg.model.regularization.message_norm,
        layer_norm_decoder=cfg.model.regularization.layer_norm_decoder,  #! Should always be false
        res_net=cfg.model.res_net,
        RBF_encoder_kwargs=RBF_encoder_kwargs,
    )
    return model


def setup_WindfarmGNO_probe(cfg) -> Windfarm_GNO_probe:
    RBF_encoder_kwargs = get_RBF_kwargs(cfg)

    if "decoder_strategy" in cfg.model:
        """Handle case where decoder_strategy is not in the config"""
        decoder_strategy = cfg.model.decoder_strategy
    else:
        decoder_strategy = "shared"

    model = Windfarm_GNO_probe(
        latent_size=cfg.model.latent_size,
        hidden_layer_size=cfg.model.hidden_layer_size,
        num_mlp_layers=cfg.model.num_mlp_layers,
        wt_message_passing_steps=cfg.model.wt_message_passing_steps,
        probe_message_passing_steps=cfg.model.probe_message_passing_steps,
        target_size=cfg.model.output_shape,
        decoder_hidden_layer_size=cfg.model.decoder_hidden_layer_size,
        num_decoder_layers=cfg.model.num_decoder_layers,
        decoder_strategy=decoder_strategy,
        encoder_dropout_rate=cfg.model.regularization.encoder_dropout_rate,
        processor_dropout_rate=cfg.model.regularization.processor_dropout_rate,
        layer_norm_encoder=cfg.model.regularization.layer_norm_encoder,
        layer_norm_processor=cfg.model.regularization.layer_norm_processor,
        message_norm=cfg.model.regularization.message_norm,
        layer_norm_decoder=cfg.model.regularization.layer_norm_decoder,  #! Should always be false
        res_net=cfg.model.res_net,
        RBF_encoder_kwargs=RBF_encoder_kwargs,
    )

    return model


def setup_model(cfg: DictConfig) -> nn.Module:
    """Dispacter function to setup a model based on the config"""
    if cfg.model.type == "WindfarmGNO_probe":
        model = setup_WindfarmGNO_probe(cfg)
    elif cfg.model.type == "WindfarmGNN":
        model = setup_WindfarmGNN(cfg)
    else:
        raise NotImplementedError("Model type not implemented")
    return model


def load_model(path):
    """Load a model from a checkpoint
    Parameters
    ----------
    path : str
        path to checkpoint

    Returns
    -------
    Flax model, model parameters, config dict
    """
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    if "checkpoints" in path:
        split_path = path.split("checkpoints/")
        checkpoint_path = split_path[0] + "checkpoints"
        cp_manager = ocp.CheckpointManager(checkpoint_path, orbax_checkpointer)
        raw_restored = cp_manager.restore(split_path[1])
    else:
        raw_restored = orbax_checkpointer.restore(path)
    cfg = DictConfig(raw_restored["config"])
    metrics = raw_restored["metrics"]
    if cfg.model.type == "WindfarmGNN":
        model = setup_WindfarmGNN(cfg)
    elif cfg.model.type == "WindfarmGNO_probe":
        model = setup_WindfarmGNO_probe(cfg)
    else:
        raise NotImplementedError(f"Model type: {cfg.model_type} not implemented")

    if "opt_state" not in raw_restored["train_state"]:
        # HACK handles a case where the params are already extracted, which was a mistake in the past
        params = raw_restored["train_state"]
    else:
        params = raw_restored["train_state"]["params"]

    return model, params, metrics, cfg


def model_parameter_stats(params: Dict) -> Dict:
    """Get the number of parameters in a model and their shapes"""
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    detailed_params = jax.tree_util.tree_map(lambda x: (x.size, x.shape), params)
    total_params_dict = {"total_params": total_params}

    params_overview = {**total_params_dict, **detailed_params.pop("params")}
    return params_overview
