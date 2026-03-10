"""Model loading, saving, and checkpoint management utilities."""

import logging

import jax
import orbax.checkpoint as ocp
from flax import linen as nn
from omegaconf import DictConfig

from models import Windfarm_GNN, Windfarm_GNO_probe

logger = logging.getLogger(__name__)


def get_RBF_kwargs(cfg: DictConfig):
    """Get the range for the RBF encoder"""
    if "RBF_dist_encoder" in cfg.model:
        if cfg.model.RBF_dist_encoder.type == "off":
            RBF_encoder_kwargs = None
        elif cfg.model.RBF_dist_encoder.type == "gaussian_cosine_cutoff":
            if cfg.model.RBF_dist_encoder.extrema_strategy == "pre_processed_ones":
                assert cfg.data.pre_processed is True, (
                    "Data must be pre-processed, otherwise range has to be provided"
                )
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

    # Global conditioning
    use_global_conditioning = cfg.model.get("use_global_conditioning", False)

    # Get LoRA parameters - check new finetuning.lora config first, fall back to old model.use_lora_* format
    if hasattr(cfg, "finetuning") and cfg.finetuning.get("enabled", False):
        lora_cfg = cfg.finetuning.get("lora", {})
        if lora_cfg.get("enabled", False):
            # LoRA enabled in finetuning config
            lora_rank = lora_cfg.get("rank", 8)
            lora_alpha = lora_cfg.get("alpha", 16.0)

            # Determine which components get LoRA
            # Strategy: Apply LoRA to components that are trainable (not frozen)
            # This enables parameter-efficient fine-tuning on trainable parts
            freezing_cfg = cfg.finetuning.get("freezing", {})
            frozen_components = freezing_cfg.get("frozen_components", [])

            # Apply LoRA to trainable components only
            # Note: If a component is frozen, adding LoRA to it would be wasteful
            # since the base weights won't change anyway
            use_lora_embedder = "embedder" not in frozen_components
            use_lora_processor = (
                "wt_processor" not in frozen_components
                or "probe_processor" not in frozen_components
            )
            use_lora_decoder = "decoder" not in frozen_components

            # Log LoRA configuration for debugging
            logger.info("=" * 60)
            logger.info("LoRA Configuration (from finetuning config)")
            logger.info("=" * 60)
            logger.info(f"Rank: {lora_rank}, Alpha: {lora_alpha}")
            logger.info("Applying LoRA to trainable components:")
            logger.info(f"  - Embedder: {use_lora_embedder}")
            logger.info(f"  - Processor: {use_lora_processor}")
            logger.info(f"  - Decoder: {use_lora_decoder}")
            logger.info(f"Frozen components: {frozen_components}")
            logger.info("=" * 60)
        else:
            # Finetuning enabled but LoRA disabled
            use_lora_embedder = False
            use_lora_processor = False
            use_lora_decoder = False
            lora_rank = 8
            lora_alpha = 16.0
    else:
        # Fall back to old model.use_lora_* format for backward compatibility
        use_lora_all = cfg.model.get("use_lora", False)
        use_lora_embedder = cfg.model.get("use_lora_embedder", use_lora_all)
        use_lora_processor = cfg.model.get("use_lora_processor", use_lora_all)
        use_lora_decoder = cfg.model.get("use_lora_decoder", use_lora_all)
        lora_rank = cfg.model.get("lora_rank", 8)
        lora_alpha = cfg.model.get("lora_alpha", 16.0)

    if use_global_conditioning:
        logger.info("Global conditioning enabled")

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
        use_global_conditioning=use_global_conditioning,
        use_lora_embedder=use_lora_embedder,
        use_lora_processor=use_lora_processor,
        use_lora_decoder=use_lora_decoder,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    return model


def setup_model(cfg: DictConfig) -> nn.Module:
    """Dispatcher function to setup a model based on the config"""
    if cfg.model.type in ("WindfarmGNO_probe", "WindfarmGNO_probe_v2"):
        model = setup_WindfarmGNO_probe(cfg)
    elif cfg.model.type == "WindfarmGNN":
        model = setup_WindfarmGNN(cfg)
    else:
        raise NotImplementedError(f"Model type '{cfg.model.type}' not implemented")
    return model


def _convert_jax_arrays_to_python(obj):
    """Recursively convert JAX arrays to Python scalars/lists for DictConfig compatibility."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    if isinstance(obj, jnp.ndarray | np.ndarray):
        # Move JAX arrays to host first to avoid PyWake's np.asarray monkey-patch
        if isinstance(obj, jnp.ndarray):
            obj = jax.device_get(obj)
        # Convert JAX/numpy arrays to Python types
        if obj.ndim == 0:
            # Scalar array
            item = obj.item()
            # Ensure boolean arrays become Python bools
            if isinstance(item, np.bool_):
                return bool(item)
            return item
        else:
            # Multi-dimensional array - convert to nested list
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_jax_arrays_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        converted = [_convert_jax_arrays_to_python(v) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    else:
        return obj


def load_model(path):
    """Load a model from a checkpoint

    Parameters
    ----------
    path : str
        path to checkpoint

    Returns
    -------
    Flax model, model parameters, config dict

    Notes
    -----
    This function supports Orbax checkpoint formats with cross-topology sharding.
    Checkpoints saved on GPU systems (e.g., Sophia HPC) can be loaded on CPU-only
    systems by using explicit single-device sharding.

    Supported formats:
    - Direct checkpoint with _METADATA at root
    - CheckpointManager format with _METADATA in default/ subdirectory
    """
    from pathlib import Path

    from jax.sharding import SingleDeviceSharding

    # Ensure path is a Path object and absolute (required by newer Orbax)
    checkpoint_path = Path(path).resolve()

    # Determine the actual data directory (handles CheckpointManager format)
    # CheckpointManager saves data in checkpoint/default/ subdirectory
    metadata_file = checkpoint_path / "_METADATA"
    default_metadata = checkpoint_path / "default" / "_METADATA"

    if default_metadata.exists():
        # CheckpointManager format: data is in default/ subdirectory
        data_path = checkpoint_path / "default"
    elif metadata_file.exists():
        # Direct checkpoint format: data is at checkpoint root
        data_path = checkpoint_path
    else:
        # No _METADATA found - this shouldn't happen for valid checkpoints
        raise FileNotFoundError(
            f"No _METADATA file found in {checkpoint_path} or {checkpoint_path / 'default'}"
        )

    # Use PyTreeCheckpointHandler with explicit sharding for cross-topology restore
    target_device = jax.devices()[0]
    single_device_sharding = SingleDeviceSharding(target_device)
    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True)

    # Get metadata to understand the checkpoint structure
    from etils import epath

    metadata = handler.metadata(epath.Path(data_path))

    # Create restore args with single-device sharding for all arrays
    # This handles cross-topology restore (e.g., GPU checkpoint -> CPU system)
    def make_restore_args(meta):
        """Create restore args that map any device to the local device."""
        if hasattr(meta, "sharding") or hasattr(meta, "dtype"):
            # This is an array leaf
            return ocp.ArrayRestoreArgs(sharding=single_device_sharding)
        return None

    restore_args = jax.tree.map(
        make_restore_args,
        metadata,
        is_leaf=lambda x: hasattr(x, "sharding") or hasattr(x, "dtype"),
    )

    raw_restored = handler.restore(
        epath.Path(data_path),
        args=ocp.args.PyTreeRestore(restore_args=restore_args),
    )

    # Convert JAX arrays in config to Python types for DictConfig compatibility
    config_dict = _convert_jax_arrays_to_python(raw_restored["config"])
    cfg = DictConfig(config_dict)
    metrics = raw_restored["metrics"]
    if cfg.model.type in ("WindfarmGNN",):
        model = setup_WindfarmGNN(cfg)
    elif cfg.model.type in ("WindfarmGNO_probe", "WindfarmGNO_probe_v2"):
        model = setup_WindfarmGNO_probe(cfg)
    else:
        raise NotImplementedError(f"Model type: {cfg.model.type} not implemented")

    if "opt_state" not in raw_restored["train_state"]:
        # HACK handles a case where the params are already extracted, which was a mistake in the past
        params = raw_restored["train_state"]
    else:
        params = raw_restored["train_state"]["params"]

    return model, params, metrics, cfg


def model_parameter_stats(params: dict) -> dict:
    """Get the number of parameters in a model and their shapes"""
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    detailed_params = jax.tree_util.tree_map(lambda x: (x.size, x.shape), params)
    total_params_dict = {"total_params": total_params}

    params_overview = {**total_params_dict, **detailed_params.pop("params")}
    return params_overview
