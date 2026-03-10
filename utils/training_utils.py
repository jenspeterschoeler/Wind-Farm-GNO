"""
Training Utilities

Helper functions for training workflow, extracted from the monolithic
train_GNO_probe function to improve modularity and testability.
"""

import logging
import os
from typing import Any

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import orbax.checkpoint as ocp
from flax import struct
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import wandb
from utils import add_to_hydra_cfg, convert_to_wandb_format, setup_optimizer
from utils.data_tools import (
    setup_refresh_iterator,
    setup_test_val_iterator,
    setup_train_dataset,
)
from utils.GNO_probe import initialize_GNO_probe, inverse_scale_rel_ws, scale_rel_ws
from utils.model_tools import model_parameter_stats, setup_model
from utils.plotting import plot_probe_graph_fn

logger = logging.getLogger(__name__)


def merge_pretrained_params(initialized_params, pretrained_params, cfg):
    """
    Merge pretrained parameters into initialized parameters.

    Copies pretrained weights into base model while keeping newly initialized
    LoRA parameters untouched. This is necessary when:
    1. Model is initialized with LoRA layers (creates all params including LoRA)
    2. We want to load pretrained base weights
    3. We want to keep LoRA params randomly initialized for training

    Args:
        initialized_params: Freshly initialized params (has LoRA if enabled)
        pretrained_params: Pretrained params (no LoRA)
        cfg: Configuration with fine-tuning settings

    Returns:
        Merged parameters with pretrained base weights and fresh LoRA params
    """
    import jax

    # Check if LoRA is enabled
    has_lora = (
        hasattr(cfg, "finetuning")
        and cfg.finetuning.get("enabled", False)
        and cfg.finetuning.get("lora", {}).get("enabled", False)
    )

    if not has_lora:
        # No LoRA, just use pretrained params directly
        return pretrained_params

    # LoRA enabled: need to merge carefully
    logger.info("Merging pretrained params with LoRA structure:")
    logger.info(f"  Initialized params keys: {list(initialized_params['params'].keys())}")
    logger.info(f"  Pretrained params keys: {list(pretrained_params['params'].keys())}")

    # Deep copy initialized params structure
    merged_params = jax.tree_util.tree_map(lambda x: x, initialized_params)

    # Copy pretrained weights into base model parameters
    # We need to handle LoRA specially:
    # 1. Skip lora_a, lora_b params (keep initialized)
    # 2. Handle LoRADense_* -> Dense_* key mapping
    def copy_pretrained_recursive(init_tree, pretrain_tree, path_prefix=""):
        """Recursively copy pretrained params, skipping LoRA params."""
        if isinstance(init_tree, dict) and isinstance(pretrain_tree, dict):
            result = {}
            for key in init_tree:
                current_path = f"{path_prefix}/{key}" if path_prefix else key

                # Check if this is a LoRA parameter (lowercase lora_a, lora_b)
                if "lora_a" in key or "lora_b" in key:
                    logger.debug(f"  Keeping initialized LoRA param: {current_path}")
                    result[key] = init_tree[key]
                elif key in pretrain_tree:
                    # Direct match - recursively copy this branch
                    result[key] = copy_pretrained_recursive(
                        init_tree[key], pretrain_tree[key], current_path
                    )
                else:
                    # No direct match - try LoRADense_* -> Dense_* mapping
                    pretrain_key = (
                        key.replace("LoRADense_", "Dense_")
                        if key.startswith("LoRADense_")
                        else None
                    )
                    if pretrain_key and pretrain_key in pretrain_tree:
                        logger.debug(f"  Mapping {key} -> {pretrain_key} from pretrained")
                        result[key] = copy_pretrained_recursive(
                            init_tree[key], pretrain_tree[pretrain_key], current_path
                        )
                    else:
                        # Key exists in init but not pretrain (e.g., new LoRA-only params)
                        logger.debug(
                            f"  Keeping initialized param (not in pretrained): {current_path}"
                        )
                        result[key] = init_tree[key]
            return result
        else:
            # Leaf node - copy pretrained value
            return pretrain_tree

    merged_params["params"] = copy_pretrained_recursive(
        initialized_params["params"], pretrained_params["params"]
    )

    logger.info("Successfully merged pretrained weights with LoRA parameters!")

    return merged_params


class TrainStateWithPretrained(TrainState):
    """Extended TrainState for fine-tuning with pretrained weights.

    Attributes:
        pretrained_params: Pretrained model parameters
        finetuning_config: Fine-tuning configuration dict
    """

    pretrained_params: dict | None = struct.field(pytree_node=False, default=None)
    finetuning_config: dict | None = struct.field(pytree_node=False, default=None)


def setup_data_loaders(cfg: DictConfig):
    """
    Setup training and validation data loaders.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (train_dataset, get_refreshed_train_fn, get_refreshed_val_fn, cfg_updated)
    """
    train_dataset, cfg = setup_train_dataset(cfg)
    get_refreshed_train_fn, unpadded_train_iterator = setup_refresh_iterator(cfg, train_dataset)
    get_refreshed_val_fn, val_dataset, _, _ = setup_test_val_iterator(
        cfg, type_str="val", return_positions=True
    )
    return train_dataset, get_refreshed_train_fn, get_refreshed_val_fn, cfg


def initialize_model_and_params(cfg: DictConfig, first_batch, pretrained_checkpoint_path=None):
    """
    Initialize model and parameters from first batch, optionally loading from pretrained checkpoint.

    Args:
        cfg: Configuration object
        first_batch: Tuple of (graphs, probe_graphs, node_array_tuple)
        pretrained_checkpoint_path: Optional path to pretrained checkpoint for transfer learning

    Returns:
        Tuple of (model, params, dropout_active, rng_key, cfg_updated)
    """
    from utils.model_tools import load_model

    graphs, probe_graphs, node_array_tuple = first_batch
    targets, wt_mask, probe_mask = node_array_tuple

    rng_key = jax.random.PRNGKey(0)

    # Load pretrained architecture and weights if checkpoint path is provided
    if pretrained_checkpoint_path is not None:
        logger.info("=" * 80)
        logger.info("TRANSFER LEARNING MODE")
        logger.info("=" * 80)
        logger.info(f"Loading pretrained model from: {pretrained_checkpoint_path}")

        _, pretrained_params, pretrained_metrics, pretrained_cfg = load_model(
            pretrained_checkpoint_path
        )

        logger.info(
            f"Pretrained model validation loss: {pretrained_metrics.get('val_loss', 'N/A')}"
        )

        # Use pretrained model architecture (override current cfg.model with pretrained architecture)
        logger.info("Using pretrained model architecture:")
        logger.info(f"  Model type: {pretrained_cfg.model.type}")
        logger.info(f"  Latent size: {pretrained_cfg.model.latent_size}")
        logger.info(f"  Hidden layer size: {pretrained_cfg.model.hidden_layer_size}")
        logger.info(f"  Num MLP layers: {pretrained_cfg.model.num_mlp_layers}")

        # IMPORTANT: Save finetuning config before overriding model
        # This preserves LoRA and other fine-tuning settings
        finetuning_cfg = cfg.get("finetuning", None)

        # Override current model config with pretrained architecture
        cfg.model = pretrained_cfg.model

        # Restore finetuning config (it gets lost when we override cfg.model)
        if finetuning_cfg is not None:
            cfg = add_to_hydra_cfg(cfg, "finetuning", finetuning_cfg)
            logger.info("Fine-tuning config preserved:")
            if finetuning_cfg.get("lora", {}).get("enabled", False):
                logger.info(
                    f"  LoRA enabled: rank={finetuning_cfg.lora.rank}, alpha={finetuning_cfg.lora.alpha}"
                )
            if finetuning_cfg.get("freezing", {}).get("enabled", False):
                logger.info(f"  Freezing: {finetuning_cfg.freezing.get('frozen_components', [])}")

        # Set output shape for current task
        cfg.model = add_to_hydra_cfg(
            cfg.model,
            "output_shape",
            targets.shape[-1],
        )

        # Initialize model with pretrained architecture
        model = setup_model(cfg)

        # Initialize model parameters (creates LoRA layers if enabled)
        logger.info("Initializing model (creates LoRA layers if enabled)...")
        params, dropout_active = initialize_GNO_probe(
            cfg,
            model,
            rng_key,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
        )

        # Now copy pretrained weights into base model (keeps LoRA params untouched)
        logger.info("Copying pretrained weights into base model...")
        params = merge_pretrained_params(params, pretrained_params, cfg)

        logger.info("Pretrained model architecture and weights loaded successfully!")
        logger.info("=" * 80)
    else:
        # Standard initialization from scratch
        cfg.model = add_to_hydra_cfg(
            cfg.model,
            "output_shape",
            targets.shape[-1],
        )
        model = setup_model(cfg)

        params, dropout_active = initialize_GNO_probe(
            cfg,
            model,
            rng_key,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
        )

    return model, params, dropout_active, rng_key, cfg


def log_model_parameters(params, wandb_run=None):
    """
    Log model parameter statistics.

    Args:
        params: Model parameters
        wandb_run: Optional W&B run object for logging
    """
    params_stats = model_parameter_stats(params)
    logger.info(f"Total parameters: {params_stats['total_params']}")

    if wandb_run is not None:
        wandb_params_stats = convert_to_wandb_format(params_stats)
        for dict_key, value in wandb_params_stats.items():
            wandb.summary[f"params/{dict_key}"] = value


def setup_training_components(
    cfg: DictConfig,
    model,
    params,
    dropout_active,
    pretrained_params: dict | None = None,
):
    """
    Setup optimizer, early stopping, and train state.

    Args:
        cfg: Configuration object
        model: Initialized model
        params: Model parameters
        dropout_active: Whether dropout is active
        pretrained_params: Optional pretrained parameters (for fine-tuning)

    Returns:
        Tuple of (train_state, optimizer, early_stop)
    """
    # Build optimizer (fine-tuning aware if enabled)
    if cfg.get("finetuning", {}).get("enabled", False):
        from utils.finetuning import build_finetuning_optimizer

        optimizer = build_finetuning_optimizer(cfg, params, pretrained_params)
    else:
        optimizer = setup_optimizer(cfg)

    early_stop = None
    if "early_stop" in cfg.optimizer:
        early_stop = EarlyStopping(
            min_delta=cfg.optimizer.early_stop.criteria,
            patience=int(
                cfg.optimizer.early_stop.patience / cfg.optimizer.validation.rate_of_validation
            ),
        )

    # Create train state (extended if fine-tuning)
    is_finetuning = cfg.get("finetuning", {}).get("enabled", False)

    if dropout_active:

        def apply_fn(params, graphs, probe_graphs, wt_mask, probe_mask, rngs):
            return model.apply(
                params,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                train=True,
                rngs=rngs,
            )

    else:

        def apply_fn(params, graphs, probe_graphs, wt_mask, probe_mask):
            return model.apply(
                params,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                train=True,
            )

    if is_finetuning:
        # Use extended train state with pretrained params
        train_state = TrainStateWithPretrained.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer,
            pretrained_params=pretrained_params,
            finetuning_config=OmegaConf.to_container(cfg.finetuning, resolve=True),
        )
    else:
        # Standard train state
        train_state = TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer,
        )

    return train_state, optimizer, early_stop


def create_prediction_fn(model):
    """
    Create JIT-compiled prediction function.

    Args:
        model: Model to use for predictions

    Returns:
        JIT-compiled prediction function
    """
    return jax.jit(
        lambda params, graphs, probe_graphs, wt_mask, probe_mask: model.apply(
            params,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
            train=False,
        )
    )


def create_train_step_fn(cfg: DictConfig, dropout_active):
    """
    Create JIT-compiled training step function.

    Args:
        cfg: Configuration object
        dropout_active: Whether dropout is active

    Returns:
        JIT-compiled train step function
    """

    @jax.jit
    def train_step_fn(
        train_state: TrainState,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask: jnp.ndarray,
        probe_mask: jnp.ndarray,
        targets: jnp.ndarray,
        rngs: dict | None = None,
    ) -> tuple[Any, TrainState, jnp.ndarray]:
        if cfg.model.scale_rel_ws:
            combined_mask = wt_mask + probe_mask
            targets = scale_rel_ws(graphs, targets, combined_mask)

        # Check for specialized loss configuration
        loss_cfg = cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "mse")

        if dropout_active:

            def loss_fn(params):
                prediction = train_state.apply_fn(
                    params,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                    rngs=rngs,
                )
                # Task loss based on loss type
                combined_mask = wt_mask + probe_mask

                if loss_type == "wake_aware_mse":
                    from utils.finetuning.wake_loss import wake_aware_mse_loss

                    task_loss = wake_aware_mse_loss(
                        prediction,
                        targets,
                        combined_mask,
                        deficit_threshold=loss_cfg.get("deficit_threshold", 0.1),
                        alpha_under=loss_cfg.get("alpha_under", 0.7),
                        alpha_over=loss_cfg.get("alpha_over", 0.3),
                        wake_weight=loss_cfg.get("wake_weight", 2.0),
                    )
                elif loss_type == "far_wake_weighted":
                    from utils.finetuning.wake_loss import far_wake_weighted_loss

                    task_loss = far_wake_weighted_loss(
                        prediction,
                        targets,
                        combined_mask,
                        strong_wake_threshold=loss_cfg.get("strong_wake_threshold", 0.15),
                        weak_wake_threshold=loss_cfg.get("weak_wake_threshold", 0.05),
                        far_wake_weight=loss_cfg.get("far_wake_weight", 2.0),
                    )
                elif loss_type == "stratified_wake":
                    from utils.finetuning.wake_loss import stratified_wake_loss

                    task_loss = stratified_wake_loss(
                        prediction,
                        targets,
                        combined_mask,
                        strong_wake_threshold=loss_cfg.get("strong_wake_threshold", 0.15),
                        weak_wake_threshold=loss_cfg.get("weak_wake_threshold", 0.05),
                        strong_wake_weight=loss_cfg.get("strong_wake_weight", 1.0),
                        moderate_wake_weight=loss_cfg.get("moderate_wake_weight", 1.5),
                        far_wake_weight=loss_cfg.get("far_wake_weight", 2.0),
                    )
                else:
                    # Default: standard MSE
                    task_loss = jnp.mean((targets - prediction) ** 2)

                return task_loss, prediction

        else:

            def loss_fn(params):
                prediction = train_state.apply_fn(
                    params,
                    graphs,
                    probe_graphs,
                    wt_mask,
                    probe_mask,
                )
                # Task loss based on loss type
                combined_mask = wt_mask + probe_mask

                if loss_type == "wake_aware_mse":
                    from utils.finetuning.wake_loss import wake_aware_mse_loss

                    task_loss = wake_aware_mse_loss(
                        prediction,
                        targets,
                        combined_mask,
                        deficit_threshold=loss_cfg.get("deficit_threshold", 0.1),
                        alpha_under=loss_cfg.get("alpha_under", 0.7),
                        alpha_over=loss_cfg.get("alpha_over", 0.3),
                        wake_weight=loss_cfg.get("wake_weight", 2.0),
                    )
                elif loss_type == "far_wake_weighted":
                    from utils.finetuning.wake_loss import far_wake_weighted_loss

                    task_loss = far_wake_weighted_loss(
                        prediction,
                        targets,
                        combined_mask,
                        strong_wake_threshold=loss_cfg.get("strong_wake_threshold", 0.15),
                        weak_wake_threshold=loss_cfg.get("weak_wake_threshold", 0.05),
                        far_wake_weight=loss_cfg.get("far_wake_weight", 2.0),
                    )
                elif loss_type == "stratified_wake":
                    from utils.finetuning.wake_loss import stratified_wake_loss

                    task_loss = stratified_wake_loss(
                        prediction,
                        targets,
                        combined_mask,
                        strong_wake_threshold=loss_cfg.get("strong_wake_threshold", 0.15),
                        weak_wake_threshold=loss_cfg.get("weak_wake_threshold", 0.05),
                        strong_wake_weight=loss_cfg.get("strong_wake_weight", 1.0),
                        moderate_wake_weight=loss_cfg.get("moderate_wake_weight", 1.5),
                        far_wake_weight=loss_cfg.get("far_wake_weight", 2.0),
                    )
                else:
                    # Default: standard MSE
                    task_loss = jnp.mean((targets - prediction) ** 2)

                return task_loss, prediction

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, prediction), grads = grad_fn(train_state.params)

        if cfg.model.scale_rel_ws:
            combined_mask = wt_mask + probe_mask
            prediction = inverse_scale_rel_ws(graphs, prediction, combined_mask)

        new_train_state = train_state.apply_gradients(grads=grads)
        return loss, new_train_state, prediction

    return train_step_fn


def create_val_errors_fn(cfg: DictConfig, prediction_fn):
    """
    Create JIT-compiled validation errors function.

    Args:
        cfg: Configuration object
        prediction_fn: Prediction function

    Returns:
        JIT-compiled validation errors function
    """

    @jax.jit
    def val_errors_fn(
        train_state: TrainState,
        graphs: jraph.GraphsTuple,
        probe_graphs: jraph.GraphsTuple,
        wt_mask,
        probe_mask,
        targets: jnp.ndarray,
    ):
        """This function assumes the graphs are padded"""
        predictions = prediction_fn(
            train_state.params,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
        )
        if cfg.model.scale_rel_ws:
            combined_mask = wt_mask + probe_mask
            predictions_scaled_rel_ws = predictions
            targets_scaled_rel_ws = scale_rel_ws(graphs, targets, combined_mask)
            errors_scaled_rel_ws = targets_scaled_rel_ws - predictions_scaled_rel_ws
            sq_errors_scaled_rel_ws = errors_scaled_rel_ws**2
            abs_errors_scaled_rel_ws = jnp.abs(errors_scaled_rel_ws)
            predictions = inverse_scale_rel_ws(graphs, predictions, combined_mask)

        errors = targets - predictions
        sq_errors = errors**2
        abs_errors = jnp.abs(errors)

        n_samples = jnp.sum(wt_mask) + jnp.sum(probe_mask)

        if cfg.model.scale_rel_ws:
            return (
                jnp.sum(errors_scaled_rel_ws),
                jnp.sum(sq_errors_scaled_rel_ws),
                jnp.sum(abs_errors_scaled_rel_ws),
                jnp.sum(errors),
                jnp.sum(sq_errors),
                jnp.sum(abs_errors),
                n_samples,
            )
        else:
            return (
                jnp.sum(errors),
                jnp.sum(sq_errors),
                jnp.sum(abs_errors),
                n_samples,
            )

    return val_errors_fn


def _get_layout_hash(positions, wt_mask):
    """Create hash to identify unique layouts by turbine positions.

    Samples sharing the same physical layout have identical turbine positions.
    This function creates a deterministic hash based on sorted WT positions.

    Args:
        positions: Node positions array (n_nodes, 2)
        wt_mask: Wind turbine mask (n_nodes, 1) - 1 for WT nodes, 0 otherwise

    Returns:
        Hash value identifying the layout
    """
    wt_positions = positions[wt_mask.flatten() == 1]
    # Sort to ensure consistent ordering regardless of node order
    sorted_pos = wt_positions[np.lexsort((wt_positions[:, 1], wt_positions[:, 0]))]
    # Round to avoid floating point precision issues
    sorted_pos_rounded = np.round(sorted_pos, decimals=2)
    return hash(sorted_pos_rounded.tobytes())


def load_plot_components(cfg: DictConfig, get_refreshed_val_fn=None):
    """
    Load plotting components from validation data.

    Prioritizes finding a 'cluster' layout with 120+ turbines, then collects
    4 different inflow conditions for that same layout. This ensures the
    4-panel plot shows the same wind farm under varying conditions.

    Selection strategy (in priority order):
    1. Cluster layout, 120+ turbines, 4 different inflow conditions
    2. Cluster layout, 120+ turbines, fewer inflow conditions (use what's available)
    3. Any cluster layout with 4 inflow conditions
    4. First 4 valid samples (original behavior)

    Args:
        cfg: Configuration object
        get_refreshed_val_fn: Validation data iterator function

    Returns:
        List of dicts with plot components (one per sample), or None if unavailable
    """
    # Construct from validation data
    if get_refreshed_val_fn is not None:
        logger.info(
            "Constructing plot components from validation data with intelligent layout selection"
        )
        try:
            # Try to create iterator with layout info for intelligent selection
            # Some datasets (e.g., AWF) don't have layout metadata, so fall back gracefully
            has_layout_info = True
            try:
                plot_iterator_fn, _, _, _ = setup_test_val_iterator(
                    cfg,
                    type_str="val",
                    return_positions=True,
                    return_layout_info=True,
                )
                plot_iterator = plot_iterator_fn()
                # Test if first batch has layout info by peeking
                first_batch = next(iter(plot_iterator_fn()))
                if len(first_batch) != 5:
                    raise AttributeError("Batch doesn't have layout info")
            except (AttributeError, TypeError) as e:
                logger.info(f"Dataset doesn't have layout info ({e}), using fallback selection")
                has_layout_info = False
                plot_iterator_fn, _, _, _ = setup_test_val_iterator(
                    cfg,
                    type_str="val",
                    return_positions=True,
                    return_layout_info=False,
                )
                plot_iterator = plot_iterator_fn()

            # Collect all samples and group by layout
            layout_samples = {}  # layout_hash -> list of samples
            all_samples = []  # Fallback: all valid samples in order

            for batch in plot_iterator:
                # Unpack batch - format depends on whether layout info is available
                if has_layout_info:
                    val_graphs, val_probe_graphs, val_node_arrays, layout_type, wt_spacing = batch
                else:
                    val_graphs, val_probe_graphs, val_node_arrays = batch
                    layout_type = None

                # Extract node arrays (targets, wt_mask, probe_mask, positions)
                (
                    all_targets,
                    all_wt_mask,
                    all_probe_mask,
                    all_positions,
                ) = val_node_arrays

                # Extract each graph from this batch
                n_graphs_in_batch = len(val_probe_graphs.n_node) - 1  # -1 for padding graph

                node_offset_main = 0
                edge_offset_main = 0
                node_offset_probe = 0
                edge_offset_probe = 0

                for i in range(n_graphs_in_batch):
                    n_nodes_main = int(val_graphs.n_node[i])
                    n_edges_main = int(val_graphs.n_edge[i])
                    n_nodes_probe = int(val_probe_graphs.n_node[i])
                    n_edges_probe = int(val_probe_graphs.n_edge[i])

                    # Skip padding graphs (n_node == 0)
                    if n_nodes_main == 0 or n_nodes_probe == 0:
                        node_offset_main += n_nodes_main
                        edge_offset_main += n_edges_main
                        node_offset_probe += n_nodes_probe
                        edge_offset_probe += n_edges_probe
                        continue

                    # Extract this sample's graph
                    sample_graph = val_graphs._replace(
                        nodes=val_graphs.nodes[node_offset_main : node_offset_main + n_nodes_main],
                        edges=val_graphs.edges[edge_offset_main : edge_offset_main + n_edges_main],
                        senders=val_graphs.senders[
                            edge_offset_main : edge_offset_main + n_edges_main
                        ]
                        - node_offset_main,
                        receivers=val_graphs.receivers[
                            edge_offset_main : edge_offset_main + n_edges_main
                        ]
                        - node_offset_main,
                        globals=val_graphs.globals[i : i + 1],
                        n_node=val_graphs.n_node[i : i + 1],
                        n_edge=val_graphs.n_edge[i : i + 1],
                    )
                    sample_probe_graph = val_probe_graphs._replace(
                        nodes=val_probe_graphs.nodes[
                            node_offset_probe : node_offset_probe + n_nodes_probe
                        ],
                        edges=val_probe_graphs.edges[
                            edge_offset_probe : edge_offset_probe + n_edges_probe
                        ],
                        senders=val_probe_graphs.senders[
                            edge_offset_probe : edge_offset_probe + n_edges_probe
                        ]
                        - node_offset_probe,
                        receivers=val_probe_graphs.receivers[
                            edge_offset_probe : edge_offset_probe + n_edges_probe
                        ]
                        - node_offset_probe,
                        globals=val_probe_graphs.globals[i : i + 1],
                        n_node=val_probe_graphs.n_node[i : i + 1],
                        n_edge=val_probe_graphs.n_edge[i : i + 1],
                    )

                    # Extract node arrays for this sample
                    sample_positions = all_positions[
                        node_offset_probe : node_offset_probe + n_nodes_probe
                    ]
                    sample_targets_scaled = all_targets[
                        node_offset_probe : node_offset_probe + n_nodes_probe
                    ]
                    sample_wt_mask = all_wt_mask[
                        node_offset_probe : node_offset_probe + n_nodes_probe
                    ]
                    sample_probe_mask = all_probe_mask[
                        node_offset_probe : node_offset_probe + n_nodes_probe
                    ]

                    # Unscale targets
                    sample_targets = (
                        sample_targets_scaled * cfg.data.scale_stats["velocity"]["range"][0]
                    )

                    # Extract globals (wind speed and TI)
                    sample_U = np.array(
                        sample_graph.globals[0, 0].squeeze()
                        * cfg.data.scale_stats["velocity"]["range"][0]
                    )
                    sample_TI = np.array(
                        sample_graph.globals[0, 1].squeeze()
                        * cfg.data.scale_stats["ti"]["range"][0]
                    )

                    # Filter to probe nodes only
                    probe_mask_flat = sample_probe_mask.flatten() == 1
                    sample_targets_filtered = sample_targets[probe_mask_flat]

                    # Calculate number of wind turbines
                    n_wt = int(np.sum(sample_wt_mask))

                    # Get layout type for this sample (handle both list and single value)
                    # layout_type may be None if dataset doesn't have layout metadata
                    if layout_type is not None:
                        sample_layout_type = (
                            layout_type[i] if isinstance(layout_type, list) else layout_type
                        )
                    else:
                        sample_layout_type = "unknown"

                    sample_dict = {
                        "graph": sample_graph,
                        "probe_graph": sample_probe_graph,
                        "positions": sample_positions,
                        "targets": sample_targets_filtered,
                        "wt_mask": sample_wt_mask,
                        "probe_mask": sample_probe_mask,
                        "U": sample_U,
                        "TI": sample_TI,
                        "layout_type": sample_layout_type,
                        "n_wt": n_wt,
                    }

                    all_samples.append(sample_dict)

                    # Group by layout using position hash
                    layout_hash = _get_layout_hash(sample_positions, sample_wt_mask)
                    if layout_hash not in layout_samples:
                        layout_samples[layout_hash] = []
                    layout_samples[layout_hash].append(sample_dict)

                    # Update offsets
                    node_offset_main += n_nodes_main
                    edge_offset_main += n_edges_main
                    node_offset_probe += n_nodes_probe
                    edge_offset_probe += n_edges_probe

            if len(all_samples) == 0:
                logger.warning("No valid samples found in validation data")
                return None

            # Selection strategy: find best layout meeting criteria
            selected_samples = None

            # Priority 1: Cluster with 120+ turbines and 4+ inflow conditions
            for _layout_hash, samples in layout_samples.items():
                if (
                    samples[0]["layout_type"] == "cluster"
                    and samples[0]["n_wt"] >= 120
                    and len(samples) >= 4
                ):
                    selected_samples = samples[:4]
                    logger.info(
                        f"Selected cluster layout with {samples[0]['n_wt']} turbines, "
                        f"{len(samples)} inflow conditions available"
                    )
                    break

            # Priority 2: Cluster with 120+ turbines (fewer inflows OK)
            if selected_samples is None:
                for _layout_hash, samples in layout_samples.items():
                    if samples[0]["layout_type"] == "cluster" and samples[0]["n_wt"] >= 120:
                        selected_samples = samples[:4]
                        logger.info(
                            f"Selected cluster layout with {samples[0]['n_wt']} turbines, "
                            f"{len(samples)} inflow conditions (< 4 available)"
                        )
                        break

            # Priority 3: Any cluster with 4+ inflow conditions
            if selected_samples is None:
                for _layout_hash, samples in layout_samples.items():
                    if samples[0]["layout_type"] == "cluster" and len(samples) >= 4:
                        selected_samples = samples[:4]
                        logger.info(
                            f"Selected cluster layout with {samples[0]['n_wt']} turbines "
                            f"(< 120), {len(samples)} inflow conditions"
                        )
                        break

            # Priority 4: Any layout with 100+ turbines (for datasets without layout metadata)
            if selected_samples is None:
                for _layout_hash, samples in layout_samples.items():
                    if samples[0]["n_wt"] >= 100:
                        selected_samples = samples[:4]
                        logger.info(
                            f"Selected layout with {samples[0]['n_wt']} turbines "
                            f"(no cluster metadata), {len(samples)} inflow conditions"
                        )
                        break

            # Priority 5: Fallback to first 4 samples
            if selected_samples is None:
                selected_samples = all_samples[:4]
                logger.info("Using fallback: first 4 valid samples (no cluster layout found)")

            logger.info(f"Constructed plot components: {len(selected_samples)} samples")
            for i, s in enumerate(selected_samples):
                layout_info = f" [{s['layout_type']}, n={s['n_wt']}]" if "layout_type" in s else ""
                logger.info(f"  Sample {i + 1}: U={s['U']:.2f}, TI={s['TI']:.4f}{layout_info}")

            return selected_samples

        except Exception as e:
            logger.warning(f"Failed to construct plot components from validation data: {e}")
            import traceback

            traceback.print_exc()
            return None

    else:
        logger.warning(
            "Plot components unavailable: no pickle file and no validation data provided. "
            "Skipping WandB visualization plots."
        )
        return None


def setup_checkpointing(cfg: DictConfig):
    """
    Setup checkpoint manager for model saving.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (checkpoint_manager, orbax_checkpointer)
    """
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_dir = os.path.join(cfg.model_save_path, "checkpoints")
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)
    return checkpoint_manager, orbax_checkpointer


def setup_multi_metric_checkpointing(cfg: DictConfig):
    """
    Setup checkpoint managers for multiple metrics (MSE, MAE, hybrid).

    Creates separate checkpoint directories for:
    - best_mse: Best model by MSE (original behavior)
    - best_mae: Best model by MAE
    - best_hybrid: Best model by geometric mean of normalized MSE and MAE

    Args:
        cfg: Configuration object

    Returns:
        Dict with checkpoint managers for each metric type and orbax_checkpointer
    """
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    orbax_checkpointer = ocp.PyTreeCheckpointer()

    checkpoint_managers = {}
    for metric_type in ["best_mse", "best_mae", "best_hybrid"]:
        checkpoint_dir = os.path.join(cfg.model_save_path, f"checkpoints_{metric_type}")
        checkpoint_managers[metric_type] = ocp.CheckpointManager(
            checkpoint_dir, orbax_checkpointer, options
        )

    return checkpoint_managers, orbax_checkpointer


def setup_periodic_checkpointing(cfg: DictConfig, interval: int = 10):
    """
    Setup periodic checkpoint manager for weights-only snapshots.

    Creates a checkpoint directory for saving model weights (no optimizer state)
    at regular intervals. These checkpoints are useful for:
    - Post-hoc analysis of training dynamics
    - Recovery to any training iteration
    - Fine-tuning from intermediate states

    Unlike best-metric checkpoints, ALL periodic checkpoints are kept.

    Args:
        cfg: Configuration object
        interval: Save checkpoint every N epochs (default: 10)

    Returns:
        Tuple of (checkpoint_manager, orbax_checkpointer, interval)
    """
    # Keep all periodic checkpoints (no max_to_keep limit)
    options = ocp.CheckpointManagerOptions(max_to_keep=None, create=True)
    checkpoint_dir = os.path.join(cfg.model_save_path, "checkpoints_periodic")
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)
    return checkpoint_manager, orbax_checkpointer, interval


def save_periodic_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    train_state,
    epoch: int,
    metrics: dict,
    cfg: DictConfig,
) -> None:
    """
    Save weights-only periodic checkpoint.

    Saves only model parameters (no optimizer state) to minimize storage.
    Each checkpoint is ~5MB vs ~15MB for full state.

    Args:
        checkpoint_manager: Periodic checkpoint manager
        train_state: Current training state
        epoch: Current epoch number
        metrics: Current training metrics
        cfg: Configuration object
    """
    from omegaconf import OmegaConf

    # Extract only params (weights), not full train_state
    ckpt = {
        "params": train_state.params,
        "config": OmegaConf.to_container(cfg),
        "metrics": {
            "epoch": epoch,
            "val_mse": metrics.get("val_mse"),
            "val_mae": metrics.get("val_mae"),
        },
    }

    checkpoint_manager.save(epoch, ckpt)
    logger.debug(f"Saved periodic checkpoint at epoch {epoch}")


def compute_hybrid_metric(
    mse: float, mae: float, mse_baseline: float, mae_baseline: float
) -> float:
    """
    Compute hybrid metric as geometric mean of normalized MSE and MAE.

    The hybrid metric balances both MSE and MAE by:
    1. Normalizing each to [0, 1] using initial baseline values
    2. Computing geometric mean to ensure both metrics must improve

    Args:
        mse: Current MSE value
        mae: Current MAE value
        mse_baseline: Initial MSE baseline for normalization
        mae_baseline: Initial MAE baseline for normalization

    Returns:
        Hybrid metric (lower is better)
    """
    # Normalize by baseline (first validation values)
    # Clamp to avoid division by zero
    mse_norm = mse / max(mse_baseline, 1e-10)
    mae_norm = mae / max(mae_baseline, 1e-10)

    # Geometric mean ensures both must improve
    import math

    return math.sqrt(mse_norm * mae_norm)


def save_final_model(cfg: DictConfig, train_state, metrics, epoch, orbax_checkpointer):
    """
    Save final model checkpoint.

    Args:
        cfg: Configuration object
        train_state: Final training state
        metrics: Training metrics
        epoch: Final epoch number
        orbax_checkpointer: Orbax checkpointer instance

    Returns:
        Updated configuration with final model path
    """
    cfg = add_to_hydra_cfg(
        cfg, "final_model_path", os.path.join(cfg.model_save_path, f"final_e_{epoch}")
    )

    ckpt = {
        "train_state": train_state,
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
    }

    orbax_checkpointer.save(
        os.path.join(cfg.model_save_path, f"final_e_{epoch}"),
        ckpt,
        force=True,
    )

    return cfg


def run_training_epoch(
    train_state,
    get_refreshed_train_fn,
    train_step_fn,
    dropout_active,
    rng_key=None,
):
    """
    Run single training epoch.

    Args:
        train_state: Current training state
        get_refreshed_train_fn: Function to get fresh training iterator
        train_step_fn: Training step function
        dropout_active: Whether dropout is active
        rng_key: Random key for dropout

    Returns:
        Tuple of (train_loss, train_state, rng_key)
    """
    train_loss = 0
    train_iterator = get_refreshed_train_fn()
    num_batches = 0

    for graphs, probe_graphs, node_array_tuple in train_iterator:
        targets, wt_mask, probe_mask = node_array_tuple

        if dropout_active:
            rng_key, params_key, dropout_key = jax.random.split(rng_key, 3)  # type: ignore[arg-type]
            batch_loss, train_state, predictions = train_step_fn(
                train_state,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                targets,
                rngs={"params": params_key, "dropout": dropout_key},
            )
        else:
            batch_loss, train_state, predictions = train_step_fn(
                train_state,
                graphs,
                probe_graphs,
                wt_mask,
                probe_mask,
                targets,
            )

        train_loss += batch_loss
        num_batches += 1

    avg_train_loss = train_loss / num_batches
    return avg_train_loss, train_state, rng_key


def run_validation(
    cfg,
    train_state,
    get_refreshed_val_fn,
    val_errors_fn,
):
    """
    Run validation loop.

    Args:
        cfg: Configuration object
        train_state: Current training state
        get_refreshed_val_fn: Function to get validation iterator
        val_errors_fn: Validation errors function

    Returns:
        Dictionary with validation metrics
    """
    val_iterator = get_refreshed_val_fn()

    errors = np.float64(0)
    sq_errors = np.float64(0)
    abs_errors = np.float64(0)
    n_samples = np.int64(0)

    if cfg.model.scale_rel_ws:
        errors_scaled_rel_ws = np.float64(0)
        sq_errors_scaled_rel_ws = np.float64(0)
        abs_errors_scaled_rel_ws = np.float64(0)

    for _i, (graphs, probe_graphs, node_array_tuple) in enumerate(val_iterator):
        # node_array_tuple is (targets, wt_mask, probe_mask, [positions])
        # positions may or may not be present, we only need first 3
        targets, wt_mask, probe_mask = node_array_tuple[:3]

        val_err_output = val_errors_fn(
            train_state,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
            targets,
        )

        if cfg.model.scale_rel_ws:
            (
                err_sum_scaled_rel_ws_,
                sq_err_sum_scaled_rel_ws_,
                abs_err_sum_scaled_rel_ws_,
                err_sum_,
                sq_err_sum_,
                abs_err_sum_,
                n_samples_,
            ) = val_err_output

            errors_scaled_rel_ws += np.float64(err_sum_scaled_rel_ws_)
            sq_errors_scaled_rel_ws += np.float64(sq_err_sum_scaled_rel_ws_)
            abs_errors_scaled_rel_ws += np.float64(abs_err_sum_scaled_rel_ws_)
        else:
            (
                err_sum_,
                sq_err_sum_,
                abs_err_sum_,
                n_samples_,
            ) = val_err_output

        errors += np.float64(err_sum_)
        sq_errors += np.float64(sq_err_sum_)
        abs_errors += np.float64(abs_err_sum_)
        n_samples += np.int64(n_samples_)

    metrics = {
        "val_mse": sq_errors / n_samples,
        "val_mae": abs_errors / n_samples,
        "val_RMSE": jnp.sqrt(sq_errors / n_samples),
    }

    if cfg.model.scale_rel_ws:
        metrics["val_mse_scaled_rel_ws"] = sq_errors_scaled_rel_ws / n_samples
        metrics["val_mae_scaled_rel_ws"] = abs_errors_scaled_rel_ws / n_samples
        metrics["val_rmse_scaled_rel_ws"] = jnp.sqrt(sq_errors_scaled_rel_ws / n_samples)

    return metrics


def save_resume_checkpoint(
    cfg: DictConfig,
    checkpoint_manager: ocp.CheckpointManager,
    train_state: TrainState,
    metrics: dict,
    epoch: int,
    rng_key: jax.Array,
    best_metric: float,
    early_stop: EarlyStopping | None,
    loss_hist: list,
    val_hist: list,
    val_epochs: list,
    is_preemption: bool = False,
) -> None:
    """
    Save checkpoint with full resume state for auto-resubmission.

    This function saves all state needed to resume training from a checkpoint,
    including epoch number, RNG key, early stopping state, and training history.

    Args:
        cfg: Configuration object
        checkpoint_manager: Orbax CheckpointManager instance
        train_state: Current training state
        metrics: Training metrics dict
        epoch: Current epoch number
        rng_key: JAX random key for reproducibility
        best_metric: Best validation metric seen so far
        early_stop: EarlyStopping instance (or None if disabled)
        loss_hist: Training loss history list
        val_hist: Validation loss history list
        val_epochs: Epochs when validation was run
        is_preemption: If True, forces save even if metric didn't improve
    """
    resume_state = {
        "epoch": epoch,
        "rng_key": np.array(rng_key),  # Convert to numpy for serialization
        "best_metric": best_metric,
        "early_stop_state": {
            "best_metric": float(early_stop.best_metric),
            "patience_count": int(early_stop.patience_count),
        }
        if early_stop is not None
        else None,
        "loss_hist": loss_hist,
        "val_hist": val_hist,
        "val_epochs": val_epochs,
    }

    ckpt = {
        "train_state": train_state,
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
        "resume_state": resume_state,
    }

    checkpoint_manager.save(epoch, ckpt, force=is_preemption)
    logger.info(
        f"Saved resume checkpoint at epoch {epoch} "
        f"(preemption={is_preemption}, best_metric={best_metric:.6f})"
    )


def save_resume_checkpoint_multi_metric(
    cfg: DictConfig,
    checkpoint_manager: ocp.CheckpointManager,
    train_state: TrainState,
    metrics: dict,
    epoch: int,
    rng_key: jax.Array,
    best_mse: float,
    best_mae: float,
    best_hybrid: float,
    mse_baseline: float | None,
    mae_baseline: float | None,
    early_stop: EarlyStopping | None,
    loss_hist: list,
    val_hist: list,
    val_epochs: list,
    is_preemption: bool = False,
) -> None:
    """
    Save checkpoint with full resume state for multi-metric training.

    Extends save_resume_checkpoint to track all three best metrics
    (MSE, MAE, hybrid) for resumable training.

    Args:
        cfg: Configuration object
        checkpoint_manager: Orbax CheckpointManager instance
        train_state: Current training state
        metrics: Training metrics dict
        epoch: Current epoch number
        rng_key: JAX random key for reproducibility
        best_mse: Best MSE seen so far
        best_mae: Best MAE seen so far
        best_hybrid: Best hybrid metric seen so far
        mse_baseline: Baseline MSE for hybrid normalization
        mae_baseline: Baseline MAE for hybrid normalization
        early_stop: EarlyStopping instance (or None if disabled)
        loss_hist: Training loss history list
        val_hist: Validation loss history list
        val_epochs: Epochs when validation was run
        is_preemption: If True, forces save even if metric didn't improve
    """
    resume_state = {
        "epoch": epoch,
        "rng_key": np.array(rng_key),  # Convert to numpy for serialization
        # Multi-metric state
        "best_mse": best_mse,
        "best_mae": best_mae,
        "best_hybrid": best_hybrid,
        "mse_baseline": mse_baseline,
        "mae_baseline": mae_baseline,
        # Legacy compatibility
        "best_metric": best_mse,
        # Early stopping
        "early_stop_state": {
            "best_metric": float(early_stop.best_metric),
            "patience_count": int(early_stop.patience_count),
        }
        if early_stop is not None
        else None,
        # History
        "loss_hist": loss_hist,
        "val_hist": val_hist,
        "val_epochs": val_epochs,
    }

    ckpt = {
        "train_state": train_state,
        "config": OmegaConf.to_container(cfg),
        "metrics": metrics,
        "resume_state": resume_state,
    }

    checkpoint_manager.save(epoch, ckpt, force=is_preemption)
    logger.info(
        f"Saved resume checkpoint at epoch {epoch} "
        f"(preemption={is_preemption}, best_mse={best_mse:.6f}, "
        f"best_mae={best_mae:.6f}, best_hybrid={best_hybrid:.4f})"
    )


def load_resume_checkpoint(checkpoint_path: str) -> dict | None:
    """
    Load checkpoint with resume state if available.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., model_save_path/checkpoints)

    Returns:
        Dict with checkpoint data including resume_state, or None if not found
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint path does not exist: {checkpoint_path}")
        return None

    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(checkpoint_path, orbax_checkpointer)

    # Get latest checkpoint step
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        logger.info("No checkpoints found for resume")
        return None

    logger.info(f"Found checkpoint at step {latest_step}, loading for resume...")
    restored = checkpoint_manager.restore(latest_step)

    return dict(restored)


def restore_early_stop_state(
    early_stop_state: dict | None,
    cfg: DictConfig,
) -> EarlyStopping | None:
    """
    Restore EarlyStopping from saved state with preserved patience counter.

    This ensures that early stopping continues from where it left off
    across job resubmissions, rather than resetting.

    Args:
        early_stop_state: Saved early stop state dict (or None)
        cfg: Configuration object

    Returns:
        Restored EarlyStopping instance, or None if disabled
    """
    if "early_stop" not in cfg.optimizer:
        return None

    # Create new EarlyStopping with config params
    early_stop = EarlyStopping(
        min_delta=cfg.optimizer.early_stop.criteria,
        patience=int(
            cfg.optimizer.early_stop.patience / cfg.optimizer.validation.rate_of_validation
        ),
    )

    # If we have saved state, restore it
    if early_stop_state is not None:
        # EarlyStopping is a struct.PyTreeNode, use replace to update fields
        early_stop = early_stop.replace(
            best_metric=early_stop_state["best_metric"],
            patience_count=early_stop_state["patience_count"],
        )
        logger.info(
            f"Restored early stopping: best_metric={early_stop.best_metric:.6f}, "
            f"patience_count={early_stop.patience_count}"
        )
    else:
        logger.info("No early stop state to restore, using fresh instance")

    return early_stop


def create_wandb_plot(cfg, model, train_state, plot_components):
    """
    Create W&B visualization plot with 4 inflow conditions.

    Args:
        cfg: Configuration object
        model: Model instance
        train_state: Current training state
        plot_components: List of dictionaries with plot components (one per sample)
                        or single dictionary (legacy format)

    Returns:
        Matplotlib figure with 4x2 grid (graph visualization + Q-Q plot per row)
    """
    # Handle both list (new) and dict (legacy) formats
    if isinstance(plot_components, dict):
        plot_components = [plot_components]

    n_samples = min(len(plot_components), 4)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))

    # Ensure axes is 2D even for single sample
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(plot_components[:n_samples]):
        # Get predictions for this sample
        predictions = model.apply(
            train_state.params,
            sample["graph"],
            sample["probe_graph"],
            sample["wt_mask"],
            sample["probe_mask"],
            train=False,
        )

        # Filter to probe nodes and unscale
        predictions = np.where(sample["probe_mask"] == 1, predictions, np.nan)
        predictions = predictions[~np.isnan(predictions)]
        predictions = predictions * cfg.data.scale_stats["velocity"]["range"][0]

        targets = np.array(sample["targets"])

        # Left column: Graph visualization
        plot_probe_graph_fn(
            sample["graph"],
            sample["probe_graph"],
            sample["positions"],
            include_probe_edges=False,
            ax=axes[i, 0],
        )
        # Include layout metadata in title if available
        layout_info = ""
        if "layout_type" in sample and "n_wt" in sample:
            layout_info = f" [{sample['layout_type']}, n={sample['n_wt']}]"
        axes[i, 0].set_title(f"U={sample['U']:.1f} m/s, TI={sample['TI']:.2%}{layout_info}")

        # Right column: Q-Q plot (predictions vs targets)
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        pred_plot = predictions[:min_len]
        targ_plot = targets[:min_len]

        # Q-Q scatter plot
        axes[i, 1].scatter(targ_plot, pred_plot, alpha=0.5, s=10, c="blue")

        # Add 1:1 line
        min_val = min(np.min(targ_plot), np.min(pred_plot))
        max_val = max(np.max(targ_plot), np.max(pred_plot))
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="1:1")

        # Calculate R² and RMSE
        ss_res = np.sum((pred_plot - targ_plot) ** 2)
        ss_tot = np.sum((targ_plot - np.mean(targ_plot)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((pred_plot - targ_plot) ** 2))

        axes[i, 1].set_xlabel("Target [m/s]")
        axes[i, 1].set_ylabel("Prediction [m/s]")
        axes[i, 1].set_title(f"Q-Q Plot: R²={r2:.3f}, RMSE={rmse:.3f} m/s")
        axes[i, 1].set_aspect("equal", adjustable="box")
        axes[i, 1].legend(loc="lower right")

    plt.tight_layout()
    return fig
