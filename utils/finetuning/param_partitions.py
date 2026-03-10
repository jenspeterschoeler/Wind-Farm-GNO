"""
Parameter partitioning utilities for selective freezing in GNO models.

Provides PyTree manipulation functions to partition parameters by:
- Component (embedder, wt_processor, probe_processor, decoder)
- Layer (specific GNN layers within processors)
- Hybrid (combination of component + layer)

Used by freezing.py and optimizer_builders.py for optax.multi_transform().
"""

import logging
from collections.abc import Callable

from jax import tree_util

logger = logging.getLogger(__name__)

# Component name mapping for GNO model
# This is GNO-specific - other model architectures may have different components
GNO_COMPONENT_MAP = {
    "embedder": ["embed_node", "embed_edge", "RBFEncoder_0"],
    "wt_processor": ["Windfarm_GNN_0"],
    "probe_processor": ["Windfarm_GNN_1"],
    "decoder": ["decoder", "decoder_wt", "decoder_probe"],  # Both strategies
}


def _path_to_string(path: tuple) -> str:
    """Convert PyTree path tuple to string for matching.

    Args:
        path: Tuple of keys representing path in PyTree

    Returns:
        String path like 'params/embed_node/Dense_0/kernel'
    """
    return "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)


def _get_component_from_path(path_str: str) -> str | None:
    """Extract component name from parameter path.

    Args:
        path_str: String path like 'Windfarm_GNN_0/node_update_0/Dense_0/kernel'
                  or 'params/Windfarm_GNN_0/node_update_0/Dense_0/kernel' (with optional prefix)

    Returns:
        Component name ('embedder', 'wt_processor', etc.) or None
    """
    parts = path_str.split("/")
    if len(parts) < 1:
        return None

    # First level is the module name (or second if params/ prefix exists)
    # Check both parts[0] and parts[1] to handle both cases
    module_name = parts[0] if parts[0] != "params" else (parts[1] if len(parts) > 1 else None)
    if module_name is None:
        return None

    # Check which component this module belongs to
    for component, module_names in GNO_COMPONENT_MAP.items():
        if module_name in module_names:
            return component

    return None


def _get_layer_index_from_path(path_str: str, processor_name: str) -> int | None:
    """Extract layer index from GNN processor path.

    Args:
        path_str: String path like 'Windfarm_GNN_0/node_update_2/Dense_0/kernel'
                  or 'params/Windfarm_GNN_0/node_update_2/Dense_0/kernel' (with optional prefix)
        processor_name: Name of processor ('Windfarm_GNN_0' or 'Windfarm_GNN_1')

    Returns:
        Layer index (e.g., 2 from 'node_update_2') or None
    """
    parts = path_str.split("/")
    if len(parts) < 2:
        return None

    # Handle optional params/ prefix
    processor_idx = 0 if parts[0] != "params" else 1
    layer_idx = processor_idx + 1

    if len(parts) <= layer_idx:
        return None

    # Check if this is the right processor
    if parts[processor_idx] != processor_name:
        return None

    # Layer name is next level: node_update_0, node_update_1, message_norm_scale_0, etc.
    layer_name = parts[layer_idx]

    # Extract numeric suffix
    for i, char in enumerate(layer_name):
        if char.isdigit():
            try:
                return int(layer_name[i:])
            except ValueError:
                return None

    return None


def partition_params_by_component(
    params: dict,
    component_names: list[str] | str,
) -> tuple[dict, dict]:
    """Partition parameters by component names.

    Args:
        params: Full parameter dictionary (PyTree)
        component_names: Single component name or list of component names
            Options: 'embedder', 'wt_processor', 'probe_processor', 'decoder'

    Returns:
        Tuple of (selected_params, other_params)
        - selected_params: Params matching component_names
        - other_params: All other params

    Example:
        >>> embedder_params, other = partition_params_by_component(params, 'embedder')
        >>> frozen_params, trainable = partition_params_by_component(
        ...     params, ['embedder', 'wt_processor']
        ... )
    """
    if isinstance(component_names, str):
        component_names = [component_names]

    component_set = set(component_names)

    def is_selected(path, _value):
        path_str = _path_to_string(path)
        component = _get_component_from_path(path_str)
        return component in component_set if component else False

    # Create boolean mask
    mask = tree_util.tree_map_with_path(is_selected, params)

    # Partition using mask
    selected_params, other_params = _partition_by_mask(params, mask)

    return selected_params, other_params


def partition_params_by_layer(
    params: dict,
    layer_indices: list[int],
    processor_name: str = "Windfarm_GNN_0",
) -> tuple[dict, dict]:
    """Partition parameters by layer indices within a GNN processor.

    Args:
        params: Full parameter dictionary (PyTree)
        layer_indices: List of layer indices to select (e.g., [0, 1] for first 2 layers)
        processor_name: Processor module name ('Windfarm_GNN_0' or 'Windfarm_GNN_1')
            - 'Windfarm_GNN_0' = wt_processor
            - 'Windfarm_GNN_1' = probe_processor

    Returns:
        Tuple of (layer_params, other_params)

    Example:
        >>> # Freeze first 2 layers of wt_processor
        >>> frozen_layers, other = partition_params_by_layer(
        ...     params, [0, 1], 'Windfarm_GNN_0'
        ... )
    """
    layer_set = set(layer_indices)

    def is_selected(path, _value):
        path_str = _path_to_string(path)
        layer_idx = _get_layer_index_from_path(path_str, processor_name)
        return layer_idx in layer_set if layer_idx is not None else False

    # Create boolean mask
    mask = tree_util.tree_map_with_path(is_selected, params)

    # Partition using mask
    layer_params, other_params = _partition_by_mask(params, mask)

    return layer_params, other_params


def _partition_by_mask(params: dict, mask: dict) -> tuple[dict, dict]:
    """Core utility to partition params by boolean mask PyTree.

    Args:
        params: Parameter dictionary
        mask: Boolean PyTree with same structure (True = select, False = other)

    Returns:
        Tuple of (selected_params, other_params)
    """

    def select_true(param, mask_val):
        return param if mask_val else None

    def select_false(param, mask_val):
        return None if mask_val else param

    selected = tree_util.tree_map(select_true, params, mask)
    other = tree_util.tree_map(select_false, params, mask)

    return selected, other


def create_partition_spec(freeze_config: dict) -> Callable[[dict], dict]:
    """Factory function to create partition specification from freeze config.

    Creates a function that returns parameter labels ('trainable' or 'frozen')
    for use with optax.multi_transform().

    Args:
        freeze_config: Freezing configuration dict with keys:
            - strategy: 'disabled', 'component', 'layer', or 'hybrid'
            - frozen_components: List of component names (for component/hybrid)
            - frozen_layers: Dict mapping processor_name → layer_indices (for layer/hybrid)

    Returns:
        Function that takes params and returns label PyTree

    Example:
        >>> freeze_config = {
        ...     'strategy': 'hybrid',
        ...     'frozen_components': ['embedder'],
        ...     'frozen_layers': {'Windfarm_GNN_0': [0, 1]}
        ... }
        >>> partition_fn = create_partition_spec(freeze_config)
        >>> param_labels = partition_fn(params)
        >>> # param_labels is a PyTree with 'frozen' or 'trainable' at each leaf
    """
    strategy = freeze_config.get("strategy", "disabled")

    if strategy == "disabled":
        # All params trainable
        def partition_fn(params):
            return tree_util.tree_map(lambda _: "trainable", params)

        return partition_fn

    frozen_components = freeze_config.get("frozen_components", [])
    frozen_layers = freeze_config.get("frozen_layers", {})

    def partition_fn(params):
        """Label each parameter as 'trainable' or 'frozen'."""

        def label_param(path, _value):
            path_str = _path_to_string(path)

            # Check component-level freezing
            if frozen_components:
                component = _get_component_from_path(path_str)
                if component in frozen_components:
                    return "frozen"

            # Check layer-level freezing
            if frozen_layers:
                for processor_name, layer_indices in frozen_layers.items():
                    layer_idx = _get_layer_index_from_path(path_str, processor_name)
                    if layer_idx is not None and layer_idx in layer_indices:
                        return "frozen"

            return "trainable"

        param_labels = tree_util.tree_map_with_path(label_param, params)
        return param_labels

    return partition_fn


def count_params_by_partition(params: dict, param_labels: dict) -> dict[str, int]:
    """Count parameters in each partition.

    Args:
        params: Parameter dictionary
        param_labels: Label PyTree (e.g., 'trainable' or 'frozen' at each leaf)

    Returns:
        Dictionary mapping label → parameter count

    Example:
        >>> counts = count_params_by_partition(params, param_labels)
        >>> print(counts)
        {'trainable': 1234567, 'frozen': 234567, 'total': 1469134}
    """
    import numpy as np

    counts = {}

    # Flatten both trees
    params_flat = tree_util.tree_leaves(params)
    labels_flat = tree_util.tree_leaves(param_labels)

    # Count by label
    for param, label in zip(params_flat, labels_flat):
        if label not in counts:
            counts[label] = 0
        counts[label] += int(np.prod(param.shape))

    # Add total
    counts["total"] = sum(counts.values())

    return counts


def log_partition_info(params: dict, param_labels: dict, logger_instance=None):
    """Log partition statistics to console.

    Args:
        params: Parameter dictionary
        param_labels: Label PyTree
        logger_instance: Optional logger (uses module logger if None)
    """
    if logger_instance is None:
        logger_instance = logger

    counts = count_params_by_partition(params, param_labels)

    logger_instance.info("=" * 60)
    logger_instance.info("PARAMETER PARTITIONING")
    logger_instance.info("=" * 60)

    for label, count in sorted(counts.items()):
        if label == "total":
            continue
        percentage = 100.0 * count / counts["total"]
        logger_instance.info(f"{label:>12s}: {count:>10,d} params ({percentage:>5.1f}%)")

    logger_instance.info("-" * 60)
    logger_instance.info(f"{'Total':>12s}: {counts['total']:>10,d} params (100.0%)")
    logger_instance.info("=" * 60)
