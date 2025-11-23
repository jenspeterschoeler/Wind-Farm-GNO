# from . import run_pywake
from . import data_tools, graph, plotting
from .GNO_probe import initialize_GNO_probe, inverse_scale_rel_ws, scale_rel_ws
from .misc import (
    add_to_hydra_cfg,
    convert_ndarray,
    convert_to_wandb_format,
    get_model_paths,
    get_model_save_paths,
    get_run_info,
    setup_optimizer,
)
from .to_graph import append_globals_to_nodes, get_node_indexes, min_max_scale, to_graph
from .torch_loader import (
    JraphDataLoader,
    Torch_Geomtric_Dataset,
    dynamically_batch_graph_probe_operator,
    load_sample_probabilities,
    sum_by_parts_torch,
)

# HACK, lazy import to avoid circular import on package import
_allowed_lazy = {"load_portable_model", "save_portable_model"}


def __getattr__(name):
    if name in _allowed_lazy:
        from importlib import import_module

        mod = import_module(".weight_converter", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
