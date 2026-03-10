"""
Utility module for Article 2 publication figures.

Contains configuration, model loading, data extraction, prediction helpers,
and error metrics. All plotting functions are in article2_plotting.py.

Usage:
    from article2_utils import Article2Config, load_model_for_prediction, create_turbopark_wf_model
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# Section 1: Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a trained GNO model."""

    name: str
    description: str
    checkpoint_path: Path | None = None  # Orbax checkpoint dir
    portable_path: Path | None = None  # Exported msgpack dir
    dataset_path: Path | None = None  # Dataset root (for scale_stats.json)

    @property
    def run_id(self) -> str:
        """Derive unique run identifier from checkpoint path.

        Handles both multirun and single-run Hydra output structures:
            multirun → "L2_dropout_layernorm_2026-02-03_12-12-20_run5_mse267"
            single   → "L2_dropout_layernorm_2026-02-03_12-12-20_mse267"
            no ckpt  → falls back to self.name
        """
        if self.checkpoint_path is None:
            return self.name

        parts = self.checkpoint_path.parts
        epoch = parts[-1]  # e.g. "267"
        ckpt_type = parts[-2].replace("checkpoints_best_", "")  # e.g. "mse"
        # parts[-3] = "model"
        variant = parts[-4]  # e.g. "L2_dropout_layernorm"

        if "multirun" in parts:
            mi = parts.index("multirun")
            date = parts[mi + 1]
            time = parts[mi + 2]
            run_num = parts[mi + 3].split("_")[0]
            return f"{variant}_{date}_{time}_run{run_num}_{ckpt_type}{epoch}"
        else:
            vi = list(parts).index(variant)
            date = parts[vi - 2]
            time = parts[vi - 1]
            return f"{variant}_{date}_{time}_{ckpt_type}{epoch}"

    @property
    def dataset_name(self) -> str:
        """Return dataset directory name for cache organization."""
        if self.dataset_path is not None:
            return self.dataset_path.name
        return "default"


@dataclass
class Article2Config:
    """Configuration for Article 2 publication figures."""

    model: ModelConfig
    test_data_path: Path
    dataset_root: Path
    x_downstream: list[int] = field(default_factory=lambda: [100, 200, 300])
    U_free: list[float] = field(default_factory=lambda: [6.0, 12.0, 18.0])
    TI_flow: float = 0.05
    grid_density: int = 3
    layout_types: list[str] = field(
        default_factory=lambda: [
            "cluster",
            "single string",
            "multiple string",
            "parallel string",
        ]
    )


# Pre-defined model configurations
MODELS = {
    "L2_global": ModelConfig(
        name="L2_global",
        description="Phase 1 best model - L2 with dropout + layernorm",
        checkpoint_path=Path(
            "/home/jpsch/Documents/Sophia_work/gno/outputs/transfer_learning"
            "/phase1_global/multirun/2026-02-03/12-12-20"
            "/5_+experiment=phase1_global/L2_dropout_layernorm"
            "/model/checkpoints_best_mse/267"
        ),
        portable_path=REPO_ROOT / "assets" / "best_model_L2_global",
        dataset_path=Path(
            "/home/jpsch/Documents/Sophia_work/gno/data-generation/data/turbopark_2500layouts"
        ),
    ),
    "L1_F1_phase2": ModelConfig(
        name="L1_F1_phase2",
        description="Phase 2 - L1 global, F1 full fine-tune on AWF RANS (no clipping)",
        checkpoint_path=Path(
            "/home/jpsch/Documents/Sophia_work/gno/outputs/transfer_learning"
            "/phase2_global/multirun/2026-02-09/16-05-48"
            "/1_+experiment=phase2/L1_global_F1,finetuning"
            "/gradient_clipping=disabled/model/checkpoints_best_mse/159"
        ),
        portable_path=REPO_ROOT / "assets" / "best_model_L1_F1_phase2",
        dataset_path=Path("/home/jpsch/Documents/Sophia_work/gno/data-generation/data/awf_graphs"),
    ),
    "L1_L3_phase2": ModelConfig(
        name="L1_L3_phase2",
        description="Phase 2 best - L1 global, L3 LoRA r=16 on AWF RANS (no clipping)",
        checkpoint_path=Path(
            "/work/users/jpsch/gno/outputs/transfer_learning"
            "/phase2_global/multirun/2026-02-23/11-20-35"
            "/9_+experiment=phase2/L1_global_L3,finetuning"
            "/gradient_clipping=disabled/model/checkpoints_best_mse/216"
        ),
        portable_path=REPO_ROOT / "assets" / "best_model_L1_L3_phase2",
        dataset_path=Path("/home/jpsch/Documents/Sophia_work/gno/data-generation/data/awf_graphs"),
    ),
    "L1_global_scratch": ModelConfig(
        name="L1_global_scratch",
        description="Phase 2 baseline - L1 architecture trained from scratch on AWF RANS",
        checkpoint_path=Path(
            "/work/users/jpsch/gno/outputs/transfer_learning"
            "/phase2_global/multirun/2026-02-11/15-24-23"
            "/0_+experiment=phase2/L1_global_scratch"
            "/model/checkpoints_best_mse/1710"
        ),
        portable_path=REPO_ROOT / "assets" / "best_model_L1_global_scratch",
        dataset_path=Path("/home/jpsch/Documents/Sophia_work/gno/data-generation/data/awf_graphs"),
    ),
}


def get_sophia_test_path() -> str:
    """Detect whether running on Sophia cluster or locally and return test data path."""
    sophia_path = (
        "/work/users/jpsch/gno/data-generation/data/turbopark_2500layouts/test_pre_processed"
    )
    if Path(sophia_path).exists():
        return sophia_path
    # SSHFS mount
    sshfs_path = (
        Path.home()
        / "Documents"
        / "Sophia_work"
        / "gno"
        / "data-generation"
        / "data"
        / "turbopark_2500layouts"
        / "test_pre_processed"
    )
    if sshfs_path.exists():
        return str(sshfs_path)
    # Local copy
    local_path = REPO_ROOT / "data" / "turbopark_2500layouts" / "test_pre_processed"
    if local_path.exists():
        return str(local_path)
    raise FileNotFoundError(
        f"Test data not found at:\n  {sophia_path}\n  {sshfs_path}\n  {local_path}\n"
        "Ensure the turbopark_2500layouts dataset is accessible."
    )


def get_awf_test_path() -> str:
    """Detect whether running on Sophia cluster or locally and return AWF test data path."""
    sophia_path = "/work/users/jpsch/gno/data-generation/data/awf_graphs/test_pre_processed"
    if Path(sophia_path).exists():
        return sophia_path
    sshfs_path = (
        Path.home()
        / "Documents"
        / "Sophia_work"
        / "gno"
        / "data-generation"
        / "data"
        / "awf_graphs"
        / "test_pre_processed"
    )
    if sshfs_path.exists():
        return str(sshfs_path)
    local_path = REPO_ROOT / "data" / "awf_graphs" / "test_pre_processed"
    if local_path.exists():
        return str(local_path)
    raise FileNotFoundError(
        f"AWF test data not found at:\n  {sophia_path}\n  {sshfs_path}\n  {local_path}\n"
        "Ensure the awf_graphs dataset is accessible."
    )


def get_awf_database_path() -> str:
    """Detect whether running on Sophia cluster or locally and return AWF database path."""
    sophia_path = "/work/users/jpsch/gno/data-generation/data/awf_database.nc"
    if Path(sophia_path).exists():
        return sophia_path
    sshfs_path = (
        Path.home()
        / "Documents"
        / "Sophia_work"
        / "gno"
        / "data-generation"
        / "data"
        / "awf_database.nc"
    )
    if sshfs_path.exists():
        return str(sshfs_path)
    submodule_path = REPO_ROOT / "data-generation" / "data" / "awf_database.nc"
    if submodule_path.exists():
        return str(submodule_path)
    raise FileNotFoundError(
        f"AWF database not found at:\n  {sophia_path}\n  {sshfs_path}\n  {submodule_path}\n"
        "Ensure the awf_database.nc file is accessible."
    )


AWF_TURBINE_DIAMETER = 178.3  # DTU10MW rotor diameter in meters


def construct_awf_probe_graph(
    wt_positions_m: np.ndarray,
    wseff: np.ndarray,
    ws_inf: float,
    ti_inf: float,
    probe_positions_m: np.ndarray,
    probe_velocities: np.ndarray,
    scale_stats: dict,
    return_positions: bool = False,
) -> tuple:
    """Build a jraph probe graph from AWF data without PyWake.

    Mirrors construct_on_the_fly_probe_graph() but uses RANS data directly.
    AWF node features are 1D (wseff only), requiring manual scaling instead
    of min_max_scale() which expects 3D node features [U, TI, CT].

    Args:
        wt_positions_m: (n_wt, 2) WT positions in meters
        wseff: (n_wt,) effective wind speed at each WT [m/s]
        ws_inf: freestream wind speed [m/s]
        ti_inf: turbulence intensity [-]
        probe_positions_m: (n_probes, 2) probe positions in meters
        probe_velocities: (n_probes,) RANS velocities at probes [m/s]
        scale_stats: dataset scaling statistics
        return_positions: if True, include node positions in return tuple

    Returns:
        (jraph_graph, jraph_probe_graphs, node_array_tuple)
    """
    import torch

    from utils.graph import torch_pyg_to_jraph
    from utils.to_graph import append_globals_to_nodes, to_graph

    # Step 1: Build PyG graph
    pyg_graph = to_graph(
        points=wt_positions_m,
        connectivity="delaunay",
        add_edge="cartesian",
        node_features=wseff.reshape(-1, 1),
        global_features=np.array([ws_inf, ti_inf]),
        trunk_inputs=probe_positions_m,
        output_features=probe_velocities.reshape(-1, 1),
        rel_wd=None,
    )

    # Step 2: Add batch dimensions
    pyg_graph.output_features = pyg_graph.output_features.unsqueeze(0)
    pyg_graph.n_edge = pyg_graph.n_edge.unsqueeze(0)
    pyg_graph.trunk_inputs = pyg_graph.trunk_inputs.unsqueeze(0)
    pyg_graph.n_node = pyg_graph.n_node.unsqueeze(0)
    pyg_graph.global_features = pyg_graph.global_features.unsqueeze(0)

    # Step 3: Manual scaling (run4: all mins = 0)
    # AWF has 1D node features [wseff], so min_max_scale (which expects 3D
    # [U, TI, CT]) would broadcast incorrectly. Scale each component directly.
    vel_range = torch.tensor(scale_stats["velocity"]["range"])
    dist_range = torch.tensor(scale_stats["distance"]["range"])
    ti_range = torch.tensor(scale_stats["ti"]["range"])

    pyg_graph.node_features = pyg_graph.node_features / vel_range
    pyg_graph.output_features = pyg_graph.output_features / vel_range
    pyg_graph.global_features = pyg_graph.global_features / torch.cat([vel_range, ti_range])
    pyg_graph.edge_attr = pyg_graph.edge_attr / dist_range
    pyg_graph.trunk_inputs = pyg_graph.trunk_inputs / dist_range
    pyg_graph.pos = pyg_graph.pos / dist_range

    # Step 4: Append globals to nodes → [wseff_s, ws_inf_s, ti_inf_s] (3D)
    pyg_graph = append_globals_to_nodes(pyg_graph)

    # Step 5: Convert to jraph with probe graphs
    # After add_pos_to_nodes: [wseff_s, ws_inf_s, ti_inf_s, pos_x_s, pos_y_s] (5D)
    # input_node_feature_idxs=[3, 4] selects [pos_x_s, pos_y_s] for WT nodes
    # Probe nodes get globals [ws_inf_s, ti_inf_s] — matches AWF training config
    jraph_graph, jraph_probe_graphs, node_array_tuple = torch_pyg_to_jraph(
        pyg_graph,
        graphs_only=False,
        probe_graphs=True,
        input_node_feature_idxs=[3, 4],
        target_node_feature_idxs=[0],
        add_pos_to_nodes=True,
        add_pos_to_edges=False,
        return_positions=return_positions,
        return_idxs=False,
    )

    return jraph_graph, jraph_probe_graphs, node_array_tuple


# =============================================================================
# Section 2: Model Loading
# =============================================================================


def export_orbax_to_portable(
    checkpoint_path: Path, output_dir: Path, dataset_path: Path | None = None
) -> Path:
    """Export Orbax checkpoint to portable msgpack format.

    Args:
        checkpoint_path: Path to Orbax checkpoint (step directory)
        output_dir: Where to save exported files
        dataset_path: Optional dataset root to copy scale_stats.json from

    Returns:
        Path to output directory
    """
    import flax.serialization
    from omegaconf import OmegaConf

    from utils.model_tools import load_model

    print(f"Exporting model from checkpoint: {checkpoint_path}")
    model, params, metrics, cfg_model = load_model(checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params to msgpack
    params_path = output_dir / "best_params.msgpack"
    bytes_data = flax.serialization.to_bytes(params)
    params_path.write_bytes(bytes_data)
    print(f"  Saved: {params_path}")

    # Save model config
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg_model), f, indent=4)
    print(f"  Saved: {config_path}")

    # Copy scale_stats from dataset if available
    if dataset_path is not None:
        src_stats = dataset_path / "scale_stats.json"
        if src_stats.exists():
            dst_stats = output_dir / "scale_stats.json"
            dst_stats.write_text(src_stats.read_text())
            print(f"  Saved: {dst_stats}")

    # Extract scale_stats from model config as fallback
    cfg_dict = OmegaConf.to_container(cfg_model)
    if "data" in cfg_dict and "scale_stats" in cfg_dict["data"]:
        scale_stats_path = output_dir / "scale_stats.json"
        if not scale_stats_path.exists():
            with open(scale_stats_path, "w") as f:
                json.dump(cfg_dict["data"]["scale_stats"], f, indent=4)
            print(f"  Saved: {scale_stats_path}")

    return output_dir


def ensure_model_exported(model_cfg: ModelConfig) -> Path:
    """Ensure model is exported to portable format. Returns path to portable dir."""
    if (
        model_cfg.portable_path is not None
        and (model_cfg.portable_path / "best_params.msgpack").exists()
    ):
        return model_cfg.portable_path

    if model_cfg.checkpoint_path is None:
        raise FileNotFoundError(
            f"Model {model_cfg.name}: No portable model found at {model_cfg.portable_path} "
            "and no checkpoint_path specified for export."
        )

    output_dir = model_cfg.portable_path or (REPO_ROOT / "assets" / f"best_model_{model_cfg.name}")
    return export_orbax_to_portable(model_cfg.checkpoint_path, output_dir, model_cfg.dataset_path)


def load_model_for_prediction(model_cfg: ModelConfig, scale_stats: dict | None = None) -> tuple:
    """Load model and create JIT-compiled prediction function.

    Args:
        model_cfg: Model configuration
        scale_stats: Optional pre-loaded scale stats (overrides loading from file)

    Returns:
        Tuple of (pred_fn, inverse_scale_target, scale_stats, model, params)
    """
    import jax
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.data_tools import setup_unscaler
    from utils.run_pywake import construct_on_the_fly_probe_graph
    from utils.weight_converter import load_portable_model

    model_path = ensure_model_exported(model_cfg)
    print(f"Loading model: {model_cfg.name}")
    print(f"  Path: {model_path}")

    model_cfg_path = model_path / "model_config.json"
    params_path = model_path / "best_params.msgpack"
    scale_stats_path = model_path / "scale_stats.json"

    # Load model config
    with open(model_cfg_path) as f:
        restored_cfg_model = DictConfig(json.load(f))

    # Load scale_stats
    if scale_stats is None:
        with open(scale_stats_path) as f:
            scale_stats = json.load(f)

    # Fix ct stats if None (turbopark_2500layouts lacks CT data;
    # with run4 scaling ct_min is overridden to 0 anyway)
    if scale_stats.get("ct", {}).get("min") is None:
        scale_stats["ct"] = {"min": [0.0], "range": [1.0]}

    # Setup unscaler
    unscaler = setup_unscaler(restored_cfg_model, scale_stats=scale_stats)
    inverse_scale_target = unscaler.inverse_scale_output

    # Initialize model with dummy probe graph
    print("Initializing model with dummy probe graph...")
    wt = DTU10MW()
    # Use a simple 3-turbine layout for initialization
    _init_positions = jnp.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
    _init_grid = HorizontalGrid(x=[0], y=[0], h=wt.hub_height())

    _init_jraph, _init_probe, _init_tuple = construct_on_the_fly_probe_graph(
        positions=_init_positions,
        U=[10.0],
        TI=[0.05],
        grid=_init_grid,
        scale_stats=scale_stats,
        return_positions=False,
    )
    _init_targets, _init_wt_mask, _init_probe_mask = _init_tuple
    _init_wt_mask = jnp.atleast_2d(_init_wt_mask).T
    _init_probe_mask = jnp.atleast_2d(_init_probe_mask).T
    _init_tuple_reshaped = (_init_targets, _init_wt_mask, _init_probe_mask)

    restored_params, restored_cfg_model, model, _ = load_portable_model(
        str(params_path),
        str(model_cfg_path),
        dataset=None,
        inputs=(_init_jraph, _init_probe, _init_tuple_reshaped),
    )
    print("Model initialized successfully.")

    # Create JIT-compiled prediction function
    def model_prediction_fn(
        input_graphs, input_probe_graphs, input_wt_mask, input_probe_mask
    ) -> jnp.ndarray:
        return model.apply(
            restored_params,
            input_graphs,
            input_probe_graphs,
            input_wt_mask,
            input_probe_mask,
        )

    pred_fn = jax.jit(model_prediction_fn)

    return pred_fn, inverse_scale_target, scale_stats, model, restored_params


def create_turbopark_wf_model():
    """Create TurbOPark PyWake wind farm model matching data-generation config.

    Configuration:
    - TurboGaussianDeficit: Gaussian wake deficit (TurbOPark / Nygaard_2022)
    - SquaredSum: Quadratic superposition
    - PropagateDownwind: Simple downwind propagation
    - No blockage model
    - No turbulence model
    """
    from py_wake.deficit_models.gaussian import TurboGaussianDeficit
    from py_wake.examples.data.dtu10mw import DTU10MW
    from py_wake.site._site import UniformSite
    from py_wake.superposition_models import SquaredSum
    from py_wake.wind_farm_models import PropagateDownwind

    wt = DTU10MW()
    site = UniformSite()

    return PropagateDownwind(
        site,
        wt,
        wake_deficitModel=TurboGaussianDeficit(),
        superpositionModel=SquaredSum(),
    )


# =============================================================================
# Section 3: Data Extraction
# =============================================================================


def select_representative_layouts(
    dataset,
    target_n_wt: int = 50,
    target_wt_spacing: float = 5.0,
    idx_start: int = 0,
) -> dict[str, int | None]:
    """Select representative graph indices for each layout type based on y-range similarity.

    Finds one representative graph per layout type that has similar y-range to a
    reference cluster case (with specified n_wt and wt_spacing constraints).
    """
    layout_type_idxs = {
        "cluster": None,
        "single string": None,
        "multiple string": None,
        "parallel string": None,
    }

    best_offsets = {lt: 1e9 for lt in layout_type_idxs}
    y_chosen = False
    y_range_chosen = None

    for idx, data in tqdm(enumerate(dataset), desc="Selecting representative layouts"):
        layout_type = data.layout_type
        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()

        if (
            idx > idx_start
            and layout_type == "cluster"
            and not y_chosen
            and np.round(data.wt_spacing, 0) == target_wt_spacing
            and data.n_wt >= target_n_wt
        ):
            y_range_chosen = y_range
            y_chosen = True

        if y_chosen:
            y_range_offset = np.abs(y_range_chosen - y_range)
            if y_range_offset < best_offsets[layout_type]:
                layout_type_idxs[layout_type] = idx
                best_offsets[layout_type] = y_range_offset

    return layout_type_idxs


def select_representative_layouts_per_windspeed(
    dataset,
    scale_stats: dict,
    target_windspeeds: list[float] | None = None,
    windspeed_tolerance: float = 0.5,
    target_n_wt: int = 50,
    target_wt_spacing: float = 5.0,
    idx_start: int = 0,
) -> dict[str, dict[float, int | None]]:
    """Select representative graph indices for each layout type AND wind speed."""
    if target_windspeeds is None:
        target_windspeeds = [6.0, 12.0, 18.0]

    layout_types = ["cluster", "single string", "multiple string", "parallel string"]
    layout_ws_idxs = {lt: {ws: None for ws in target_windspeeds} for lt in layout_types}
    best_offsets = {lt: {ws: 1e9 for ws in target_windspeeds} for lt in layout_types}

    vel_min = scale_stats["velocity"]["min"][0]
    vel_range = scale_stats["velocity"]["range"][0]

    # Pass 1: Find reference y-range
    y_range_chosen = None
    for idx, data in enumerate(dataset):
        if idx <= idx_start:
            continue
        if data.layout_type != "cluster":
            continue
        if np.round(data.wt_spacing, 0) != target_wt_spacing:
            continue
        if data.n_wt < target_n_wt:
            continue
        y_range_chosen = data.pos[:, 1].max() - data.pos[:, 1].min()
        break

    if y_range_chosen is None:
        print("  Warning: No valid cluster reference found, using y_range=0")
        y_range_chosen = 0

    # Pass 2: Select samples
    for idx, data in tqdm(
        enumerate(dataset), desc="Selecting per-windspeed layouts", total=len(dataset)
    ):
        layout_type = data.layout_type
        if layout_type not in layout_types:
            continue

        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()
        y_range_offset = np.abs(y_range_chosen - y_range)

        scaled_U = data.global_features[0].item()
        actual_U = scaled_U * vel_range + vel_min

        for target_ws in target_windspeeds:
            if (
                np.abs(actual_U - target_ws) <= windspeed_tolerance
                and y_range_offset < best_offsets[layout_type][target_ws]
            ):
                layout_ws_idxs[layout_type][target_ws] = idx
                best_offsets[layout_type][target_ws] = y_range_offset

    print("\nPer-windspeed selection results:")
    for lt in layout_types:
        for ws in target_windspeeds:
            idx = layout_ws_idxs[lt][ws]
            status = f"idx={idx}" if idx is not None else "NOT FOUND"
            print(f"  {lt}, U={ws} m/s: {status}")

    return layout_ws_idxs


def get_max_plot_distance(dataset, layout_type_idxs: dict) -> tuple[float, float]:
    """Compute max distance and y_range for consistent plot scaling."""
    max_distance = 0
    max_y_range = 0
    for idx in layout_type_idxs.values():
        if idx is None:
            continue
        data = dataset[idx]
        x_range = data.pos[:, 0].max() - data.pos[:, 0].min()
        y_range = data.pos[:, 1].max() - data.pos[:, 1].min()
        distance = max(x_range, y_range)
        if distance > max_distance:
            max_distance = distance
        if y_range > max_y_range:
            max_y_range = y_range

    plot_distance = max_distance * 1.5
    return plot_distance, max_y_range


def convert_graph_to_serializable(graph) -> dict:
    """Convert a jraph GraphsTuple to a serializable dictionary."""
    return {
        "nodes": np.array(graph.nodes) if graph.nodes is not None else None,
        "edges": np.array(graph.edges) if graph.edges is not None else None,
        "receivers": np.array(graph.receivers) if graph.receivers is not None else None,
        "senders": np.array(graph.senders) if graph.senders is not None else None,
        "globals": np.array(graph.globals) if graph.globals is not None else None,
        "n_node": np.array(graph.n_node) if graph.n_node is not None else None,
        "n_edge": np.array(graph.n_edge) if graph.n_edge is not None else None,
    }


def extract_layout_data(
    dataset,
    cfg: Article2Config,
    layout_type_idxs: dict,
    layout_ws_idxs: dict,
    plot_graphs: dict,
    test_dataset,
    scale_stats: dict,
) -> dict:
    """Extract and serialize all layout data into a pickle-ready dictionary.

    Args:
        dataset: The full dataset
        cfg: Article2Config instance
        layout_type_idxs: {layout_type: dataset_idx} from select_representative_layouts
        layout_ws_idxs: {layout_type: {ws: idx}} from select_representative_layouts_per_windspeed
        plot_graphs: Loaded graph data from setup_plot_iterator
        test_dataset: The test dataset from setup_plot_iterator
        scale_stats: Dataset scaling statistics

    Returns:
        Serialized data dict ready for pickle.dump
    """
    plot_distance, max_y_range = get_max_plot_distance(dataset, layout_type_idxs)

    serialized_data = {
        "metadata": {
            "layout_type_idxs": layout_type_idxs,
            "plot_distance": plot_distance,
            "max_y_range": max_y_range,
        },
        "scale_stats": scale_stats,
        "layout_data": {},
    }

    # Main layout data
    for layout_name, val in tqdm(plot_graphs.items(), desc="Processing layouts"):
        if "_" in layout_name and any(layout_name.startswith(f"{lt}_") for lt in cfg.layout_types):
            continue  # Skip per-windspeed keys

        graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = val
        targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

        test_idx = layout_type_idxs[layout_name]
        wt_positions = test_dataset[test_idx].pos.numpy()

        serialized_data["layout_data"][layout_name] = {
            "graphs": convert_graph_to_serializable(graphs),
            "probe_graphs": convert_graph_to_serializable(probe_graphs),
            "targets": np.array(targets),
            "wt_mask": np.array(wt_mask),
            "probe_mask": np.array(probe_mask),
            "node_positions": np.array(node_positions),
            "trunk_idxs": np.array(trunk_idxs),
            "layout_type": layout_type,
            "wt_spacing": float(wt_spacing.numpy()[0])
            if hasattr(wt_spacing, "numpy")
            else float(wt_spacing),
            "wt_positions": wt_positions,
            "test_idx": test_idx,
        }

    # Per-windspeed data
    vel_min = scale_stats["velocity"]["min"][0]
    vel_range = scale_stats["velocity"]["range"][0]

    serialized_data["per_windspeed_data"] = {
        "target_windspeeds": cfg.U_free,
        "layout_ws_idxs": layout_ws_idxs,
        "layout_data": {},
    }

    for layout_name in layout_ws_idxs:
        serialized_data["per_windspeed_data"]["layout_data"][layout_name] = {}

        for ws, idx in layout_ws_idxs[layout_name].items():
            if idx is None:
                print(f"  Warning: No data found for {layout_name} at U={ws} m/s")
                continue

            key = f"{layout_name}_{ws}"
            data_in = plot_graphs.get(key)
            if data_in is None:
                continue

            graphs, probe_graphs, node_array_tuple, layout_type, wt_spacing = data_in
            targets, wt_mask, probe_mask, node_positions, trunk_idxs = node_array_tuple

            actual_U = graphs.globals[0][0] * vel_range + vel_min

            serialized_data["per_windspeed_data"]["layout_data"][layout_name][ws] = {
                "graphs": convert_graph_to_serializable(graphs),
                "targets": np.array(targets),
                "wt_mask": np.array(wt_mask),
                "probe_mask": np.array(probe_mask),
                "node_positions": np.array(node_positions),
                "trunk_idxs": np.array(trunk_idxs),
                "wt_positions": dataset[idx].pos.numpy(),
                "test_idx": idx,
                "actual_U": float(actual_U),
            }

    return serialized_data


# =============================================================================
# Section 4: Prediction Helpers
# =============================================================================


def dict_to_graph(graph_dict):
    """Convert a serialized dictionary back to a jraph GraphsTuple."""
    import jraph

    return jraph.GraphsTuple(
        nodes=graph_dict["nodes"],
        edges=graph_dict["edges"],
        receivers=graph_dict["receivers"],
        senders=graph_dict["senders"],
        globals=graph_dict["globals"],
        n_node=graph_dict["n_node"],
        n_edge=graph_dict["n_edge"],
    )


def apply_mask(arr, mask):
    """Apply mask and remove NaN values."""
    arr = np.asarray(arr)
    mask = np.asarray(mask)
    arr = np.where(mask != 0, arr, np.nan)
    arr = arr[~np.isnan(arr)]
    return arr


def apply_normalizations(
    predictions: np.ndarray,
    targets: np.ndarray,
    U_flow: float,
    plot_velocity_deficit: bool = True,
    normalize_by_U: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply velocity deficit and normalization transformations."""
    label = ""
    if plot_velocity_deficit:
        predictions = predictions - U_flow
        targets = targets - U_flow
        label += r"$\Delta u"
    else:
        label += r"$u"

    if normalize_by_U:
        predictions = predictions / U_flow
        targets = targets / U_flow
        label += r"/U$ [-]"
    else:
        label += r"$ [$\mathrm{ms}^{-1}$]"

    return predictions, targets, label


def predict_crossstream_profile(
    pred_fn,
    inverse_scale_target,
    scale_stats: dict,
    wt_positions_m: np.ndarray,
    x_downstream_D: int,
    y_probes: np.ndarray,
    U_flow: float,
    TI_flow: float,
    wf_model=None,
) -> dict:
    """Generate crossstream profile predictions at a specific downstream distance.

    Args:
        pred_fn: JIT-compiled prediction function
        inverse_scale_target: Unscaling function for predictions
        scale_stats: Scaling statistics dict
        wt_positions_m: WT positions in meters, shape (n_wt, 2)
        x_downstream_D: Downstream distance in rotor diameters
        y_probes: Y-coordinates for probe grid in meters
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        wf_model: Optional PyWake wind farm model (None = default NiayifarGaussian)

    Returns:
        Dict with predictions, targets, errors at this x/U combination
    """
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.run_pywake import construct_on_the_fly_probe_graph

    wt = DTU10MW()
    D = wt.diameter()

    grid = HorizontalGrid(
        x=[x_downstream_D * D + wt_positions_m[:, 0].max()],
        y=y_probes,
        h=wt.hub_height(),
    )

    jraph_graph, jraph_probe, node_tuple = construct_on_the_fly_probe_graph(
        positions=wt_positions_m,
        U=[U_flow],
        TI=[TI_flow],
        grid=grid,
        scale_stats=scale_stats,
        return_positions=True,
        wf_model=wf_model,
    )
    targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_tuple

    prediction = pred_fn(
        jraph_graph,
        jraph_probe,
        jnp.atleast_2d(wt_mask_gen).T,
        jnp.atleast_2d(probe_mask_gen).T,
    ).squeeze()

    unscaled_predictions = inverse_scale_target(apply_mask(prediction, probe_mask_gen))
    unscaled_targets = inverse_scale_target(apply_mask(targets_gen.squeeze(), probe_mask_gen))

    # Compute error metrics
    errors = compute_crossstream_errors(
        np.array(unscaled_targets), np.array(unscaled_predictions), U_flow
    )

    # Normalized profiles for plotting
    norm_preds, norm_tgts, _ = apply_normalizations(
        np.array(unscaled_predictions),
        np.array(unscaled_targets),
        U_flow,
        plot_velocity_deficit=True,
        normalize_by_U=True,
    )

    return {
        "unscaled_predictions": np.array(unscaled_predictions),
        "unscaled_targets": np.array(unscaled_targets),
        "normalized_predictions": norm_preds,
        "normalized_targets": norm_tgts,
        "y/D": y_probes / D,
        "errors": errors,
        "jraph_graph": jraph_graph,
        "jraph_probe": jraph_probe,
        "node_positions": node_positions_gen,
        "wt_mask": wt_mask_gen,
        "probe_mask": probe_mask_gen,
    }


def predict_wt_errors(
    pred_fn,
    model_params,
    model,
    inverse_scale_target,
    scale_stats: dict,
    wt_positions_m: np.ndarray,
    U_flow: float,
    TI_flow: float,
    wf_model=None,
) -> dict:
    """Predict WT-level errors at a specific wind speed using PyWake regeneration.

    Args:
        pred_fn: JIT-compiled prediction function (or model.apply)
        model_params: Model parameters (used with model.apply)
        model: The model (used if pred_fn wraps model.apply)
        inverse_scale_target: Unscaling function
        scale_stats: Scaling statistics
        wt_positions_m: WT positions in meters
        U_flow: Freestream wind speed [m/s]
        TI_flow: Turbulence intensity [-]
        wf_model: Optional PyWake wind farm model

    Returns:
        Dict with wt_rel_errors, wt_predictions, wt_targets
    """
    from jax import numpy as jnp
    from py_wake import HorizontalGrid
    from py_wake.examples.data.dtu10mw import DTU10MW

    from utils.run_pywake import construct_on_the_fly_probe_graph

    wt = DTU10MW()

    # Minimal grid for WT-only predictions
    grid = HorizontalGrid(
        x=wt_positions_m[:, 0][:1],
        y=wt_positions_m[:, 1][:1],
        h=wt.hub_height(),
    )

    jraph_graph, jraph_probe, node_tuple = construct_on_the_fly_probe_graph(
        positions=wt_positions_m,
        U=[U_flow],
        TI=[TI_flow],
        grid=grid,
        scale_stats=scale_stats,
        return_positions=True,
        wf_model=wf_model,
    )
    targets_gen, wt_mask_gen, probe_mask_gen, node_positions_gen = node_tuple

    prediction = pred_fn(
        jraph_graph,
        jraph_probe,
        jnp.atleast_2d(wt_mask_gen).T,
        jnp.atleast_2d(probe_mask_gen).T,
    ).squeeze()

    wt_idx = np.where(wt_mask_gen != 0)[0]
    wt_predictions = np.array(inverse_scale_target(prediction[wt_idx]).squeeze())
    wt_targets = np.array(inverse_scale_target(np.array(targets_gen)[wt_idx]).squeeze())

    wt_errors = wt_targets - wt_predictions
    wt_rel_errors = (wt_errors / U_flow) * 100

    return {
        "wt_rel_errors": wt_rel_errors,
        "wt_predictions": wt_predictions,
        "wt_targets": wt_targets,
    }


# =============================================================================
# Section 5: Error Metrics
# =============================================================================


@dataclass
class CrossstreamErrors:
    """Error metrics for a single crossstream profile."""

    freestream_normalized: np.ndarray  # |u - û| / U
    percentage: np.ndarray  # |u - û| / |u| [%]
    deficit_normalized: np.ndarray  # |u - û| / |U - u| [%]
    deficit_normalized_filtered: np.ndarray  # filtered by 5% deficit threshold
    prediction_error_ms: np.ndarray  # u - û [m/s] (signed)
    target_deficit_ms: np.ndarray  # U - u [m/s]
    avg_deficit_rel_error: float  # |mean(U-û)-mean(U-u)|/mean(U-u) [%]
    avg_velocity_rel_error: float  # |mean(u)-mean(û)|/mean(u) [%]
    velocity_rel_error: np.ndarray  # |u-û|/u [%]


def compute_crossstream_errors(
    targets: np.ndarray,
    predictions: np.ndarray,
    U_freestream: float,
    threshold_pct: float = 5.0,
) -> CrossstreamErrors:
    """Compute all crossstream error metrics.

    Args:
        targets: Target velocities (u) in m/s
        predictions: Predicted velocities (û) in m/s
        U_freestream: Freestream velocity (U) in m/s
        threshold_pct: Minimum deficit threshold as % of U for filtered metric

    Returns:
        CrossstreamErrors dataclass
    """
    u = targets
    u_hat = predictions
    U = U_freestream

    error_abs = np.abs(u - u_hat)
    freestream_norm = error_abs / U
    percentage = (error_abs / np.abs(u)) * 100

    target_deficit = np.abs(U - u)
    deficit_norm = np.where(target_deficit > 1e-6, (error_abs / target_deficit) * 100, np.nan)

    min_deficit = (threshold_pct / 100.0) * U
    deficit_norm_filtered = np.where(
        target_deficit > min_deficit, (error_abs / target_deficit) * 100, np.nan
    )

    prediction_error = u - u_hat

    # Average deficit relative error
    pred_deficit = U - u_hat
    avg_pred_deficit = np.mean(pred_deficit)
    avg_target_deficit = np.mean(target_deficit)
    avg_deficit_rel_error = (
        np.abs(avg_pred_deficit - avg_target_deficit) / avg_target_deficit * 100
        if avg_target_deficit > 1e-6
        else float("nan")
    )

    # Average velocity relative error
    avg_u = np.mean(u)
    avg_u_hat = np.mean(u_hat)
    avg_velocity_rel_error = (
        np.abs(avg_u - avg_u_hat) / avg_u * 100 if avg_u > 1e-6 else float("nan")
    )

    velocity_rel_error = np.where(u > 1e-6, error_abs / u * 100, np.nan)

    return CrossstreamErrors(
        freestream_normalized=freestream_norm,
        percentage=percentage,
        deficit_normalized=deficit_norm,
        deficit_normalized_filtered=deficit_norm_filtered,
        prediction_error_ms=prediction_error,
        target_deficit_ms=target_deficit,
        avg_deficit_rel_error=avg_deficit_rel_error,
        avg_velocity_rel_error=avg_velocity_rel_error,
        velocity_rel_error=velocity_rel_error,
    )


def summarize_errors(errors: CrossstreamErrors, y_D: np.ndarray | None = None) -> dict:
    """Summarize CrossstreamErrors into {metric_name: {min, max, mean, std, max_y_D}} dict."""

    def get_max_y_D(arr):
        if y_D is None or np.all(np.isnan(arr)):
            return float("nan")
        idx = np.nanargmax(arr)
        return float(y_D[idx])

    def stats(arr, name):
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "max_y_D": get_max_y_D(arr),
        }

    def scalar_stats(val):
        return {
            "min": float(val),
            "max": float(val),
            "mean": float(val),
            "std": 0.0,
            "max_y_D": float("nan"),
        }

    return {
        "freestream_normalized": stats(errors.freestream_normalized, "freestream_normalized"),
        "percentage": stats(errors.percentage, "percentage"),
        "deficit_normalized": stats(errors.deficit_normalized, "deficit_normalized"),
        "deficit_normalized_filtered": stats(
            errors.deficit_normalized_filtered, "deficit_normalized_filtered"
        ),
        "target_deficit_ms": stats(errors.target_deficit_ms, "target_deficit_ms"),
        "prediction_error_ms": stats(errors.prediction_error_ms, "prediction_error_ms"),
        "avg_deficit_rel_error": scalar_stats(errors.avg_deficit_rel_error),
        "avg_velocity_rel_error": scalar_stats(errors.avg_velocity_rel_error),
        "velocity_rel_error": stats(errors.velocity_rel_error, "velocity_rel_error"),
    }


def compute_wt_relative_errors(
    targets: np.ndarray, predictions: np.ndarray, U_freestream: float
) -> np.ndarray:
    """Compute per-WT relative errors: (targets - predictions) / U * 100 [%]."""
    return (targets - predictions) / U_freestream * 100


def aggregate_wt_errors_across_windspeeds(
    per_ws_errors: dict[float, np.ndarray],
) -> dict[str, np.ndarray]:
    """Aggregate per-windspeed WT errors into mean, std, abs_max.

    Args:
        per_ws_errors: {U: wt_rel_errors_array} dict

    Returns:
        Dict with 'mean', 'std', 'abs_max' arrays
    """
    arr = np.array(list(per_ws_errors.values()))
    return {
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "abs_max": np.max(np.abs(arr), axis=0),
    }
