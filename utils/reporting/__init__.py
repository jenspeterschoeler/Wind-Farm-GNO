"""
Reporting utilities for experiment analysis.

This module provides tools for post-hoc analysis of experiments:
- WandB data fetching with CSV caching
- LaTeX table generation with proper formatting
- CSV export for data analysis
- Sophia HPC path mapping utilities

These complement existing utilities:
- utils/plotting.py: Matplotlib visualization (graphs, training curves, contours)
- utils/training_utils.py: Training-time WandB logging
"""

from utils.reporting.latex_tables import (
    create_comparison_table,
    create_latex_table,
    highlight_best,
    save_dataframe_csv,
    save_table,
)
from utils.reporting.sophia_paths import (
    SOPHIA_LOCAL_MOUNT,
    SOPHIA_REMOTE_PREFIX,
    convert_wandb_path_to_local,
    find_checkpoint_step_dir,
    get_checkpoint_paths,
    local_path_to_sophia_path,
)
from utils.reporting.wandb_fetch import (
    WANDB_PROJECTS,
    enrich_with_best_metrics,
    extract_config,
    extract_convergence_from_history,
    extract_summary_metrics,
    fetch_wandb_runs,
    get_run_history,
    load_test_metrics,
)

__all__ = [
    # WandB fetching
    "fetch_wandb_runs",
    "enrich_with_best_metrics",
    "extract_summary_metrics",
    "extract_config",
    "get_run_history",
    "WANDB_PROJECTS",
    "load_test_metrics",
    "extract_convergence_from_history",
    # LaTeX tables
    "create_latex_table",
    "save_table",
    "highlight_best",
    # CSV export
    "save_dataframe_csv",
    "create_comparison_table",
    # Sophia path utilities
    "SOPHIA_REMOTE_PREFIX",
    "SOPHIA_LOCAL_MOUNT",
    "convert_wandb_path_to_local",
    "local_path_to_sophia_path",
    "find_checkpoint_step_dir",
    "get_checkpoint_paths",
]
