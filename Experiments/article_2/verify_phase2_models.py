"""
Verify Phase 2 trained models.

Loads trained models from Phase 2 experiments and verifies that fine-tuning
techniques (freezing, LoRA, weight anchoring) were applied correctly.

Usage:
    pixi run python Experiments/article_2/verify_phase2_models.py
    pixi run python Experiments/article_2/verify_phase2_models.py \
        --phase2-base /path/to/multirun/2026-01-22/14-29-59
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU execution for loading checkpoints
os.environ["JAX_PLATFORMS"] = "cpu"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import jax.numpy as jnp  # noqa: E402
from jax import tree_util  # noqa: E402

from utils.finetuning import (
    compute_weight_anchoring_loss,
    count_lora_parameters,
    create_frozen_mask,
    log_freezing_statistics,
)
from utils.finetuning import verify_freezing as utils_verify_freezing  # noqa: E402
from utils.model_tools import load_model  # noqa: E402

# ============================================================================
# Configuration
# ============================================================================

# Search v2 first (fixed LoRA), then original (freezing experiments)
DEFAULT_PHASE2_BASES = [
    Path(
        "/home/jpsch/Documents/Sophia_work/gno/outputs/transfer_learning/phase2_test_v2"
    ),
    Path("/home/jpsch/Documents/Sophia_work/gno/outputs/transfer_learning/phase2_test"),
]
PHASE1_PRETRAINED = Path(
    "/home/jpsch/Documents/Sophia_work/gno/outputs/transfer_learning/phase1/"
    "multirun/2026-01-19/19-39-33/2_+experiment=phase1/S2_dropout_layernorm/"
    "model/checkpoints_best_mse/2961"
)
EXPERIMENT_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "configurations" / "experiment" / "phase2"
)


# ============================================================================
# Parameter comparison utilities
# ============================================================================


def extract_base_params_flat(params: dict) -> dict[str, any]:
    """Extract base parameters as flat dict with normalized paths.

    Filters out LoRA parameters and normalizes LoRADense -> Dense.
    """
    flat = {}
    for path, value in tree_util.tree_leaves_with_path(params):
        parts = [
            str(k.key if hasattr(k, "key") else k).replace("LoRADense_", "Dense_")
            for k in path
        ]
        path_str = "/".join(parts)
        if "lora_a" not in path_str and "lora_b" not in path_str:
            flat[path_str] = value
    return flat


def compare_base_weights(pretrained: dict, finetuned: dict, tol: float = 1e-6) -> dict:
    """Compare base weights between pretrained and fine-tuned models.

    Returns counts of frozen vs changed parameters.
    """
    pre_flat = extract_base_params_flat(pretrained)
    ft_flat = extract_base_params_flat(finetuned)

    frozen, changed, violations = 0, 0, []
    for path, pre_val in pre_flat.items():
        if path not in ft_flat:
            continue
        ft_val = ft_flat[path]
        diff = float(jnp.max(jnp.abs(ft_val - pre_val)))
        size = int(pre_val.size)
        if diff < tol:
            frozen += size
        else:
            changed += size
            if len(violations) < 5:
                violations.append({"path": path, "max_diff": diff})

    total = frozen + changed
    return {
        "frozen": frozen,
        "changed": changed,
        "total": total,
        "ratio": f"{100 * frozen / total:.1f}%" if total > 0 else "N/A",
        "is_valid": changed == 0,
        "violations": violations,
    }


def compute_l2_diff(pretrained: dict, finetuned: dict) -> float:
    """Compute L2 difference between base weights."""
    pre_flat = extract_base_params_flat(pretrained)
    ft_flat = extract_base_params_flat(finetuned)
    total = 0.0
    for path, pre_val in pre_flat.items():
        if path in ft_flat:
            total += float(jnp.sum((ft_flat[path] - pre_val) ** 2))
    return total


# ============================================================================
# Technique loading
# ============================================================================


def load_techniques_from_configs() -> dict:
    """Load technique definitions from YAML config files."""
    if not EXPERIMENT_CONFIG_DIR.exists():
        return {}

    techniques = {}
    for config_file in sorted(EXPERIMENT_CONFIG_DIR.glob("*.yaml")):
        name = config_file.stem
        try:
            content = config_file.read_text()
            # Extract description from comment
            desc = name
            for line in content.split("\n"):
                if line.startswith("# Phase 2") and ":" in line:
                    desc = line.split(":", 1)[1].strip()
                    break

            # Parse defaults
            techniques[name] = {
                "description": desc,
                "lora": "/finetuning/lora:" in content
                and "disabled"
                not in content.split("/finetuning/lora:")[1].split("\n")[0],
                "freezing": (
                    "/finetuning/freezing:" in content
                    and "disabled"
                    not in content.split("/finetuning/freezing:")[1].split("\n")[0]
                    if "/finetuning/freezing:" in content
                    else False
                ),
                "anchoring": (
                    "/finetuning/weight_anchoring:" in content
                    and "disabled"
                    not in content.split("/finetuning/weight_anchoring:")[1].split(
                        "\n"
                    )[0]
                    if "/finetuning/weight_anchoring:" in content
                    else False
                ),
            }
        except Exception as e:
            logger.warning(f"Failed to parse {config_file}: {e}")
    return techniques


# ============================================================================
# Checkpoint discovery
# ============================================================================


def find_latest_multirun(base_dir: Path) -> Path | None:
    """Find the latest multirun directory."""
    multirun_dir = base_dir / "multirun"
    if not multirun_dir.exists():
        return None
    for date_dir in sorted(multirun_dir.iterdir(), reverse=True):
        if date_dir.is_dir():
            for time_dir in sorted(date_dir.iterdir(), reverse=True):
                if time_dir.is_dir():
                    return time_dir
    return None


def find_checkpoint(base_dirs: list[Path], technique: str) -> Path | None:
    """Find checkpoint for a technique across multiple directories."""
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for exp_dir in base_dir.iterdir():
            if not exp_dir.is_dir() or "+experiment=phase2" not in exp_dir.name:
                continue
            for tech_dir in exp_dir.iterdir():
                if tech_dir.is_dir() and technique in tech_dir.name:
                    for ckpt in tech_dir.glob("**/model/checkpoints_best_mse"):
                        for step in ckpt.iterdir():
                            if step.is_dir():
                                if (step / "default" / "_METADATA").exists():
                                    return step
                                if (step / "_METADATA").exists():
                                    return step
    return None


# ============================================================================
# Verification functions
# ============================================================================


def check_freezing_technique(pretrained: dict, phase2: dict, cfg) -> dict:
    """Check that frozen parameters match pretrained."""
    if not hasattr(cfg, "finetuning"):
        return {"status": "N/A"}

    freeze_cfg = cfg.finetuning.get("freezing", {})
    if not freeze_cfg.get("enabled", False):
        return {"status": "N/A"}

    lora_enabled = cfg.finetuning.get("lora", {}).get("enabled", False)

    if lora_enabled:
        # LoRA-aware comparison
        cmp = compare_base_weights(pretrained, phase2)
        return {
            "status": "PASS" if cmp["is_valid"] else "FAIL",
            "freeze_ratio": cmp["ratio"],
            "violations": cmp["violations"],
        }

    # Standard comparison
    try:
        mask = create_frozen_mask(phase2, dict(freeze_cfg))
        valid = utils_verify_freezing(pretrained, phase2, mask)
        stats = log_freezing_statistics(phase2, mask)
        return {
            "status": "PASS" if valid else "FAIL",
            "freeze_ratio": f"{100 * stats['freeze_ratio']:.1f}%",
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def check_lora_technique(pretrained: dict, phase2: dict, cfg) -> dict:
    """Check LoRA parameters exist and base weights remained frozen."""
    if not hasattr(cfg, "finetuning"):
        return {"status": "N/A"}

    lora_cfg = cfg.finetuning.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return {"status": "N/A"}

    try:
        counts = count_lora_parameters(phase2)
        if counts["lora_params"] == 0:
            return {"status": "FAIL", "message": "No LoRA params found"}

        cmp = compare_base_weights(pretrained, phase2)
        return {
            "status": "PASS" if cmp["is_valid"] else "FAIL",
            "lora_ratio": f"{100 * counts['lora_ratio']:.2f}%",
            "lora_params": counts["lora_params"],
            "base_weights_frozen": cmp["is_valid"],
            "base_frozen_ratio": cmp["ratio"],
            "base_violations": cmp["violations"],
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def check_anchoring_technique(pretrained: dict, phase2: dict, cfg) -> dict:
    """Check weight anchoring loss value."""
    if not hasattr(cfg, "finetuning"):
        return {"status": "N/A"}

    anchor_cfg = cfg.finetuning.get("weight_anchoring", {})
    if not anchor_cfg.get("enabled", False):
        return {"status": "N/A"}

    try:
        lmbda = anchor_cfg.get("lambda_anchor", 0.01)
        lora_enabled = cfg.finetuning.get("lora", {}).get("enabled", False)

        if lora_enabled:
            l2_diff = compute_l2_diff(pretrained, phase2)
            return {"status": "PASS", "anchor_loss": lmbda * l2_diff}

        loss = compute_weight_anchoring_loss(phase2, pretrained, lmbda)
        return {"status": "PASS", "anchor_loss": float(loss)}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


# ============================================================================
# Output formatting
# ============================================================================


def format_status(result: dict, key: str) -> tuple[str, str]:
    """Format a verification result for display. Returns (main_str, extra_str)."""
    r = result.get(key, {})
    status = r.get("status", "N/A")

    if status == "N/A":
        return "N/A", "N/A"
    if status == "ERROR":
        return "ERROR", "-"
    if status == "FAIL":
        if key == "lora":
            return "FAIL", f"CHANGED ({r.get('base_frozen_ratio', '?')})"
        return "FAIL", "-"

    # PASS cases
    if key == "freezing":
        return f"PASS ({r.get('freeze_ratio', '?')})", "-"
    if key == "lora":
        return (
            f"PASS ({r.get('lora_ratio', '?')})",
            f"Frozen ({r.get('base_frozen_ratio', '?')})",
        )
    if key == "anchoring":
        loss = r.get("anchor_loss")
        return f"{loss:.4f}" if loss is not None else "PASS", "-"
    return "PASS", "-"


def print_summary_table(results: dict) -> bool:
    """Print formatted results table. Returns True if all passed."""
    print("\n" + "=" * 120)
    print("Phase 2 Model Verification Results")
    print("=" * 120)
    print(
        f"{'Technique':<30} {'Freezing':<15} {'LoRA':<15} {'LoRA Base':<18} {'Anchoring':<15} {'Status':<10}"
    )
    print("-" * 120)

    all_pass = True
    for technique, result in results.items():
        if "error" in result:
            print(
                f"{technique:<30} {'ERROR':<15} {'-':<15} {'-':<18} {'-':<15} {'FAIL':<10}"
            )
            all_pass = False
            continue

        freeze_str, _ = format_status(result, "freezing")
        lora_str, lora_base = format_status(result, "lora")
        anchor_str, _ = format_status(result, "anchoring")

        # Overall status
        statuses = [
            result[k].get("status", "N/A") for k in ["freezing", "lora", "anchoring"]
        ]
        passed = not any(s in ["FAIL", "ERROR"] for s in statuses)
        if not passed:
            all_pass = False

        print(
            f"{technique:<30} {freeze_str:<15} {lora_str:<15} {lora_base:<18} {anchor_str:<15} {'PASS' if passed else 'FAIL':<10}"
        )

    print("=" * 120)
    print(
        "All techniques verified successfully!"
        if all_pass
        else "Some techniques failed verification."
    )
    return all_pass


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 2 trained models")
    parser.add_argument("--phase2-base", type=Path, help="Phase 2 multirun directory")
    parser.add_argument("--phase1-checkpoint", type=Path, default=PHASE1_PRETRAINED)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--techniques", nargs="+", help="Specific techniques to verify")
    args = parser.parse_args()

    # Load technique definitions
    techniques = load_techniques_from_configs()
    if not techniques:
        logger.error("No technique configs found")
        sys.exit(1)
    logger.info(f"Loaded {len(techniques)} techniques from configs")

    # Find Phase 2 directories
    if args.phase2_base:
        phase2_bases = [args.phase2_base]
    else:
        phase2_bases = [
            m for b in DEFAULT_PHASE2_BASES if (m := find_latest_multirun(b))
        ]
        if not phase2_bases:
            logger.error("No multirun directories found")
            sys.exit(1)
    logger.info(f"Phase 2 directories: {[str(p) for p in phase2_bases]}")

    # Load pretrained model
    if not args.phase1_checkpoint.exists():
        logger.error(f"Phase 1 checkpoint not found: {args.phase1_checkpoint}")
        sys.exit(1)

    logger.info("Loading Phase 1 pretrained model...")
    try:
        _, pretrained_params, _, _ = load_model(str(args.phase1_checkpoint))
    except Exception as e:
        logger.error(f"Failed to load Phase 1 model: {e}")
        sys.exit(1)

    # Filter techniques
    to_verify = args.techniques if args.techniques else list(techniques.keys())
    invalid = [t for t in to_verify if t not in techniques]
    if invalid:
        logger.error(f"Unknown techniques: {invalid}")
        sys.exit(1)

    # Verify each technique
    results = {}
    for technique in to_verify:
        logger.info(f"\nVerifying {technique}...")

        ckpt = find_checkpoint(phase2_bases, technique)
        if not ckpt:
            logger.warning(f"  Checkpoint not found")
            results[technique] = {"error": "Checkpoint not found"}
            continue

        try:
            logger.info(f"  Loading: {ckpt}")
            _, params, _, cfg = load_model(str(ckpt))

            # # Print params for LoRA techniques (debug)
            # if "lora" in technique.lower():
            #     print(params)
            #     print(pretrained_params)

            results[technique] = {
                "checkpoint": str(ckpt),
                "freezing": check_freezing_technique(pretrained_params, params, cfg),
                "lora": check_lora_technique(pretrained_params, params, cfg),
                "anchoring": check_anchoring_technique(pretrained_params, params, cfg),
            }
            logger.info("  Done")
        except Exception as e:
            logger.error(f"  Error: {e}")
            results[technique] = {"error": str(e)}

    # Print and save results
    all_pass = print_summary_table(results) if results else False

    output_path = (
        args.output or Path(__file__).parent / "outputs" / "phase2_verification.json"
    )
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
