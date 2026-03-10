"""Calculate the center of wind farms from mini_graphs dataset.

Only wind turbine positions are included in the center calculation.
"""

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from utils.torch_loader import Torch_Geomtric_Dataset


def calculate_windfarm_center(data):
    """Calculate the center of a wind farm from graph data.

    Args:
        data: PyTorch Geometric Data object with 'pos' and 'n_wt' attributes.

    Returns:
        Tensor of shape (2,) containing the (x, y) center coordinates.
    """
    n_wt = data.n_wt
    wt_positions = data.pos[:n_wt]
    center = wt_positions.mean(dim=0)
    return center


def main():
    # Load mini_graphs test dataset
    data_path = repo_root / "data" / "mini_graphs" / "test_pre_processed"
    dataset = Torch_Geomtric_Dataset(str(data_path), in_mem=False)

    print(f"Loaded dataset with {len(dataset)} graphs")
    print("-" * 60)

    all_centers = []

    for idx, data in enumerate(dataset):
        center = calculate_windfarm_center(data)
        all_centers.append(center)

        print(f"Graph {idx:3d}: n_wt={data.n_wt:3d}, center=({center[0]:8.4f}, {center[1]:8.4f})")

    # Stack all centers and compute statistics
    all_centers = torch.stack(all_centers)

    print("-" * 60)
    print(f"Mean center across all farms: ({all_centers[:, 0].mean():8.4f}, {all_centers[:, 1].mean():8.4f})")
    print(f"Std of centers:               ({all_centers[:, 0].std():8.4f}, {all_centers[:, 1].std():8.4f})")


if __name__ == "__main__":
    main()
