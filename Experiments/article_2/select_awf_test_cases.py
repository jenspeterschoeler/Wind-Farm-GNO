"""
AWF test case selection script for Article 2.

Classifies AWF test farms by size (small/medium/large), finds flowcases at
low/medium/high wind speed + TI combinations, and produces overview plots
for manual selection.

Usage:
    cd /home/jpsch/code/gno
    python Experiments/article_2/select_awf_test_cases.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from article2_utils import get_awf_test_path  # noqa: E402

from utils.data_tools import retrieve_dataset_stats  # noqa: E402
from utils.plotting import matplotlib_set_rcparams  # noqa: E402
from utils.torch_loader import Torch_Geomtric_Dataset  # noqa: E402

# Output directory
FIGURES_DIR = Path(__file__).parent / "figures" / "awf_case_selection"


# =============================================================================
# Step 1: Load and catalog all AWF test graphs
# =============================================================================


def catalog_test_graphs(dataset, scale_stats: dict) -> pd.DataFrame:
    """Iterate all test graphs and record properties in a DataFrame."""
    vel_range = scale_stats["velocity"]["range"]
    if isinstance(vel_range, list):
        vel_range = vel_range[0]
    ti_range = scale_stats["ti"]["range"]
    if isinstance(ti_range, list):
        ti_range = ti_range[0]
    dist_range = scale_stats["distance"]["range"]
    if isinstance(dist_range, list):
        dist_range = dist_range[0]

    records = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        n_wt = int(data.n_wt) if hasattr(data, "n_wt") else int(data.pos.shape[0])
        # run4 scaling: scaled = value / range, so unscale = scaled * range
        ws_inf = data.global_features[0].item() * vel_range
        ti_inf = data.global_features[1].item() * ti_range
        layout_group = idx // 4

        records.append(
            {
                "idx": idx,
                "n_wt": n_wt,
                "ws_inf": ws_inf,
                "ti_inf": ti_inf,
                "layout_group": layout_group,
            }
        )

    df = pd.DataFrame(records)
    print(f"Cataloged {len(df)} test graphs ({df['layout_group'].nunique()} layouts)")
    print(f"  n_wt range: [{df['n_wt'].min()}, {df['n_wt'].max()}]")
    print(f"  ws_inf range: [{df['ws_inf'].min():.2f}, {df['ws_inf'].max():.2f}] m/s")
    print(f"  ti_inf range: [{df['ti_inf'].min():.4f}, {df['ti_inf'].max():.4f}]")
    return df


# =============================================================================
# Step 2: Classify farms by size using terciles
# =============================================================================


def classify_farm_sizes(df: pd.DataFrame, large_min: int = 110) -> pd.DataFrame:
    """Assign size_category based on n_wt thresholds.

    Uses the median of sub-large layouts as the small/medium boundary,
    and ``large_min`` as the medium/large boundary.
    """
    layout_nwt = df.groupby("layout_group")["n_wt"].first()
    t2 = large_min
    # Split small/medium at median of layouts below the large threshold
    sub_large = layout_nwt[layout_nwt < t2]
    t1 = int(sub_large.median()) if len(sub_large) > 0 else t2 // 2
    print(f"\nFarm size thresholds: small <= {t1}, medium <= {t2}, large > {t2}")

    def categorize(n_wt):
        if n_wt <= t1:
            return "small"
        elif n_wt < t2:
            return "medium"
        else:
            return "large"

    df["size_category"] = df["n_wt"].apply(categorize)

    for cat in ["small", "medium", "large"]:
        n_layouts = df.loc[df["size_category"] == cat, "layout_group"].nunique()
        n_wt_range = df.loc[df["size_category"] == cat, "n_wt"]
        print(f"  {cat}: {n_layouts} layouts, n_wt in [{n_wt_range.min()}, {n_wt_range.max()}]")

    return df


# =============================================================================
# Step 3: Classify inflow conditions using terciles
# =============================================================================


def classify_inflow(df: pd.DataFrame) -> pd.DataFrame:
    """Assign inflow_category based on WS and TI terciles."""
    ws_t1 = df["ws_inf"].quantile(1 / 3)
    ws_t2 = df["ws_inf"].quantile(2 / 3)
    ti_t1 = df["ti_inf"].quantile(1 / 3)
    ti_t2 = df["ti_inf"].quantile(2 / 3)
    print("\nInflow tercile thresholds:")
    print(f"  WS: t1={ws_t1:.2f}, t2={ws_t2:.2f} m/s")
    print(f"  TI: t1={ti_t1:.4f}, t2={ti_t2:.4f}")

    def ws_cat(ws):
        if ws <= ws_t1:
            return "low"
        elif ws <= ws_t2:
            return "med"
        else:
            return "high"

    def ti_cat(ti):
        if ti <= ti_t1:
            return "low"
        elif ti <= ti_t2:
            return "med"
        else:
            return "high"

    df["ws_category"] = df["ws_inf"].apply(ws_cat)
    df["ti_category"] = df["ti_inf"].apply(ti_cat)
    df["inflow_category"] = df["ws_category"] + "_ws_" + df["ti_category"] + "_ti"

    # Store thresholds as metadata on the DataFrame
    df.attrs["ws_terciles"] = (ws_t1, ws_t2)
    df.attrs["ti_terciles"] = (ti_t1, ti_t2)

    return df


# =============================================================================
# Step 4: Select candidates
# =============================================================================


def select_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Select 9 cases: 3 sizes x 3 diagonal inflow targets (low-low, med-med, high-high)."""
    # Compute target centroids for each diagonal inflow category
    inflow_targets = {
        "low_ws_low_ti": ("low", "low"),
        "med_ws_med_ti": ("med", "med"),
        "high_ws_high_ti": ("high", "high"),
    }

    # Normalize ws and ti to [0, 1] for distance calculation
    ws_min, ws_max = df["ws_inf"].min(), df["ws_inf"].max()
    ti_min, ti_max = df["ti_inf"].min(), df["ti_inf"].max()
    ws_span = ws_max - ws_min if ws_max > ws_min else 1.0
    ti_span = ti_max - ti_min if ti_max > ti_min else 1.0

    df["ws_norm"] = (df["ws_inf"] - ws_min) / ws_span
    df["ti_norm"] = (df["ti_inf"] - ti_min) / ti_span

    # Compute centroids in normalized space for each diagonal category
    centroids = {}
    for name, (ws_label, ti_label) in inflow_targets.items():
        mask = (df["ws_category"] == ws_label) & (df["ti_category"] == ti_label)
        subset = df[mask]
        if len(subset) > 0:
            centroids[name] = (subset["ws_norm"].mean(), subset["ti_norm"].mean())
        else:
            # Fallback: use tercile midpoints
            ws_t1, ws_t2 = df.attrs["ws_terciles"]
            ti_t1, ti_t2 = df.attrs["ti_terciles"]
            midpoints = {
                "low": 0.0,
                "med": 0.5,
                "high": 1.0,
            }
            centroids[name] = (midpoints[ws_label], midpoints[ti_label])

    # Select best match for each (size, inflow) combination
    selected = []
    for size_cat in ["small", "medium", "large"]:
        size_mask = df["size_category"] == size_cat
        for inflow_name, centroid in centroids.items():
            subset = df[size_mask].copy()
            if len(subset) == 0:
                print(f"  Warning: No graphs for {size_cat} / {inflow_name}")
                continue
            # Euclidean distance in normalized (ws, ti) space
            subset = subset.copy()
            subset["dist"] = np.sqrt(
                (subset["ws_norm"] - centroid[0]) ** 2 + (subset["ti_norm"] - centroid[1]) ** 2
            )
            best = subset.loc[subset["dist"].idxmin()]
            selected.append(
                {
                    "size_category": size_cat,
                    "inflow_target": inflow_name,
                    "idx": int(best["idx"]),
                    "n_wt": int(best["n_wt"]),
                    "ws_inf": best["ws_inf"],
                    "ti_inf": best["ti_inf"],
                    "layout_group": int(best["layout_group"]),
                    "dist_to_centroid": best["dist"],
                }
            )

    selected_df = pd.DataFrame(selected)
    print("\nSelected 9 test cases:")
    print(selected_df.to_string(index=False))
    return selected_df


# =============================================================================
# Step 5: Plots
# =============================================================================

def _get_size_colors():
    import cmcrameri.cm as cmc

    _batlow = cmc.batlow
    return {"small": _batlow(0.15), "medium": _batlow(0.5), "large": _batlow(0.85)}
INFLOW_SHORT = {
    "low_ws_low_ti": "Low",
    "med_ws_med_ti": "Med",
    "high_ws_high_ti": "High",
}


def plot_overview_scatter(df: pd.DataFrame, selected_df: pd.DataFrame, output_dir: Path):
    """All test points in (WS, TI) space, colored by farm size, 9 selected highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    SIZE_COLORS = _get_size_colors()
    matplotlib_set_rcparams("paper")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Background scatter: all test points, bubble area proportional to n_wt
    # Scale so smallest farm ~15pt and largest ~300pt
    n_min, n_max = df["n_wt"].min(), df["n_wt"].max()
    s_min, s_max = 15, 300

    for size_cat, color in SIZE_COLORS.items():
        mask = df["size_category"] == size_cat
        subset = df[mask]
        sizes = s_min + (subset["n_wt"] - n_min) / (n_max - n_min) * (s_max - s_min)
        ax.scatter(
            subset["ws_inf"],
            subset["ti_inf"],
            color=color,
            s=sizes,
            alpha=0.4,
            edgecolors="none",
        )

    # Tercile boundary lines
    ws_t1, ws_t2 = df.attrs["ws_terciles"]
    ti_t1, ti_t2 = df.attrs["ti_terciles"]
    for ws_t in [ws_t1, ws_t2]:
        ax.axvline(ws_t, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    for ti_t in [ti_t1, ti_t2]:
        ax.axhline(ti_t, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # Tighten axis limits to data range with small padding
    ws_pad = 0.04 * (df["ws_inf"].max() - df["ws_inf"].min())
    ti_pad = 0.06 * (df["ti_inf"].max() - df["ti_inf"].min())
    ax.set_xlim(df["ws_inf"].min() - ws_pad, df["ws_inf"].max() + ws_pad)
    ax.set_ylim(df["ti_inf"].min() - ti_pad, df["ti_inf"].max() + ti_pad)

    # Shade the 3 diagonal bins (low-low, med-med, high-high)
    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()
    diag_bins = [
        ("Low", ax_xmin, ax_ymin, ws_t1 - ax_xmin, ti_t1 - ax_ymin, "left", "top", 0.04, 0.96),
        (
            "Medium",
            ws_t1,
            ti_t1,
            ws_t2 - ws_t1,
            ti_t2 - ti_t1,
            "right",
            "bottom",
            0.96,
            0.04,
        ),
        ("High", ws_t2, ti_t2, ax_xmax - ws_t2, ax_ymax - ti_t2, "left", "top", 0.04, 0.96),
    ]
    for label, x0, y0, w, h, ha, va, fx, fy in diag_bins:
        ax.add_patch(
            Rectangle((x0, y0), w, h, facecolor="gray", alpha=0.12, edgecolor="none", zorder=0)
        )
        ax.text(
            x0 + fx * w,
            y0 + fy * h,
            label,
            ha=ha,
            va=va,
            fontsize=7,
            color="dimgray",
            fontstyle="italic",
            fontweight="bold",
        )

    # Mark centroids of each diagonal bin (the selection target)
    inflow_bins = {
        "low": (df["ws_category"] == "low") & (df["ti_category"] == "low"),
        "med": (df["ws_category"] == "med") & (df["ti_category"] == "med"),
        "high": (df["ws_category"] == "high") & (df["ti_category"] == "high"),
    }
    for _name, mask in inflow_bins.items():
        subset = df[mask]
        if len(subset) > 0:
            cx, cy = subset["ws_inf"].mean(), subset["ti_inf"].mean()
            ax.plot(
                cx, cy, marker="+", color="dimgray", markersize=8, markeredgewidth=1.2, zorder=2
            )

    # Highlight selected cases with stars and manually placed labels
    ann_data = []
    for _, row in selected_df.iterrows():
        color = SIZE_COLORS[row["size_category"]]
        ax.scatter(
            row["ws_inf"],
            row["ti_inf"],
            marker="*",
            s=250,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ann_data.append(
            {
                "x": row["ws_inf"],
                "y": row["ti_inf"],
                "nwt": int(row["n_wt"]),
                "color": color,
                "size": row["size_category"],
            }
        )

    # Manual offsets keyed by (size_category, inflow_target) for problem clusters
    manual_offsets = {}
    # Bottom-left cluster (~8 m/s, TI~0.06-0.07): green left, blue back to default
    manual_offsets[("large", "low_ws_low_ti")] = (-45, -8)
    manual_offsets[("small", "low_ws_low_ti")] = (8, 8)
    manual_offsets[("medium", "low_ws_low_ti")] = (10, 8)
    # Middle cluster: nudge n_wt=120 label up a bit
    manual_offsets[("large", "med_ws_med_ti")] = (8, 14)
    # Upper-right cluster: large below-left to stay inside plot, orange right
    manual_offsets[("large", "high_ws_high_ti")] = (-50, -10)
    manual_offsets[("medium", "high_ws_high_ti")] = (8, 6)

    for i, d in enumerate(ann_data):
        row_match = selected_df.iloc[i]
        key = (row_match["size_category"], row_match["inflow_target"])
        ox, oy = manual_offsets.get(key, (8, 8))
        ax.annotate(
            f"$n_{{\\mathrm{{wt}}}}$={d['nwt']}",
            (d["x"], d["y"]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=7,
            color=d["color"],
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
            },
        )

    ax.set_xlabel(r"$U_\infty$ [ms$^{-1}$]")
    ax.set_ylabel(r"$I_0$ [-]")

    # Two-row legend: row 1 = size category, row 2 = farm size + markers
    size_labels = {"small": "S", "medium": "M", "large": "L"}

    # Row 1: Size category only
    row1_handles = [Patch(facecolor="none", edgecolor="none", label="Size category:")]
    row1_handles += [Patch(facecolor=c, label=size_labels[s]) for s, c in SIZE_COLORS.items()]

    # Row 2: Farm size + markers
    row2_handles = [Patch(facecolor="none", edgecolor="none", label="Farm size:")]
    ref_nwts = [20, 80, 150]
    ref_sizes = [s_min + (n - n_min) / (n_max - n_min) * (s_max - s_min) for n in ref_nwts]
    for n, sz in zip(ref_nwts, ref_sizes):
        row2_handles.append(
            ax.scatter(
                [],
                [],
                s=sz,
                c="gray",
                alpha=0.5,
                edgecolors="none",
                label=f"$n_{{\\mathrm{{wt}}}}$={n}",
            )
        )
    row2_handles.append(Patch(facecolor="none", edgecolor="none", label=" Markers:"))
    row2_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="*",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="Selected",
        )
    )
    row2_handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="+",
            color="dimgray",
            markersize=7,
            markeredgewidth=1.2,
            label="Centroid",
        )
    )

    # Pad shorter row to match longer row
    ncol = max(len(row1_handles), len(row2_handles))
    while len(row1_handles) < ncol:
        row1_handles.append(Patch(facecolor="none", edgecolor="none", label=""))
    while len(row2_handles) < ncol:
        row2_handles.append(Patch(facecolor="none", edgecolor="none", label=""))

    # Interleave rows so column-first layout produces correct visual rows
    interleaved = [h for pair in zip(row1_handles, row2_handles) for h in pair]
    ax.legend(
        handles=interleaved,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        fontsize=6.5,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.5,
        scatterpoints=1,
        numpoints=1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out_path = output_dir / "overview_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_layout_grid(
    dataset,
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
    scale_stats: dict,
    output_dir: Path,
):
    """3x3 grid of turbine layouts with available inflow conditions listed."""
    import matplotlib.pyplot as plt

    SIZE_COLORS = _get_size_colors()
    matplotlib_set_rcparams("paper")
    dist_range = scale_stats["distance"]["range"]
    if isinstance(dist_range, list):
        dist_range = dist_range[0]

    fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    size_order = ["small", "medium", "large"]
    inflow_order = ["low_ws_low_ti", "med_ws_med_ti", "high_ws_high_ti"]

    # Add column titles via figure text
    col_titles = [INFLOW_SHORT[inf] + " inflow" for inf in inflow_order]
    fig.subplots_adjust(top=0.91)
    for col_i, title in enumerate(col_titles):
        x_center = axes[0, col_i].get_position().x0 + axes[0, col_i].get_position().width / 2
        fig.text(x_center, 0.95, title, ha="center", fontsize=10, fontweight="bold")

    for row_i, size_cat in enumerate(size_order):
        for col_i, inflow_target in enumerate(inflow_order):
            ax = axes[row_i, col_i]
            match = selected_df[
                (selected_df["size_category"] == size_cat)
                & (selected_df["inflow_target"] == inflow_target)
            ]
            if len(match) == 0:
                ax.set_visible(False)
                continue

            case = match.iloc[0]
            data = dataset[int(case["idx"])]
            # Unscale positions from run4: pos_m = pos_scaled * dist_range
            pos = data.pos.numpy() * dist_range
            # Convert to km for readability
            pos_km = pos / 1000.0

            ax.scatter(
                pos_km[:, 0],
                pos_km[:, 1],
                marker="^",
                s=30,
                color=SIZE_COLORS[size_cat],
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )
            ax.set_aspect("equal")
            ax.set_title(f"n$_{{wt}}$={int(case['n_wt'])}", fontsize=9)
            ax.tick_params(labelsize=7)

            # List all 4 flowcases for this layout
            layout_g = int(case["layout_group"])
            flowcases = df[df["layout_group"] == layout_g].sort_values("ws_inf")
            fc_lines = [
                f"WS={r['ws_inf']:.1f}, TI={r['ti_inf']:.3f}" for _, r in flowcases.iterrows()
            ]
            fc_text = "\n".join(fc_lines)
            ax.text(
                0.02,
                0.02,
                fc_text,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="bottom",
                fontfamily="monospace",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "gray",
                    "linewidth": 0.5,
                },
            )

            # Row/column labels
            if col_i == 0:
                ax.set_ylabel(f"{size_cat.capitalize()}\ny [km]", fontsize=9)
            else:
                ax.set_ylabel("")
            if row_i == 2:
                ax.set_xlabel("x [km]", fontsize=9)
            else:
                ax.set_xlabel("")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_dir / "layout_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# Step 6: JSON export
# =============================================================================


def build_index_mapping(test_data_path: str) -> list[int]:
    """Build mapping from alphabetically-sorted zip position to NetCDF layout index.

    torch_loader sorts zip files alphabetically, so _layout105.zip comes before
    _layout11.zip. This function replicates that ordering and extracts the layout
    index from each filename.

    Returns list where position i contains the NetCDF layout index for the i-th zip.
    """
    zip_files = sorted(f for f in os.listdir(test_data_path) if f.endswith(".zip"))
    layout_indices = []
    for fname in zip_files:
        match = re.search(r"_layout(\d+)\.zip$", fname)
        if match:
            layout_indices.append(int(match.group(1)))
        else:
            raise ValueError(f"Cannot extract layout index from: {fname}")
    return layout_indices


def save_selected_cases_json(selected_df: pd.DataFrame, test_data_path: str, output_dir: Path):
    """Export selected cases as JSON with both dataset and NetCDF indices."""
    layout_mapping = build_index_mapping(test_data_path)

    cases = []
    for _, row in selected_df.iterrows():
        dataset_idx = int(row["idx"])
        zip_position = dataset_idx // 4
        flowcase_idx = dataset_idx % 4
        netcdf_layout_idx = layout_mapping[zip_position]

        cases.append(
            {
                "size_category": row["size_category"],
                "inflow_target": row["inflow_target"],
                "dataset_idx": dataset_idx,
                "netcdf_layout_idx": netcdf_layout_idx,
                "netcdf_flowcase_idx": flowcase_idx,
                "n_wt": int(row["n_wt"]),
                "ws_inf": round(float(row["ws_inf"]), 2),
                "ti_inf": round(float(row["ti_inf"]), 4),
            }
        )

    output = {
        "description": "Selected AWF test cases for Article 2",
        "n_cases": len(cases),
        "cases": cases,
    }

    out_path = output_dir / "selected_cases.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("AWF Test Case Selection")
    print("=" * 60)

    # Load dataset
    test_data_path = get_awf_test_path()
    print(f"\nTest data path: {test_data_path}")
    dataset = Torch_Geomtric_Dataset(test_data_path, in_mem=False)
    print(f"Loaded {len(dataset)} test samples")

    _, scale_stats = retrieve_dataset_stats(dataset)

    # Step 1: Catalog
    print("\n--- Step 1: Catalog test graphs ---")
    df = catalog_test_graphs(dataset, scale_stats)

    # Step 2: Classify farm sizes
    print("\n--- Step 2: Classify farm sizes ---")
    df = classify_farm_sizes(df)

    # Step 3: Classify inflow
    print("\n--- Step 3: Classify inflow conditions ---")
    df = classify_inflow(df)

    # Step 4: Select candidates
    print("\n--- Step 4: Select candidates ---")
    selected_df = select_candidates(df)

    # Step 5: Generate plots
    print("\n--- Step 5: Generate plots ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_overview_scatter(df, selected_df, FIGURES_DIR)
    plot_layout_grid(dataset, df, selected_df, scale_stats, FIGURES_DIR)

    # Step 6: Export JSON
    print("\n--- Step 6: Export selected cases JSON ---")
    save_selected_cases_json(selected_df, test_data_path, FIGURES_DIR)

    print(f"\nAll outputs saved to: {FIGURES_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
