"""
Plotting functions for Article 2 publication figures.

All plotting functions take pre-computed data dicts (no model calls).
Uses the same figure structure as Article 1 but for TurbOPark dataset.

Usage:
    from article2_plotting import plot_crossstream_figure, plot_wt_spatial_errors_aggregated
"""

import csv
import math
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils.plotting import matplotlib_set_rcparams, plot_probe_graph_fn

# =============================================================================
# Crossstream Profile Figure
# =============================================================================


def plot_crossstream_figure(
    cache_data: dict,
    output_path: str,
    model_type_str: str = "best_model",
    per_row_legend: bool = False,
):
    """Generate the crossstream profiles publication figure.

    Args:
        cache_data: Dictionary with cached crossstream prediction data containing:
            - metadata: {x_downstream, U_free, ...}
            - global_limits: {velocity_range}
            - layout_data: {layout_name: {predictions: {x: {U: {...}}}, wt_mask, wt_spacing}}
        output_path: Path to save the figure
        model_type_str: Model type string for filename
        per_row_legend: If True, show per-row mini-legends with U and TI in farm layout
            panels instead of wind speed colors in the shared legend (for AWF figures).
    """
    matplotlib_set_rcparams("paper")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    metadata = cache_data["metadata"]
    global_limits = cache_data["global_limits"]
    layout_data = cache_data["layout_data"]

    x_downstream = metadata["x_downstream"]
    velocity_range = global_limits["velocity_range"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

    layout_names = list(layout_data.keys())

    shrink_factors = {
        "single string": 0.70,
        "multiple string": 0.70,
    }

    # Pre-calculate layout-specific bounds
    layout_bounds = {}
    for layout_name in layout_names:
        data = layout_data[layout_name]
        # Get actual wind speeds available for this layout
        layout_U = sorted(data["predictions"][x_downstream[0]].keys())
        first_pred = data["predictions"][x_downstream[0]][layout_U[0]]
        positions = first_pred["unscaled_node_positions"]
        wt_mask = data["wt_mask"]
        wt_idx = np.where(wt_mask != 0)[0]
        wt_pos = positions[wt_idx]

        padding = 10
        x_min, x_max = wt_pos[:, 0].min() - padding, wt_pos[:, 0].max() + padding
        y_min, y_max = wt_pos[:, 1].min() - padding, wt_pos[:, 1].max() + padding

        y_center = (wt_pos[:, 1].min() + wt_pos[:, 1].max()) / 2
        y_extent = max(abs(wt_pos[:, 1].min() - y_center), abs(wt_pos[:, 1].max() - y_center))
        y_extent = y_extent + padding

        layout_bounds[layout_name] = {
            "x_lim": (x_min, x_max),
            "y_lim": (y_min, y_max),
            "y_center": y_center,
            "y_extent": y_extent,
            "x_range": x_max - x_min,
            "y_range": y_max - y_min,
            "shrink_factor": shrink_factors.get(layout_name, 1.0),
        }

    # Figure setup
    n_rows = len(layout_names)
    n_cols = 1 + len(x_downstream)

    total_ratio = 2.5 + len(x_downstream)
    fig_width = total_ratio * (14 / 5.5)  # scale so each ratio-unit keeps same width
    fig = plt.figure(figsize=(fig_width, 3.5 * n_rows))

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        width_ratios=[2.5] + [1.0] * len(x_downstream),
        hspace=0.18,
        wspace=0.12,
    )

    for row_idx, layout_name in enumerate(layout_names):
        data = layout_data[layout_name]
        bounds = layout_bounds[layout_name]

        # Get actual wind speeds available for this layout
        layout_U = sorted(data["predictions"][x_downstream[0]].keys())

        first_pred = data["predictions"][x_downstream[0]][layout_U[0]]
        unscaled_node_positions = first_pred["unscaled_node_positions"]
        unscaled_graphs = first_pred["unscaled_graphs"]
        unscaled_probe_graphs = first_pred["unscaled_probe_graphs"]

        wt_mask = data["wt_mask"]
        wt_spacing = data["wt_spacing"]

        # Create axes
        farm_ax = fig.add_subplot(gs[row_idx, 0])

        shrink = bounds["shrink_factor"]
        if shrink < 1.0:
            pos = farm_ax.get_position()
            new_width = pos.width * shrink
            new_height = pos.height * shrink
            new_x0 = pos.x0 + (pos.width - new_width) / 2
            new_y0 = pos.y0 + (pos.height - new_height) / 2
            farm_ax.set_position([new_x0, new_y0, new_width, new_height])

        row_axes = [farm_ax]
        for j in range(len(x_downstream)):
            row_axes.append(fig.add_subplot(gs[row_idx, j + 1]))

        # Plot farm layout
        plot_probe_graph_fn(
            unscaled_graphs,
            unscaled_probe_graphs,
            unscaled_node_positions,
            include_probe_edges=False,
            include_probe_nodes=False,
            ax=row_axes[0],
            edge_linewidth=0.8,
            wt_node_size=90,
            wt_color="black",
            wt_edgecolor="white",
            wt_linewidth=1.2,
            edge_color="gray",
            edge_alpha=0.4,
            wt_marker="2",
        )

        row_axes[0].set_xlabel(r"$x/D$ [-]")
        row_axes[0].set_ylabel(r"$y/D$ [-]")
        row_axes[0].set_xlim(bounds["x_lim"])
        row_axes[0].set_ylim(bounds["y_lim"])
        row_axes[0].set_aspect("equal")

        if row_axes[0].legend_ is not None:
            row_axes[0].legend_.remove()

        letter = chr(97 + row_idx)
        n_wt = int(np.sum(wt_mask))
        if wt_spacing > 0:
            row_title = f"({letter}) {layout_name}, $n_\\mathrm{{wt}}={n_wt}$, $s_\\mathrm{{wt}}={wt_spacing:.1f}D$"
        else:
            row_title = f"({letter}) {layout_name}, $n_\\mathrm{{wt}}={n_wt}$"
        row_axes[0].set_title(row_title, loc="center", fontsize=12, pad=8)

        # Per-row mini-legend in farm layout panel (AWF only)
        if per_row_legend:
            legend_handles = []
            for U_flow_val, color in zip(layout_U, colors):
                pred_data = data["predictions"][x_downstream[0]].get(U_flow_val)
                ti_val = pred_data.get("ti_inf") if pred_data else None
                if ti_val is not None:
                    label = f"$U_\\infty$={U_flow_val:.1f}, TI={ti_val:.2f}"
                else:
                    label = f"$U_\\infty$={U_flow_val:.1f} m/s"
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, ls="-", lw=1.5, alpha=0.75, label=label)
                )
            farm_ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=6,
                frameon=True,
                framealpha=0.9,
                edgecolor="lightgray",
            )

        # Plot cross-stream profiles
        for ds_col, x_sel in enumerate(x_downstream):
            ax = row_axes[ds_col + 1]

            for U_flow_val, color in zip(layout_U, colors):
                pred_data = data["predictions"][x_sel].get(U_flow_val)
                if pred_data is None:
                    continue

                predictions = pred_data["normalized_predictions"]
                targets = pred_data["normalized_targets"]
                y_D = pred_data["y/D"]

                ax.plot(
                    predictions,
                    y_D,
                    ls="-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.75,
                )
                ax.plot(
                    targets,
                    y_D,
                    ls="--",
                    dashes=(4, 3),
                    linewidth=1.5,
                    color=color,
                    alpha=0.75,
                )

            ax.set_xlim(velocity_range)
            ax.set_ylim(bounds["y_lim"])

            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\Delta u/U$ [-]")
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            if ds_col == 0:
                ax.set_ylabel(r"$y/D$ [-]")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(f"$\\widetilde{{x}}={x_sel}D$", fontsize=12, pad=8)

    # Legends
    total_width = 2.5 + len(x_downstream)
    farm_center_x = (2.5 / 2) / total_width + 0.06
    deficit_center_x = (2.5 + total_width) / 2 / total_width - 0.03

    first_ax = fig.axes[0]
    wt_handles, wt_labels = first_ax.get_legend_handles_labels()
    legend_y = 0.95
    if wt_handles:
        fig.legend(
            wt_handles[::-1],
            wt_labels[::-1],
            loc="upper center",
            bbox_to_anchor=(farm_center_x, legend_y),
            fontsize=11,
            frameon=True,
            ncol=2,
        )

    # Shared legend: always show GNO/Target line styles
    lines = [
        plt.Line2D([0], [0], color="black", ls="-", alpha=0.75, lw=2),
        plt.Line2D([0], [0], color="black", ls="--", dashes=(4, 3), alpha=0.75, lw=2),
    ]
    legend_labels = ["GNO", "Target"]

    if not per_row_legend:
        # Add wind speed colors to shared legend (TurbOPark case)
        first_layout_data = layout_data[layout_names[0]]
        representative_U = sorted(first_layout_data["predictions"][x_downstream[0]].keys())
        color_legend_labels = [f"{U:.1f} m/s" for U in representative_U]
        for c, lab in zip(colors, color_legend_labels):
            lines.append(plt.Line2D([0], [0], color=c, ls="-", alpha=0.75, lw=2))
            legend_labels.append(lab)

    fig.legend(
        lines,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(deficit_center_x, legend_y),
        fontsize=11,
        frameon=True,
        ncol=2 if per_row_legend else 5,
    )

    output_file = os.path.join(output_path, f"crossstream_profiles_{model_type_str}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# Crossstream Error CSV Export
# =============================================================================


def export_crossstream_errors_csv(cache_data: dict, output_path: str) -> None:
    """Export crossstream profile error statistics to CSV."""
    rows = []
    metadata = cache_data["metadata"]

    for layout_name, layout_data_item in cache_data["layout_data"].items():
        for x_sel in metadata["x_downstream"]:
            # Use per-layout wind speeds (keys in predictions dict)
            layout_U = sorted(layout_data_item["predictions"][x_sel].keys())
            for U_flow in layout_U:
                errors = layout_data_item["predictions"][x_sel][U_flow].get("errors", {})

                for metric_name, stats in errors.items():
                    rows.append(
                        {
                            "layout_type": layout_name,
                            "x_downstream_D": x_sel,
                            "U_free_ms": U_flow,
                            "metric": metric_name,
                            "min": stats["min"],
                            "max": stats["max"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "max_y_D": stats.get("max_y_D", float("nan")),
                        }
                    )

    if not rows:
        print("No error data to export")
        return

    output_file = os.path.join(output_path, "crossstream_profiles_errors.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved error statistics: {output_file}")

    # Generate summary CSV
    metric_descriptions = {
        "freestream_normalized": "|u-u_hat|/U",
        "percentage": "|u-u_hat|/|u|",
        "deficit_normalized": "|u-u_hat|/|U-u|",
        "target_deficit_ms": "U-u [m/s]",
        "prediction_error_ms": "u-u_hat [m/s]",
        "deficit_normalized_filtered": "|u-u_hat|/|U-u| (deficit>5%)",
        "avg_deficit_rel_error": "|mean(U-u_hat)-mean(U-u)|/mean(U-u)",
        "avg_velocity_rel_error": "|mean(u)-mean(u_hat)|/mean(u)",
        "velocity_rel_error": "|u-u_hat|/u",
    }

    summary_rows = []
    for metric_name in metric_descriptions:
        metric_rows = [r for r in rows if r["metric"] == metric_name]
        if not metric_rows:
            continue

        valid_rows = [r for r in metric_rows if not (math.isnan(r["max"]) or math.isnan(r["mean"]))]
        if not valid_rows:
            continue

        overall_min = min(r["min"] for r in valid_rows if not math.isnan(r["min"]))
        overall_max = max(r["max"] for r in valid_rows)
        mean_min = min(r["mean"] for r in valid_rows)
        mean_max = max(r["mean"] for r in valid_rows)
        max_values = [r["max"] for r in valid_rows]
        max_row = max(valid_rows, key=lambda r: r["max"])

        description = metric_descriptions.get(metric_name, "")
        display_name = f"{metric_name} ({description})" if description else metric_name

        summary_rows.append(
            {
                "metric": display_name,
                "overall_min_pct": round(overall_min, 2),
                "overall_max_pct": round(overall_max, 2),
                "max_min_pct": round(min(max_values), 2),
                "max_max_pct": round(max(max_values), 2),
                "mean_min_pct": round(mean_min, 2),
                "mean_max_pct": round(mean_max, 2),
                "max_at_layout": max_row["layout_type"],
                "max_at_x_D": max_row["x_downstream_D"],
                "max_at_y_D": round(max_row["max_y_D"], 2)
                if not math.isnan(max_row["max_y_D"])
                else "nan",
                "max_at_U_ms": max_row["U_free_ms"],
            }
        )

    if summary_rows:
        summary_file = os.path.join(output_path, "crossstream_errors_summary.csv")
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved error summary: {summary_file}")


# =============================================================================
# Deficit Threshold Sweep
# =============================================================================


def export_deficit_threshold_sweep(cache_data: dict, output_path: str) -> None:
    """Sweep deficit filter threshold from 0.01% to 10% and export statistics."""
    metadata = cache_data["metadata"]
    threshold_pcts = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.1,
        0.2,
        0.3,
        0.5,
        0.7,
        1,
        2,
        3,
        5,
        7,
        10,
    ]
    thresholds = [p / 100.0 for p in threshold_pcts]

    all_results = []
    for layout_name, layout_data_item in cache_data["layout_data"].items():
        for x_sel in metadata["x_downstream"]:
            layout_U = sorted(layout_data_item["predictions"][x_sel].keys())
            for U_flow in layout_U:
                pred_data = layout_data_item["predictions"][x_sel][U_flow]
                u = np.array(pred_data["unscaled_targets"])
                u_hat = np.array(pred_data["unscaled_predictions"])
                U = U_flow

                all_results.append(
                    {
                        "layout": layout_name,
                        "x_D": x_sel,
                        "U": U_flow,
                        "error_abs": np.abs(u - u_hat),
                        "target_deficit": np.abs(U - u),
                        "U_freestream": U,
                    }
                )

    rows = []
    for i, threshold in enumerate(thresholds):
        threshold_pct = threshold_pcts[i]
        all_filtered_errors = []
        n_valid_points = 0
        n_total_points = 0

        for result in all_results:
            min_deficit = threshold * result["U_freestream"]
            mask = result["target_deficit"] > min_deficit

            n_total_points += len(result["error_abs"])
            n_valid_points += np.sum(mask)

            if np.any(mask):
                filtered_error = (result["error_abs"][mask] / result["target_deficit"][mask]) * 100
                all_filtered_errors.extend(filtered_error.tolist())

        if all_filtered_errors:
            arr = np.array(all_filtered_errors)
            rows.append(
                {
                    "threshold_pct": threshold_pct,
                    "min": round(float(np.min(arr)), 2),
                    "max": round(float(np.max(arr)), 2),
                    "mean": round(float(np.mean(arr)), 2),
                    "std": round(float(np.std(arr)), 2),
                    "median": round(float(np.median(arr)), 2),
                    "p95": round(float(np.percentile(arr, 95)), 2),
                    "p99": round(float(np.percentile(arr, 99)), 2),
                    "n_valid_points": n_valid_points,
                    "pct_data_retained": round(100 * n_valid_points / n_total_points, 1),
                }
            )
        else:
            rows.append(
                {
                    "threshold_pct": threshold_pct,
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "p95": float("nan"),
                    "p99": float("nan"),
                    "n_valid_points": 0,
                    "pct_data_retained": 0.0,
                }
            )

    output_file = os.path.join(output_path, "deficit_threshold_sweep.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved threshold sweep: {output_file}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    thresholds_pct_vals = [r["threshold_pct"] for r in rows]
    key_thresholds = [0.1, 1.0]

    ax1 = axes[0]
    ax1.plot(thresholds_pct_vals, [r["max"] for r in rows], "o-", label="Max", color="C3")
    ax1.plot(
        thresholds_pct_vals, [r["p99"] for r in rows], "s-", label="99th percentile", color="C1"
    )
    ax1.plot(
        thresholds_pct_vals, [r["p95"] for r in rows], "^-", label="95th percentile", color="C2"
    )
    ax1.plot(thresholds_pct_vals, [r["mean"] for r in rows], "d-", label="Mean", color="C0")
    for thresh in key_thresholds:
        ax1.axvline(x=thresh, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Deficit threshold (% of freestream)")
    ax1.set_ylabel("Deficit-normalized error [%]")
    ax1.set_title("Error statistics vs. filter threshold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0.01, 10)
    ax1.set_xticks([0.01, 0.1, 1, 10])
    ax1.set_xticklabels(["0.01", "0.1", "1", "10"])

    ax2 = axes[1]
    ax2.plot(thresholds_pct_vals, [r["pct_data_retained"] for r in rows], "o-", color="C4")
    for thresh in key_thresholds:
        ax2.axvline(x=thresh, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Deficit threshold (% of freestream)")
    ax2.set_ylabel("Data retained [%]")
    ax2.set_title("Data coverage vs. filter threshold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(0.01, 10)
    ax2.set_xticks([0.01, 0.1, 1, 10])
    ax2.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plot_file = os.path.join(output_path, "deficit_threshold_sweep.pdf")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold sweep plot: {plot_file}")


# =============================================================================
# WT Spatial Errors - Aggregated
# =============================================================================


def plot_wt_spatial_errors_aggregated(
    cache_data: dict,
    output_path: str,
    model_type_str: str = "best_model",
    metric_to_plot: str = "abs_max",
):
    """Generate the wind turbine spatial errors publication figure (aggregated across wind speeds).

    Args:
        cache_data: Dictionary with cached WT error data containing:
            - aggregated_extremas: {metric: (min, max)}
            - layout_data: {layout_name: {wt_positions_D, aggregated_errors}}
        output_path: Path to save the figure
        model_type_str: Model type string for filename
        metric_to_plot: Which metric to plot ("mean", "std", or "abs_max")
    """
    matplotlib_set_rcparams("paper")

    aggregated_extremas = cache_data["aggregated_extremas"]
    layout_data = cache_data["layout_data"]

    all_layout_names = list(layout_data.keys())

    # Pre-calculate bounds
    layout_bounds = {}
    padding = 8
    for layout_name in all_layout_names:
        data = layout_data[layout_name]
        pos_data = data["wt_positions_D"]
        if isinstance(pos_data, dict):
            first_U = next(iter(pos_data.keys()))
            wt_positions = pos_data[first_U]
        else:
            wt_positions = pos_data
        x_min = wt_positions[:, 0].min() - padding
        x_max = wt_positions[:, 0].max() + padding
        y_min = wt_positions[:, 1].min() - padding
        y_max = wt_positions[:, 1].max() + padding
        layout_bounds[layout_name] = {
            "x_lim": (x_min, x_max),
            "y_lim": (y_min, y_max),
            "x_range": x_max - x_min,
            "y_range": y_max - y_min,
        }

    max_y_range = max(layout_bounds[ln]["y_range"] for ln in all_layout_names)

    for layout_name in all_layout_names:
        bounds = layout_bounds[layout_name]
        y_center = (bounds["y_lim"][0] + bounds["y_lim"][1]) / 2
        bounds["y_lim"] = (y_center - max_y_range / 2, y_center + max_y_range / 2)

    # Split into row1 (first N-1 layouts) and row2 (last layout)
    row1_layouts = all_layout_names[:-1]
    row2_layout = all_layout_names[-1]
    row1_ratios = [layout_bounds[ln]["x_range"] for ln in row1_layouts]

    fig = plt.figure(figsize=(10, 6))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, len(row1_layouts), subplot_spec=outer_gs[0], width_ratios=row1_ratios, wspace=0
    )
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[1])

    axes = [fig.add_subplot(gs1[i]) for i in range(len(row1_layouts))]
    axes.append(fig.add_subplot(gs2[0]))

    layout_names = row1_layouts + [row2_layout]

    extrema = np.max(np.abs(aggregated_extremas[metric_to_plot]))
    if metric_to_plot == "abs_max":
        vmin = 0
        vmax = extrema
        colormap = "gist_heat_r"
        cbar_legend = r"$\max| (\boldsymbol{u}-\hat{\boldsymbol{u}})/\boldsymbol{U} |$ [$\%$]"
        cbar_extend = "max"
    else:
        vmin = -extrema
        vmax = extrema
        colormap = "seismic"
        cbar_legend = (
            r"$\overline{u_\mathrm{err}/U}$"
            if metric_to_plot == "mean"
            else r"$\sigma(u_\mathrm{err}/U)$"
        )
        cbar_extend = "both"

    sc = None
    for i, layout_name in enumerate(layout_names):
        ax = axes[i]
        data = layout_data[layout_name]
        bounds = layout_bounds[layout_name]

        wt_positions = data["wt_positions_D"]
        if isinstance(wt_positions, dict):
            first_U = next(iter(wt_positions.keys()))
            wt_positions = wt_positions[first_U]
        wt_rel_errors_agg = data["aggregated_errors"][metric_to_plot]

        sc = ax.scatter(
            wt_positions[:, 0],
            wt_positions[:, 1],
            c=wt_rel_errors_agg,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            s=25,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.5,
            marker="o",
        )

        ax.set_xlabel("$x/D$ [-]")
        if i == 0 or i == 3:
            ax.set_ylabel("$y/D$ [-]")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.set_xlim(bounds["x_lim"])
        ax.set_ylim(bounds["y_lim"])
        ax.set_aspect("equal")

        if i < len(row1_layouts):
            if i == 0:
                ax.set_anchor("E")
            elif i == len(row1_layouts) - 1:
                ax.set_anchor("W")
            else:
                ax.set_anchor("C")

        ax.set_title(f"({chr(97 + i)}) {layout_name}", loc="left", pad=5)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, extend=cbar_extend)
    cbar.set_label(cbar_legend)

    # Align rows
    fig.canvas.draw()
    pos_d = axes[-1].get_position()
    for i in range(len(row1_layouts)):
        pos = axes[i].get_position()
        if i == 0:
            offset = pos_d.x0 - pos.x0
        axes[i].set_position([pos.x0 + offset, pos.y0, pos.width, pos.height])

    output_file = os.path.join(output_path, f"wt_spatial_errors_all_layouts_{model_type_str}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


# =============================================================================
# WT Spatial Errors - Per Wind Speed
# =============================================================================


def plot_wt_spatial_errors_per_windspeed(
    wt_error_data: dict,
    output_path: str,
) -> None:
    """Generate per-wind-speed WT spatial errors figure from pre-computed data.

    Args:
        wt_error_data: Pre-computed WT error data containing:
            - layout_errors: {layout_name: {U: wt_rel_errors}}
            - layout_positions: {layout_name: wt_positions_D}
            - layout_bounds: {layout_name: {x_lim, y_lim, x_range, y_range}}
            - layout_U_values: {layout_name: [sorted U values]}
            - global_extrema: float
            - n_wind_speed_rows: int
        output_path: Directory to save the figure
    """
    matplotlib_set_rcparams("paper")

    layout_errors = wt_error_data["layout_errors"]
    layout_positions = wt_error_data["layout_positions"]
    layout_bounds = wt_error_data["layout_bounds"]
    global_extrema = wt_error_data["global_extrema"]
    layout_U_values = wt_error_data["layout_U_values"]
    n_rows = wt_error_data["n_wind_speed_rows"]

    all_layout_names = list(layout_errors.keys())
    row1_layouts = all_layout_names[:-1]
    row2_layout = all_layout_names[-1]

    vmin = 0
    vmax = global_extrema
    colormap = "gist_heat_r"
    cbar_legend = r"$|(\boldsymbol{u}-\hat{\boldsymbol{u}})/\boldsymbol{U}|$ [$\%$]"
    cbar_extend = "max"

    row1_width_ratios = [layout_bounds[ln]["x_range"] for ln in row1_layouts if ln in layout_bounds]

    block1_y_range = max(layout_bounds[ln]["y_range"] for ln in row1_layouts if ln in layout_bounds)
    block2_y_range = layout_bounds[row2_layout]["y_range"]

    fig = plt.figure(figsize=(12, 10))
    outer_gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[n_rows * block1_y_range, n_rows * block2_y_range],
        hspace=0.25,
    )

    gs_block1 = gridspec.GridSpecFromSubplotSpec(
        n_rows,
        len(row1_layouts),
        subplot_spec=outer_gs[0],
        width_ratios=row1_width_ratios,
        wspace=0.15,
        hspace=0.12,
    )
    gs_block2 = gridspec.GridSpecFromSubplotSpec(
        n_rows,
        1,
        subplot_spec=outer_gs[1],
        hspace=0.12,
    )

    axes = {}
    all_axes = []

    for row in range(n_rows):
        for col, layout_name in enumerate(row1_layouts):
            ax = fig.add_subplot(gs_block1[row, col])
            axes[(layout_name, row)] = ax
            all_axes.append(ax)

    for row in range(n_rows):
        ax = fig.add_subplot(gs_block2[row])
        axes[(row2_layout, row)] = ax
        all_axes.append(ax)

    sc = None
    subplot_labels = {name: chr(97 + i) for i, name in enumerate(all_layout_names)}

    # Block 1
    for col, layout_name in enumerate(row1_layouts):
        if layout_name not in layout_errors:
            continue

        bounds = layout_bounds[layout_name]
        positions = layout_positions[layout_name]
        U_values = layout_U_values.get(layout_name, [])

        for row, U in enumerate(U_values):
            ax = axes[(layout_name, row)]
            wt_errors = layout_errors[layout_name][U]

            sc = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=wt_errors,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                s=20,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
                marker="o",
            )

            ax.set_xlim(bounds["x_lim"])
            ax.set_ylim(bounds["y_lim"])
            ax.set_aspect(0.6)

            if col == 0:
                ax.set_ylabel(f"$U = {U:.0f}$ m/s\n$y/D$ [-]", fontsize=9)
            else:
                ax.set_yticklabels([])

            if row == len(U_values) - 1:
                ax.set_xlabel("$x/D$ [-]")
                if col == len(row1_layouts) - 1:
                    ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(
                    f"({subplot_labels[layout_name]}) {layout_name}",
                    fontsize=10,
                    pad=8,
                )

    # Block 2
    if row2_layout in layout_errors:
        bounds = layout_bounds[row2_layout]
        positions = layout_positions[row2_layout]
        U_values = layout_U_values.get(row2_layout, [])

        for row, U in enumerate(U_values):
            ax = axes[(row2_layout, row)]
            wt_errors = layout_errors[row2_layout][U]

            sc = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=wt_errors,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                s=20,
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
                marker="o",
            )

            ax.set_xlim(bounds["x_lim"])
            ax.set_ylim(bounds["y_lim"])
            ax.set_aspect(0.6)

            ax.set_ylabel(f"$U = {U:.0f}$ m/s\n$y/D$ [-]", fontsize=9)

            if row == len(U_values) - 1:
                ax.set_xlabel("$x/D$ [-]")
            else:
                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(
                    f"({subplot_labels[row2_layout]}) {row2_layout}",
                    fontsize=10,
                    pad=8,
                )

    cbar = fig.colorbar(
        sc,
        ax=all_axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.06,
        aspect=40,
        extend=cbar_extend,
    )
    cbar.set_label(cbar_legend)

    # Reposition Block 1 axes to match Block 2 width
    fig.canvas.draw()
    block2_ax = axes[(row2_layout, 0)]
    block2_pos = block2_ax.get_position()
    target_left = block2_pos.x0
    target_right = block2_pos.x1
    target_width = target_right - target_left

    block1_axes_by_col = {
        col: [axes[(ln, row)] for row in range(n_rows)]
        for col, ln in enumerate(row1_layouts)
        if ln in layout_errors
    }

    n_block1_cols = len(block1_axes_by_col)
    if n_block1_cols >= 2:
        col_positions = []
        for col in range(n_block1_cols):
            ax = block1_axes_by_col[col][0]
            pos = ax.get_position()
            col_positions.append({"left": pos.x0, "right": pos.x1, "width": pos.width})

        block1_left = col_positions[0]["left"]
        block1_right = col_positions[-1]["right"]
        current_width = block1_right - block1_left

        target_gap = 0.025
        total_gap_adjustment = sum(
            (col_positions[i + 1]["left"] - col_positions[i]["right"]) - target_gap
            for i in range(n_block1_cols - 1)
        )
        width_after_gaps = current_width - total_gap_adjustment
        scale = target_width / width_after_gaps

        new_positions = []
        current_left = target_left
        for col in range(n_block1_cols):
            new_width = col_positions[col]["width"] * scale
            new_positions.append({"left": current_left, "width": new_width})
            current_left += new_width + target_gap * scale

        for col in range(n_block1_cols):
            for ax in block1_axes_by_col[col]:
                pos = ax.get_position()
                ax.set_position(
                    [new_positions[col]["left"], pos.y0, new_positions[col]["width"], pos.height]
                )

    output_file = os.path.join(output_path, "wt_spatial_errors_per_windspeed.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()
