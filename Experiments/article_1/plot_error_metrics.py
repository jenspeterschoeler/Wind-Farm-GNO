import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from utils.plotting import matplotlib_set_rcparams

matplotlib_set_rcparams("paper")

# Use tab10 colors but replace green with purple to avoid red-green colorblindness issues
# tab10: blue(0), orange(1), green(2), red(3), purple(4), brown(5), pink(6), gray(7), olive(8), cyan(9)
LAYOUT_PALETTE = [
    plt.cm.tab10.colors[0],  # Blue
    plt.cm.tab10.colors[1],  # Orange
    plt.cm.tab10.colors[4],  # Purple (instead of green)
    plt.cm.tab10.colors[3],  # Red
]

# %% Load data

# main_path = os.path.abspath(
#     "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
# )  # sophia
main_path = "./assets/best_model_Vj8"  # local, run from repo root
# main_path = "../../assets/best_model_Vj8"  # local, run from this folder (for ipython)

cfg_path = os.path.join(main_path, ".hydra/config.yaml")
model_cfg_path = os.path.abspath(os.path.join(main_path, "model_config.json"))
params_paths = os.path.join(main_path, "best_params.msgpack")
if "best" in params_paths:
    model_type_str = "best"
else:
    model_type_str = "final"

fig_folder_path = os.path.join(main_path, "model/plots_" + model_type_str)
local_metrics_path = os.path.join(fig_folder_path, "local_metrics.csv")


df_local_metrics = pd.read_csv(local_metrics_path)
df_local_metrics = df_local_metrics.sort_values(
    by=["layout_type", "U_freestream", "TI_ambient", "n_wt", "wt_spacing"]
)
raw_df_local_metrics = pd.read_csv(local_metrics_path)
raw_df_local_metrics = raw_df_local_metrics.sort_values(
    by=["layout_type", "U_freestream", "TI_ambient", "n_wt", "wt_spacing"]
)
# put wind speed in bins of 1 m/s
df_local_metrics["U_freestream"] = df_local_metrics["U_freestream"].round().astype(int)

# put TI in bins of .05
df_local_metrics["TI_ambient"] = (df_local_metrics["TI_ambient"] / 0.05).round() * 0.05
# set number of decimals to 2
df_local_metrics["TI_ambient"] = df_local_metrics["TI_ambient"].round(2)


nwt_lower, nwt_higher = df_local_metrics["n_wt"].min(), df_local_metrics["n_wt"].max()
nwt_n_bins = int((nwt_higher - nwt_lower) / 5)
nwt_edges = np.linspace(nwt_lower, nwt_higher, nwt_n_bins + 1)
nwt_lbs = [
    "(%d, %d]" % (nwt_edges[i], nwt_edges[i + 1]) for i in range(len(nwt_edges) - 1)
]
df_local_metrics["n_wt"] = pd.cut(
    df_local_metrics["n_wt"],
    bins=nwt_n_bins,
    right=False,
    labels=nwt_lbs,
)


min_ws = np.floor(df_local_metrics["wt_spacing"].min())
max_ws = np.ceil(df_local_metrics["wt_spacing"].max())
if max_ws - min_ws >= 1:
    bins = np.arange(min_ws, max_ws + 1, 1)  # integer edges
    labels = list(range(int(min_ws), int(max_ws)))  # integer labels for bins
    df_local_metrics["wt_spacing"] = pd.cut(
        df_local_metrics["wt_spacing"],
        bins=bins,
        right=False,
        labels=labels,
    )
    # keep as ordered categorical so seaborn plots in numeric order
    df_local_metrics["wt_spacing"] = pd.Categorical(
        df_local_metrics["wt_spacing"], categories=labels, ordered=True
    )
else:
    # fallback: single bin, label by rounded value
    df_local_metrics["wt_spacing"] = (
        df_local_metrics["wt_spacing"].round().astype(int).astype(str)
    )


# %% Plotting

metric = "rmse_local"
# metric = "mape_local"

if metric == "rmse_local":
    metric_label = r"RMSE [$\mathrm{ms}^{-1}$]"
    y_max = 0.6
elif metric == "mape_local":
    metric_label = "MAPE [%]"
    y_max = 1.6

# Count categories for each variable
n_cat_U = df_local_metrics["U_freestream"].nunique()
n_cat_TI = df_local_metrics["TI_ambient"].nunique()
n_cat_nwt = df_local_metrics["n_wt"].nunique()
n_cat_ws = df_local_metrics["wt_spacing"].nunique()

# Common bar styling
bar_edge_color = "black"
bar_edge_width = 0.5
gap = 0.08
ref_width = 0.8  # Base width for reference panel

# Setup figure with gridspec - 4x2 grid layout
# New layout:
# Row 0: (a) KDE [2/3 width] | (b) wt_spacing [1/3 width]
# Row 1: (c) Wind speed [full width]
# Row 2: (d) TI [full width]
# Row 3: (e) n_wt [full width]
fig = plt.figure(figsize=(8.5, 8.5))  # Slightly smaller to make fonts appear larger
gs = gridspec.GridSpec(
    4,
    2,
    height_ratios=[1.0, 1, 1, 1],  # Increased height for row 0
    width_ratios=[2, 1],  # 2/3 to 1/3 ratio
    wspace=0.18,  # More spacing between columns
    hspace=0.28,  # More spacing between rows
    figure=fig,
    top=0.90,  # More space for legend at top
    bottom=0.08,
    left=0.08,
    right=0.98,
)

# Create all axes first
ax0 = fig.add_subplot(gs[0, 0])  # (a) KDE - 2/3 width
ax1 = fig.add_subplot(gs[0, 1])  # (b) wt_spacing - 1/3 width
ax2 = fig.add_subplot(gs[1, :])  # (c) Wind speed - full width
ax3 = fig.add_subplot(gs[2, :])  # (d) TI - full width
ax4 = fig.add_subplot(gs[3, :])  # (e) n_wt - full width

# Draw figure to get actual axes positions
fig.canvas.draw()

# Get actual axes widths in figure coordinates
ax2_width = ax2.get_position().width  # Full width panel
ax1_width = ax1.get_position().width  # 1/3 width panel (wt_spacing)

# Calculate widths dynamically based on actual panel sizes
# Visual bar width = (panel_width / n_categories) * relative_width
# For consistent visual width across all panels

# Use densest panel as reference (most categories per unit width)
density_U = n_cat_U / ax2_width
density_TI = n_cat_TI / ax2_width
density_nwt = n_cat_nwt / ax2_width
density_ws = n_cat_ws / ax1_width

max_density = max(density_U, density_TI, density_nwt, density_ws)

# Calculate widths to match the densest panel
width_U = min(0.95, ref_width * (density_U / max_density))
width_TI = min(0.95, ref_width * (density_TI / max_density))
width_nwt = min(0.95, ref_width * (density_nwt / max_density))
width_ws = min(0.95, ref_width * (density_ws / max_density))

# (a) Distribution plot - 2/3 width (top left)
# Option: "kde" for KDE plot, "hist_line" for histogram with line tops
distribution_plot_type = "hist_line"

ax0.set_xlabel(metric_label)
line_width = 1.5
lss = ["-", "--", "-.", ":"]  # Line styles for KDE
markers = ["o", "s", "^", "D"]  # Different markers: circle, square, triangle, diamond

if distribution_plot_type == "kde":
    ax0.set_ylabel("PDF [-]")
    g = sns.kdeplot(
        data=raw_df_local_metrics,
        x=metric,
        hue="layout_type",
        cumulative=False,
        common_norm=False,
        linewidth=line_width,
        legend=True,
        ax=ax0,
        palette=LAYOUT_PALETTE,
        clip=(0, None),
        bw_adjust=0.8,
    )
    g.legend_.set_title(None)
    line_handles = g.get_legend().legend_handles
    legend_text = [t.get_text() for t in g.legend_.get_texts()]
    lines = g.lines
    ax0.legend_.remove()
    for handle, line, ls in zip(line_handles, lines[::-1], lss):
        handle.set_linestyle(ls)
        line.set_linestyle(ls)

elif distribution_plot_type == "hist_line":
    ax0.set_ylabel("Count [-]")
    # Get unique layout types and plot histogram lines for each
    layout_types = raw_df_local_metrics["layout_type"].unique()
    n_bins = 25
    # Determine common bin edges across all data
    all_values = raw_df_local_metrics[metric].dropna()
    bin_edges = np.linspace(all_values.min(), all_values.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    line_handles = []
    legend_text = list(layout_types)
    for i, (layout, marker) in enumerate(zip(layout_types, markers)):
        data = raw_df_local_metrics[raw_df_local_metrics["layout_type"] == layout][metric]
        counts, _ = np.histogram(data, bins=bin_edges)
        # Plot line connecting bar tops with markers, solid lines for all
        line, = ax0.plot(
            bin_centers,
            counts,
            color=LAYOUT_PALETTE[i],
            linewidth=line_width,
            linestyle="-",  # Solid line for all
            marker=marker,
            markersize=4,
            markerfacecolor=LAYOUT_PALETTE[i],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=layout,
        )
        line_handles.append(line)
    ax0.set_xlim(left=0)
    # Add small space below y=0 to show markers, but not enough for negative tick
    y_max_hist = ax0.get_ylim()[1]
    ax0.set_ylim(bottom=-y_max_hist * 0.03, top=y_max_hist)

# (b) wt_spacing - 1/3 width (top right)
g = sns.barplot(
    data=df_local_metrics,
    x="wt_spacing",
    y=metric,
    hue="layout_type",
    ax=ax1,
    legend=True,
    palette=LAYOUT_PALETTE,
    width=width_ws,
    gap=gap,
    edgecolor=bar_edge_color,
    linewidth=bar_edge_width,
)
g.legend_.set_title(None)
bar_handles = g.get_legend().legend_handles
ax1.legend_.remove()
ax1.set_xlabel(r"$s_\mathrm{wt}/D$ [-]")
ax1.set_ylabel(metric_label)

# (c) Wind speed - full width
g = sns.barplot(
    data=df_local_metrics,
    x="U_freestream",
    y=metric,
    hue="layout_type",
    ax=ax2,
    legend=False,
    palette=LAYOUT_PALETTE,
    width=width_U,
    gap=gap,
    edgecolor=bar_edge_color,
    linewidth=bar_edge_width,
)
ax2.set_xlabel(r"$U$ [$\mathrm{ms}^{-1}$]")
ax2.set_ylabel(metric_label)

# (d) TI - full width
g = sns.barplot(
    data=df_local_metrics,
    x="TI_ambient",
    y=metric,
    hue="layout_type",
    ax=ax3,
    legend=False,
    palette=LAYOUT_PALETTE,
    width=width_TI,
    gap=gap,
    edgecolor=bar_edge_color,
    linewidth=bar_edge_width,
)
ax3.set_xlabel(r"$I_0$ [-]")
ax3.set_ylabel(metric_label)
# Rotate labels to prevent overlap
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)

# (e) n_wt - full width
g = sns.barplot(
    data=df_local_metrics,
    x="n_wt",
    y=metric,
    hue="layout_type",
    ax=ax4,
    legend=False,
    palette=LAYOUT_PALETTE,
    width=width_nwt,
    gap=gap,
    edgecolor=bar_edge_color,
    linewidth=bar_edge_width,
)
ax4.set_xlabel(r"$n_\mathrm{wt}$ [-]")
ax4.set_ylabel(metric_label)
# Rotate labels 45 degrees
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", fontsize=8)


# Adjust spacing: move rows down for better spacing
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos2.y0 - 0.015, pos2.width, pos2.height])

pos3 = ax3.get_position()
ax3.set_position([pos3.x0, pos3.y0 - 0.030, pos3.width, pos3.height])

pos4 = ax4.get_position()
ax4.set_position([pos4.x0, pos4.y0 - 0.050, pos4.width, pos4.height])

# Create a single legend for the entire figure at top center
# Capitalize first letter of legend labels
shared_handles = [
    [line_handle, bar_handle]
    for line_handle, bar_handle in zip(line_handles, bar_handles)
]
shared_text = [[l_text.capitalize(), " "] for l_text in legend_text]  # Capitalize first letter
shared_handles = [item for sublist in shared_handles for item in sublist]
shared_text = [item for sublist in shared_text for item in sublist]
leg = fig.legend(
    handles=shared_handles,
    labels=shared_text,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.98),  # Moved down slightly
    frameon=True,
    fancybox=False,
)
frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.8)

# Set ylims for bar plots and add panel labels
axes = [ax0, ax1, ax2, ax3, ax4]
for i, ax in enumerate(axes):
    # Set ylim for bar plots (not KDE which is now ax0)
    if i != 0:
        ax.set_ylim(bottom=0, top=y_max)
    # Panel labels
    placement = [0.03, 0.92]
    ax.text(
        placement[0],
        placement[1],
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
    )
save_path = os.path.join(
    fig_folder_path, f"error_metrics_summary_{metric}" + model_type_str + ".pdf"
)
plt.savefig(save_path, bbox_inches="tight")

# %%
