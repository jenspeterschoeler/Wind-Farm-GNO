import os
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from utils.plotting import matplotlib_set_rcparams

matplotlib_set_rcparams("paper")

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

# setup figure with gridspec

plt.rcParams["patch.linewidth"] = 0
plt.rcParams["patch.edgecolor"] = "none"

# fig = plt.figure(figsize=(3.347*2, 3.347*4))
fig = plt.figure(figsize=(6, 14.5))
# fig = plt.figure(figsize=(3.347, 7))
gs = gridspec.GridSpec(
    4, 2, height_ratios=[0.7, 1, 1, 1], width_ratios=[3, 1]
)  # 4 rows, 2 column
# First subplot full row
ax0 = fig.add_subplot(gs[0, :])  # First subplot spans both columns

ax0.set_xlabel(metric_label)
ax0.set_ylabel("PDF [-]")
line_width = 2
g = sns.kdeplot(
    data=raw_df_local_metrics,
    x=metric,
    hue="layout_type",
    cumulative=False,
    common_norm=False,
    linewidth=line_width,
    legend=True,
    ax=ax0,
)
g.legend_.set_title(None)
lss = ["-", "--", "-", "--"]
line_handles = g.get_legend().legend_handles
legend_text = [t.get_text() for t in g.legend_.get_texts()]
lines = g.lines

ax0.legend_.remove()
# # Reverse the order of lines to match the legend order
for handle, line, ls in zip(line_handles, lines[::-1], lss):
    handle.set_linestyle(ls)
    line.set_linestyle(ls)


ax1 = fig.add_subplot(gs[1, :])  # Second subplot
g = sns.barplot(
    data=df_local_metrics,
    x="U_freestream",
    y=metric,
    hue="layout_type",
    ax=ax1,
    legend=True,
)
g.legend_.set_title(None)
bar_handles = g.get_legend().legend_handles
bars = g.patches

ax1.legend_.remove()
ax1.set_xlabel(r"$U$ [$\mathrm{ms}^{-1}$]")
ax1.set_ylabel(metric_label)

ax2 = fig.add_subplot(gs[2, :])  # Third subplot
g = sns.barplot(
    data=df_local_metrics,
    x="TI_ambient",
    y=metric,
    hue="layout_type",
    ax=ax2,
    legend=False,
)
ax2.set_xlabel("TI [-]")
ax2.set_ylabel(metric_label)
# rotate x labels
plt.setp(ax2.get_xticklabels(), rotation=45)  # Rotate x labels

ax3 = fig.add_subplot(gs[3, 0])  # Fourth subplot
g = sns.barplot(
    data=df_local_metrics,
    x="n_wt",
    y=metric,
    hue="layout_type",
    ax=ax3,
    legend=False,
)
ax3.set_xlabel(r"$n_\mathrm{wt}$ [-]")
ax3.set_ylabel(metric_label)
plt.setp(ax3.get_xticklabels(), rotation=90)  # Rotate x labels

ax4 = fig.add_subplot(gs[3, 1])  # Fifth subplot
g = sns.barplot(
    data=df_local_metrics,
    x="wt_spacing",
    y=metric,
    hue="layout_type",
    ax=ax4,
    legend=False,
)
ax4.set_xlabel(r"$s_\mathrm{wt}/D$ [-]")
ax4.set_ylabel(None)
# plt.setp(ax4.get_xticklabels(), rotation=90)  # Rotate x labels


# Create a single legend for the entire figure
shared_handles = [
    [line_handle, bar_handle]
    for line_handle, bar_handle in zip(line_handles, bar_handles)
]
shared_text = [[l_text, " "] for l_text in legend_text]
shared_handles = [item for sublist in shared_handles for item in sublist]
shared_text = [item for sublist in shared_text for item in sublist]
leg = fig.legend(
    handles=shared_handles,
    labels=shared_text,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.94),
    frameon=True,  # ensure frame is created
    fancybox=False,  # use rectangular box so edge is visible
)
frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.8)

# set ylims for all RMSE plots (ignoring ax0, it is PDF)
for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4]):
    if i > 0:
        ax.set_ylim(bottom=0, top=y_max)
    # add a letter in the left corner (a), (b), ...
    if i < 3:
        placement = [0.075, 0.935]
    elif i == 3:
        placement = [0.115, 0.935]
    elif i == 4:
        placement = [0.3, 0.935]
    ax.text(
        placement[0],
        placement[1],
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=12,
        # fontweight="bold",
        va="top",
        ha="right",
    )


plt.subplots_adjust(hspace=0.25)
save_path = os.path.join(
    fig_folder_path, f"error_metrics_summary_{metric}" + model_type_str + ".pdf"
)
plt.savefig(save_path, bbox_inches="tight")

# %%
