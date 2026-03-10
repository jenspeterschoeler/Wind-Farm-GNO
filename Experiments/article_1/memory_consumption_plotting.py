import glob
import os
import sys
from pathlib import Path

import jax
import pandas as pd
from jax import numpy as jnp

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils.plotting import matplotlib_set_rcparams

matplotlib_set_rcparams("paper")

# Increase text sizes while maintaining plot dimensions
plt.rcParams["font.size"] = 13
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11

# %% Initialization
# main_path = os.path.abspath(
#     "/work/users/jpsch/SPO_sophia_dir/outputs/GNO_probe_large/multirun/2025-08-18/16-06-16/1"
# )  # sophia
main_path = "./assets/best_model_Vj8"  # local, run from repo root
# main_path = "../../assets/best_model_Vj8"  # local, run from this folder (for ipython)


if "assets" in main_path:
    running_locally = True
else:
    running_locally = False


params_paths = os.path.join(main_path, "best_params.msgpack")
if "best" in params_paths:
    model_type_str = "best_model"
else:
    model_type_str = "last model"
fig_folder_path = os.path.join(main_path, "model/figures_" + model_type_str)
os.makedirs(fig_folder_path, exist_ok=True)


df_paths = glob.glob(os.path.join(fig_folder_path, "memory_and_timing_*.csv"))
li = []
for filename in df_paths:
    df_ = pd.read_csv(filename, index_col=None, header=0)
    li.append(df_)
df = pd.concat(li, axis=0, ignore_index=True)
df["size_of_graph"] = df["size_of_graph"] / 1e6


downsample = False
if downsample:
    # only keep every 101 row
    df = df[df.index % 101 == 0]

df["pywake_simulation_time[CPUh]"] = df["pywake_simulation_time[s]"] / 3600


# %% Plotting experiments

df_long = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["max_model_memory[MB]_jitted"],
    var_name="Memory Type",
    value_name="Memory [MB]",
)

plt.figure(figsize=(6, 3))
sns.lineplot(
    data=df_long,
    x="size_of_graph",
    y="Memory [MB]",
    hue="Memory Type",
)
plt.xlabel(r"$|G| \times 10^{-6}$ [-]")
plt.ylabel("Memory [MB]")
# add pywake memory as horizontal lines

sns.lineplot(
    data=df[
        df["pywake_simulation_memory[MB]"] > 0
    ],  # remove outliers in pywake simulation memory
    x="size_of_graph",
    y="pywake_simulation_memory[MB]",
    color="gray",
    label="PyWake",
    zorder=0,
)
org_legend_elements = plt.gca().get_legend_handles_labels()
py_wake_legend_elements = [
    Line2D([0], [0], color="gray", lw=5, alpha=0.7),
]
legend_handles = org_legend_elements[0] + [py_wake_legend_elements[0]]
legend_labels = ["GNO", "PyWake"]
plt.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="upper left",
)
plt.xlim(0, 4)
plt.savefig(
    os.path.join(fig_folder_path, "memory_consumption_plot.pdf"),
    bbox_inches="tight",
)


jitting_columns = [
    "embedding_time[s]_jitting",
    "wt_processing_time[s]_jitting",
    "probe_processing_time[s]_jitting",
    "decoding_time[s]_jitting",
    "total_model_time[s]_jitting",
    "total_pred_time_jitting[s]",
    "decoding_time[s]_jitting",
]
for col in jitting_columns:
    df[col] = df[col].where(df.index % 10 == 0, other=jnp.nan)


df_total_pred_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=[
        "total_pred_time_no_jit[s]",
        "total_pred_time_jitting[s]",
        "total_pred_time_jitted[s]",
    ],
    var_name="Timing Type",
    value_name="Time [s]",
)

# rename the values in Timing Type to be more readable
df_total_pred_time["Timing Type"] = df_total_pred_time["Timing Type"].replace(
    {
        "total_pred_time_no_jit[s]": "w/o JIT",
        "total_pred_time_jitting[s]": "w/ JIT",
        "total_pred_time_jitted[s]": "JIT-compiled",
    }
)
n_cpu = 32
df_total_pred_time["Time [CPUh]"] = df_total_pred_time["Time [s]"] * n_cpu / 3600

# %% total prediction time (Uncompiled, JIT compiling, JIT compiled and the combined timing but for graph size)
df_wt_pred_time = df[df["n_probes"].isin([1, 10, 100, 1000, 10000])]
df_wt_pred_time.columns = df_wt_pred_time.columns.str.replace(
    "n_probes", r"$n_\mathrm{p}$"
)

df_wt_pred_time["total_pred_time_no_jit[CPUh]"] = (
    df_wt_pred_time["total_pred_time_no_jit[s]"] * n_cpu / 3600
)
df_wt_pred_time["total_pred_time_jitting[CPUh]"] = (
    df_wt_pred_time["total_pred_time_jitting[s]"] * n_cpu / 3600
)
df_wt_pred_time["total_pred_time_jitted[CPUh]"] = (
    df_wt_pred_time["total_pred_time_jitted[s]"] * n_cpu / 3600
)

fig, axes = plt.subplots(2, 1, figsize=(4.5, 7))
sns.lineplot(
    data=df_total_pred_time,
    x="size_of_graph",
    y="Time [CPUh]",
    hue="Timing Type",
    ax=axes[0],
)
sns.lineplot(
    data=df,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    label="PyWake",
    ax=axes[0],
)
axes[0].set_yscale("log")
axes[0].set_xlabel(r"$|G| \times 10^{-6}$ [-]")
# move legend to the right outside of the plot
axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# add a text box with (a)
axes[0].text(
    -0.2,
    1,
    "(a)",
    transform=axes[0].transAxes,
    # fontsize=12,
    va="top",
)
pl2 = sns.lineplot(
    data=df_wt_pred_time,
    x="n_wt",
    y="pywake_simulation_time[CPUh]",
    hue=r"$n_\mathrm{p}$",
    palette="tab10",
    linestyle="--",
    ax=axes[1],
    legend=False,
)
pl1 = sns.lineplot(
    data=df_wt_pred_time,
    x="n_wt",
    y="total_pred_time_jitted[CPUh]",
    hue=r"$n_\mathrm{p}$",
    palette="tab10",
    ax=axes[1],
    legend=True,
)

axes[1].legend(title=r"$n_\mathrm{p}$", bbox_to_anchor=(1.05, 1), loc="upper left")
# add a second legend with a black line and a dashed line for GNO and PyWake with another title
org_legend_elements = axes[1].get_legend_handles_labels()
org_labels = org_legend_elements[1]
org_handles = org_legend_elements[0]
legend_elements = [
    Line2D([0], [0], color="k", lw=2, label="GNO"),
    Line2D([0], [0], color="k", lw=2, linestyle="--", label="PyWake"),
]
title_proxy = Rectangle((0, 0), 0, 0, color="w")
legend_handles = (
    [title_proxy]
    + [legend_elements[0], title_proxy, legend_elements[1]]
    + 2 * [title_proxy]
    + org_handles
)
legend_labels = (
    ["Model"]
    + ["GNO", "(JIT-compiled)", "PyWake"]
    + [r"", r"$n_\mathrm{p}$"]
    + org_labels
)
axes[1].legend(
    handles=legend_handles,
    labels=legend_labels,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

axes[1].set_yscale("log")
axes[1].set_xlabel(r"$n_\mathrm{wt}$ [-]")
axes[1].set_ylabel("Time [CPUh]")

# add a text box with (b)
axes[1].text(
    -0.2,
    1,
    "(b)",
    transform=axes[1].transAxes,
    # fontsize=12,
    va="top",
)

for i, ax in enumerate(axes):
    if i == 0:
        xlims = [df.min()["size_of_graph"], df.max()["size_of_graph"]]
    else:
        xlims = [df_wt_pred_time["n_wt"].min(), df_wt_pred_time["n_wt"].max()]

    # set all y limits to be the same
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlim(int(xlims[0]), int(xlims[1]))

plt.savefig(
    os.path.join(fig_folder_path, "total_prediction_time_plots.pdf"),
    bbox_inches="tight",
)

# %% For presentation
df["total_pred_time_jitted[CPUh]"] = df["total_pred_time_jitted[s]"] * n_cpu / 3600

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.lineplot(
    data=df,
    x="size_of_graph",
    y="total_pred_time_jitted[CPUh]",
    ax=ax,
    label="GNO",
)
sns.lineplot(
    data=df,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    label="PyWake",
    ax=ax,
)
ax.set_yscale("log")
ax.set_xlabel(r"$|G| \times 10^{-6}$ [-]")
ax.set_ylabel("Time [CPUh]")
# move legend to the right outside of the plot
ax.legend(loc="upper left")

xlims = [df.min()["size_of_graph"], df.max()["size_of_graph"]]

ax.set_ylim(1e-7, 1e-3)
ax.set_xlim(int(xlims[0]), int(xlims[1]))

plt.savefig(
    os.path.join(fig_folder_path, "total_prediction_simple_presentation.pdf"),
    bbox_inches="tight",
)


# %% Plot in article (embedding, wt processing, probe processing, decoding)
## Prep for plot
df_embedding_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["embedding_time[s]_jitting", "embedding_time[s]_jitted"],
    var_name="Timing Type",
    value_name="Time [s]",
)
df_embedding_time["Time [CPUh]"] = df_embedding_time["Time [s]"] * n_cpu / 3600


df_wt_processing_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["wt_processing_time[s]_jitting", "wt_processing_time[s]_jitted"],
    var_name="Timing Type",
    value_name="Time [s]",
)
df_wt_processing_time["Time [CPUh]"] = df_wt_processing_time["Time [s]"] * n_cpu / 3600


df_probe_processing_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["probe_processing_time[s]_jitting", "probe_processing_time[s]_jitted"],
    var_name="Timing Type",
    value_name="Time [s]",
)
df_probe_processing_time["Time [CPUh]"] = (
    df_probe_processing_time["Time [s]"] * n_cpu / 3600
)


df_decoding_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["decoding_time[s]_jitting", "decoding_time[s]_jitted"],
    var_name="Timing Type",
    value_name="Time [s]",
)
df_decoding_time["Time [CPUh]"] = df_decoding_time["Time [s]"] * n_cpu / 3600


df_summed_model_time = df.melt(
    id_vars=["size_of_graph"],
    value_vars=["total_model_time[s]_jitting", "total_model_time[s]_jitted"],
    var_name="Timing Type",
    value_name="Time [s]",
)
df_summed_model_time["Time [CPUh]"] = df_summed_model_time["Time [s]"] * n_cpu / 3600

# Define the order and color palette for the hue
hue_order = ["jitting", "jitted"]
palette = {"jitting": "blue", "jitted": "red"}

# Create a new column for simplified hue categories
for df_melted in [
    df_embedding_time,
    df_wt_processing_time,
    df_probe_processing_time,
    df_decoding_time,
]:
    df_melted["Timing Category"] = df_melted["Timing Type"].apply(
        lambda x: "jitting" if "jitting" in x else "jitted"
    )


# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig, axes = plt.subplots(2, 2, figsize=(6, 5.5), sharex=True)
sns.lineplot(
    data=df_embedding_time,
    x="size_of_graph",
    y="Time [CPUh]",
    hue="Timing Category",
    hue_order=hue_order,
    palette=palette,
    ax=axes[0, 0],
    legend=False,
)
sns.lineplot(
    data=df_wt_pred_time,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    palette="tab10",
    linestyle="--",
    ax=axes[0, 0],
    legend=False,
)

axes[0, 0].set_ylabel("Embedding [CPUh]")
sns.lineplot(
    data=df_wt_processing_time,
    x="size_of_graph",
    y="Time [CPUh]",
    hue="Timing Category",
    hue_order=hue_order,
    palette=palette,
    ax=axes[0, 1],
    legend=False,
)
sns.lineplot(
    data=df_wt_pred_time,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    palette="tab10",
    linestyle="--",
    ax=axes[0, 1],
    legend=False,
)

sns.lineplot(
    data=df_probe_processing_time,
    x="size_of_graph",
    y="Time [CPUh]",
    hue="Timing Category",
    hue_order=hue_order,
    palette=palette,
    ax=axes[1, 0],
    legend=False,
)
sns.lineplot(
    data=df_wt_pred_time,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    palette="tab10",
    linestyle="--",
    ax=axes[1, 0],
    legend=False,
)

axes[1, 0].set_ylabel("Probe processing [CPUh]")
sns.lineplot(
    data=df_decoding_time,
    x="size_of_graph",
    y="Time [CPUh]",
    hue="Timing Category",
    hue_order=hue_order,
    palette=palette,
    ax=axes[1, 1],
    legend=False,
)
sns.lineplot(
    data=df_wt_pred_time,
    x="size_of_graph",
    y="pywake_simulation_time[CPUh]",
    color="gray",
    palette="tab10",
    linestyle="--",
    ax=axes[1, 1],
    legend=False,
)

legend_elements = [
    Line2D([0], [0], color="blue", lw=2, label="JIT-compiling"),
    Line2D([0], [0], color="red", lw=2, label="JIT-compiled"),
    Line2D([0], [0], color="gray", lw=2, label="PyWake"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=3)


for ax in axes.flat:
    ax.set_yscale("log")
    ax.set_xlabel(r"$|G| \times 10^{-6}$ [-]")
    # add a textbox with the letter in the top left corner
    ax.text(
        0.05,
        0.95,
        f"({chr(97 + list(axes.flat).index(ax))})",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )
    xlims = [df.min()["size_of_graph"], df.max()["size_of_graph"]]

for ax in axes.flat:
    ax.label_outer()

axes[1, 1].set_ylabel("Decoding [CPUh]")
axes[0, 1].set_ylabel("WT processing [CPUh]")
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the legend
for ax in axes.flat:
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlim(0, 4)
plt.savefig(
    os.path.join(fig_folder_path, "model_time_breakdown.pdf"), bbox_inches="tight"
)


# %% Histogram of graph sizes for different n_probes, only for figuring out if i needed more data in the calculations
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="size_of_graph", hue="n_probes", palette="tab10", bins=50)
plt.legend(title="Number of Probes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel(r"$|G| \times 10^{-6}$")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
