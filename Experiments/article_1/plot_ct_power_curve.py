import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from py_wake.examples.data.dtu10mw import DTU10MW

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))


from utils.plotting import matplotlib_set_rcparams

matplotlib_set_rcparams("paper")


base_dir = os.path.abspath(".")  # run with command line from repository base dir
# base_dir = os.path.abspath("../..") # run from Experiments/articles_plotting/ with iPython


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = colors[:4]


wt = DTU10MW()
ws = np.linspace(0, 29, 1000)
CT = wt.ct(ws)
# for ct where
power = wt.power(ws)

cut_in = 4
cut_out = 25
rated = 11.4

# %% Plotting

fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True)

axes[0].plot(ws, power / 1e6, color="k")
axes[0].set_ylabel(r"$\mathrm{Power}$ [$\mathrm{MW}$]")
axes[1].plot(ws, CT, color="k")
axes[1].set_ylabel(r"$C_\mathrm{T}$ [-]")
axes[1].set_xlabel(r"$U$ [$\mathrm{ms}^{-1}$]")
# add text box (a) and (b) for figure numbering
axes[0].text(0.035, 0.8, "(a)", transform=axes[0].transAxes, fontsize=12)
axes[1].text(0.035, 0.8, "(b)", transform=axes[1].transAxes, fontsize=12)

for ax in axes:
    # ax.axvline(cut_in, color="k", linestyle="--", linewidth=0.8)
    # ax.axvline(rated, color="k", linestyle="--", linewidth=0.8)
    # ax.axvline(cut_out, color="k", linestyle="--", linewidth=0.8)
    ax.axvspan(0, cut_in, color=colors[0], alpha=0.2, label="I")
    ax.axvspan(cut_in, rated, color=colors[1], alpha=0.2, label="II")
    ax.axvspan(rated, cut_out, color=colors[2], alpha=0.2, label="III")
    ax.axvspan(cut_out, 30, color=colors[3], alpha=0.2, label="IV")

    ax.set_xlim(0, 29)
axes[0].legend(
    loc="upper right", fontsize=10, ncol=4, bbox_to_anchor=(0.9, 1.4), frameon=True
)

# plt.plot(ws, power / 1e6, label="Power [MW]")
plt.savefig(
    os.path.join(base_dir, "assets/DTU10MW_ct_power_curve.pdf"), bbox_inches="tight"
)

# %%
