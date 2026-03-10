"""Plot the trained RBF encoder from the Vj8 model.

This script visualizes the learned Radial Basis Function (RBF) kernels
from the trained GNO model stored in assets/best_model_Vj8/, comparing
the initial (untrained) kernels with the learned ones.
"""

import sys
from pathlib import Path

import flax.serialization
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import matplotlib_set_rcparams  # noqa: E402


def load_rbf_params(model_dir: Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load mu and beta parameters from a trained model.

    Args:
        model_dir: Path to the model directory containing best_params.msgpack

    Returns:
        Tuple of (mu, beta) arrays
    """
    params_path = model_dir / "best_params.msgpack"
    params = flax.serialization.msgpack_restore(params_path.read_bytes())
    rbf_params = params["params"]["RBFEncoder_0"]
    return jnp.array(rbf_params["mu"]), jnp.array(rbf_params["beta"])


def get_initial_rbf_params(
    num_kernels: int, d_min: float, d_max: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the initial (untrained) RBF parameters.

    These match the initialization in models/RBF_encoder.py

    Args:
        num_kernels: Number of RBF basis functions
        d_min: Minimum distance
        d_max: Maximum distance

    Returns:
        Tuple of (mu, beta) arrays
    """
    mu = jnp.linspace(d_min, d_max, num_kernels)
    beta = jnp.ones(num_kernels) * (num_kernels / (d_max - d_min))
    return mu, beta


def cosine_cutoff(distances: jnp.ndarray, d_max: float) -> jnp.ndarray:
    """Smooth cosine cutoff function."""
    cutoff = 0.5 * (jnp.cos(jnp.pi * distances / d_max) + 1.0)
    cutoff = jnp.where(distances < d_max, cutoff, 0.0)
    return cutoff


def compute_rbf(
    distances: jnp.ndarray, mu: jnp.ndarray, beta: jnp.ndarray, d_max: float
) -> jnp.ndarray:
    """Compute RBF features for input distances.

    Args:
        distances: Array of distances to encode
        mu: RBF kernel centers
        beta: RBF kernel scaling factors
        d_max: Maximum distance for cutoff

    Returns:
        RBF encoded distances of shape (len(distances), num_kernels)
    """
    return cosine_cutoff(distances, d_max)[..., None] * jnp.exp(
        -beta * (distances[..., None] - mu) ** 2
    )


def get_rbf_colors_and_styles(num_kernels: int) -> tuple[list, list]:
    """Get matching colors and linestyles for symmetrical kernels.

    Args:
        num_kernels: Number of RBF kernels

    Returns:
        Tuple of (colors, line_styles) lists
    """
    num_colors = (num_kernels + 1) // 2
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_colors)]
    line_styles = ["-", "--"]

    if num_kernels % 2 == 0:
        colors = colors + colors[::-1]
        line_styles = [line_styles[0]] * (num_kernels // 2) + [line_styles[1]] * (num_kernels // 2)
    else:
        colors = colors + colors[-2::-1]
        line_styles = [line_styles[0]] * (1 + (num_kernels // 2)) + [line_styles[1]] * (
            num_kernels // 2
        )

    return colors, line_styles


def plot_rbf_kernels(
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    d_min: float = -1.0,
    d_max: float = 1.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    title_inside: bool = False,
    show_legend: bool = True,
    legend_loc: str = "above",
) -> plt.Axes:
    """Plot RBF kernel functions.

    Args:
        mu: RBF kernel centers
        beta: RBF kernel scaling factors
        d_min: Minimum distance for plotting
        d_max: Maximum distance for plotting
        ax: Matplotlib axes to plot on (creates new figure if None)
        title: Optional title for the plot
        title_inside: If True, place title inside plot (top-left corner)
        show_legend: Whether to show the legend
        legend_loc: Legend location - "above" for above plot, "inside" for inside plot

    Returns:
        The matplotlib axes object
    """
    num_kernels = len(mu)
    distances = jnp.linspace(d_min, d_max, 200)
    encoded_distances = compute_rbf(distances, mu, beta, d_max)

    colors, line_styles = get_rbf_colors_and_styles(num_kernels)

    if ax is None:
        _, ax = plt.subplots()

    # Add vertical dashed grey line at x=0 for reference
    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8, zorder=0)

    for i in range(num_kernels):
        ax.plot(
            distances,
            encoded_distances[:, i],
            label=r"{\large $\mu_\mathrm{RBF}^{(" + f"{i + 1}" + r")}$}" + f"={mu[i]:.2f}",
            color=colors[i],
            linestyle=line_styles[i],
        )

    # Add buffer space at top for title label
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.20 * (ymax - ymin))

    ax.set_xlabel(r"$d_{ij}/d_c$ [-]")
    ax.set_ylabel(r"$\varphi_\mathrm{RBF}(d_{ij})$ [-]")
    if title:
        if title_inside:
            ax.text(
                0.05,
                0.95,
                title,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                horizontalalignment="left",
            )
        else:
            ax.set_title(title)
    if show_legend:
        if legend_loc == "above":
            ax.legend(bbox_to_anchor=(0.45, 1.65), loc="upper center", ncol=3)
        else:
            ax.legend(loc="best", ncol=2, fontsize=7)

    return ax


def plot_parameter_comparison(
    values_initial: jnp.ndarray,
    values_trained: jnp.ndarray,
    ax: plt.Axes,
    ylabel: str,
    title: str | None = None,
    title_inside: bool = False,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot bar chart comparing initial and trained parameter values.

    Args:
        values_initial: Initial parameter values
        values_trained: Trained parameter values
        ax: Matplotlib axes to plot on
        ylabel: Y-axis label
        title: Optional title for the plot
        title_inside: If True, place title inside plot (top-left corner)
        show_legend: Whether to show the legend

    Returns:
        The matplotlib axes object
    """
    num_kernels = len(values_initial)
    x = np.arange(num_kernels)
    width = 0.35

    ax.bar(x - width / 2, values_initial, width, label="Initial", color="steelblue")
    ax.bar(x + width / 2, values_trained, width, label="Trained", color="darkorange")

    ax.set_xlabel("Kernel index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(num_kernels)])

    if show_legend:
        ax.legend(loc="best", fontsize=8)

    # Add buffer space at top for title label
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))

    if title:
        if title_inside:
            ax.text(
                0.05,
                0.95,
                title,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                horizontalalignment="left",
            )
        else:
            ax.set_title(title)

    return ax


if __name__ == "__main__":
    matplotlib_set_rcparams(publication="paper")
    # Enable LaTeX rendering for better math symbol sizing control
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    # Configuration
    d_min = -1.0
    d_max = 1.0
    num_kernels = 9

    # Paths
    output_dir = PROJECT_ROOT / "assets" / "RBF"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "assets" / "best_model_Vj8"

    # Load trained RBF parameters
    mu_trained, beta_trained = load_rbf_params(model_dir)
    print(f"Loaded RBF parameters from {model_dir}")
    print(f"  mu (trained): {mu_trained}")
    print(f"  beta (trained): {beta_trained}")

    # Get initial (untrained) RBF parameters
    mu_initial, beta_initial = get_initial_rbf_params(num_kernels, d_min, d_max)
    print("\nInitial RBF parameters:")
    print(f"  mu (initial): {mu_initial}")
    print(f"  beta (initial): {beta_initial}")

    # Single figure size from rcparams: (3.347, 2.0)
    single_width = 3.347
    single_height = 2.0

    # --- Standalone plots (initial and trained) ---
    standalone_configs = [
        ("initial", mu_initial, beta_initial),
        ("trained", mu_trained, beta_trained),
    ]
    for name, mu, beta in standalone_configs:
        fig, ax = plt.subplots(figsize=(single_width, single_height))
        plot_rbf_kernels(mu=mu, beta=beta, d_min=d_min, d_max=d_max, ax=ax, show_legend=True)
        save_path = output_dir / f"RBF_encoder_{name}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure saved to {save_path}")

    # --- Plot 3: Side-by-side comparison (2x2 layout) ---
    # Top row: RBF kernels, Bottom row: parameter bar charts
    # Use gridspec to give bottom row slightly more height for bar charts
    fig3, axes = plt.subplots(
        2,
        2,
        figsize=(single_width * 1.5, single_height * 2.0),
        gridspec_kw={"height_ratios": [1, 0.8]},
    )
    (ax_rbf_init, ax_rbf_trained), (ax_mu, ax_beta) = axes

    # Top-left: Initial RBF kernels
    plot_rbf_kernels(
        mu=mu_initial,
        beta=beta_initial,
        d_min=d_min,
        d_max=d_max,
        ax=ax_rbf_init,
        title="(a) Initial RBF",
        title_inside=True,
        show_legend=False,
    )

    # Top-right: Trained RBF kernels
    plot_rbf_kernels(
        mu=mu_trained,
        beta=beta_trained,
        d_min=d_min,
        d_max=d_max,
        ax=ax_rbf_trained,
        title="(b) Trained RBF",
        title_inside=True,
        show_legend=False,
    )

    # Bottom-left: mu comparison bar chart
    plot_parameter_comparison(
        values_initial=mu_initial,
        values_trained=mu_trained,
        ax=ax_mu,
        ylabel=r"$\mu$ [-]",
        title="(c) Kernel centers",
        title_inside=True,
        show_legend=False,
    )

    # Bottom-right: beta comparison bar chart
    plot_parameter_comparison(
        values_initial=beta_initial,
        values_trained=beta_trained,
        ax=ax_beta,
        ylabel=r"$\beta$ [-]",
        title="(d) Kernel widths",
        title_inside=True,
        show_legend=False,
    )

    # Create shared legend for RBF plots and bar charts at top of figure
    colors, line_styles = get_rbf_colors_and_styles(num_kernels)
    from matplotlib.patches import Patch

    # Create kernel handles
    kernel_handles = [
        plt.Line2D([0], [0], color=colors[i], linestyle=line_styles[i]) for i in range(num_kernels)
    ]
    kernel_labels = [rf"{{\large $\varphi_{{{i + 1}}}$}}" for i in range(num_kernels)]

    # Bar chart handles
    initial_handle = Patch(facecolor="steelblue")
    trained_handle = Patch(facecolor="darkorange")

    # Blank handle for spacing
    blank_handle = plt.Line2D([0], [0], color="none", linestyle="none")

    # Arrange for column-major fill with ncol=6 to get:
    # Row 1: φ1, φ2, φ3, φ4, φ5, Initial
    # Row 2: φ6, φ7, φ8, φ9, blank, Trained
    all_handles = [
        kernel_handles[0], kernel_handles[5],  # col 0
        kernel_handles[1], kernel_handles[6],  # col 1
        kernel_handles[2], kernel_handles[7],  # col 2
        kernel_handles[3], kernel_handles[8],  # col 3
        kernel_handles[4], blank_handle,       # col 4
        initial_handle, trained_handle,        # col 5
    ]
    all_labels = [
        kernel_labels[0], kernel_labels[5],
        kernel_labels[1], kernel_labels[6],
        kernel_labels[2], kernel_labels[7],
        kernel_labels[3], kernel_labels[8],
        kernel_labels[4], "",
        "Initial", "Trained",
    ]

    fig3.legend(
        all_handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.0,
        labelspacing=0.15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=0.3, w_pad=0.5)  # Leave space for legend
    save_path_comparison = output_dir / "RBF_encoder_comparison.pdf"
    plt.savefig(save_path_comparison, bbox_inches="tight")
    plt.close(fig3)
    print(f"Figure saved to {save_path_comparison}")
