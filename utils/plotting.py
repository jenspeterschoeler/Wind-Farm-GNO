"""Plotting utilities for training visualization and evaluation."""

import jraph
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.path import Path


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    assert senders is not None and receivers is not None, "Senders and receivers must be present"
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])  # type: ignore[index]
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]), edge_feature=edges[e])  # type: ignore[index]
    return nx_graph


def composite_marker():
    circle = Path.circle(radius=0.175)  # "o" shape
    vert_line = Path([[0, -1], [0, 1]], [Path.MOVETO, Path.LINETO])  # "|" shape

    # Combine the two markers
    vertices = np.concatenate([circle.vertices, vert_line.vertices])
    codes = np.concatenate([circle.codes, vert_line.codes])  # type: ignore[call-overload]

    return Path(vertices, codes)


def wind_turbine_marker():
    """Create a wind turbine marker: tri_up shape with a small circle at the hub."""
    # Small filled circle at hub (center-top area)
    circle = Path.circle(center=(0, 0.3), radius=0.15)

    # Three lines radiating from hub (like turbine blades) - simplified tri_up
    blade1 = Path([[0, 0.3], [0, 1]], [Path.MOVETO, Path.LINETO])  # Top blade
    blade2 = Path([[0, 0.3], [-0.6, -0.6]], [Path.MOVETO, Path.LINETO])  # Bottom-left
    blade3 = Path([[0, 0.3], [0.6, -0.6]], [Path.MOVETO, Path.LINETO])  # Bottom-right

    # Combine all paths
    vertices = np.concatenate([circle.vertices, blade1.vertices, blade2.vertices, blade3.vertices])
    codes = np.concatenate([circle.codes, blade1.codes, blade2.codes, blade3.codes])  # type: ignore[call-overload]

    return Path(vertices, codes)


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)

    edge_labels = {
        x[:2]: np.round(nx_graph.get_edge_data(*x)["edge_feature"][0], 0) for x in nx_graph.edges
    }
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels)
    # Plot custom markers on nodes
    wt_marker = composite_marker()  # get the custom wt marker
    for _node, (x, y) in pos.items():
        plt.scatter(x, y, marker=wt_marker, color="black", s=500, zorder=5)
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        font_size=5,
        font_color="black",
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    plt.show()


def plot_mode_amplitudes(branch_modes, trunk_amplitues):
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, amplitudes, title in zip(
        axes, [branch_modes, trunk_amplitues], ["Branch Modes", "Trunk Amplitudes"]
    ):
        sns.barplot(x=np.arange(len(amplitudes)), y=amplitudes, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Amplitude")
    plt.axis("off")
    return figure, axes


def plot_loss_history(
    loss_hist: list,
    val_hist: list | None = None,
    val_epochs=None,
    additional_metrics: dict | None = None,
):
    figure, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs = np.arange(len(loss_hist))
    ax.semilogy(epochs, loss_hist, label="Training Loss")
    if val_hist is not None:
        assert val_epochs is not None, "val_epochs must be provided if val_hist is provided"
        ax.semilogy(val_epochs, val_hist, label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return figure, ax


def plot_pred_true_err_contour(trunk_input, prediction, output, normalize_distance=False):
    error = output - prediction
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    if normalize_distance:
        D = 178.3
        trunk_input /= D

    for _i, (ax, data, title) in enumerate(
        zip(axes, [prediction, output, error], ["Prediction", "Output", "Error"])
    ):
        ax.tricontourf(
            trunk_input[:, 0],
            trunk_input[:, 1],
            data,
        )
        ax.set_title(title)
        ax.axis("equal")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("$y$[m/$D$]")
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Velocity [m/s]")

        cbar.formatter = ticker.ScalarFormatter(useMathText=False)
        cbar.formatter.set_scientific(False)  # type: ignore[attr-defined]
        cbar.update_ticks()
    ax.set_xlabel("$x$[m/$D$]")

    fig.tight_layout(rect=[0, 0, 0.9, 1])  # type: ignore[arg-type]
    return fig, axes


def plot_qq_plot(outputs, predictions, metrics_dict):
    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(outputs, predictions)
    plt.plot(
        [np.min(outputs), np.max(outputs)],
        [np.min(outputs), np.max(outputs)],
        "k--",
    )
    plt.xlabel("True")
    plt.ylabel("Predicted")
    textstr = f"MSE: {metrics_dict['mse']:.3f}, \nRMSE: {metrics_dict['rmse']:.3f}\nMAE: {metrics_dict['mae']:.3f}\nMAPE: {metrics_dict['mape']:.3f}"
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    return fig, ax


def get_wt_and_probe_idxs(graphs, probe_graphs):
    unique_wt_senders_receivers = np.unique(np.concat([graphs.senders, graphs.receivers]))
    unique_probe_receivers = np.unique(probe_graphs.receivers)
    return unique_wt_senders_receivers, unique_probe_receivers


def plot_crossstream_predictions(predictions, targets, y_coords, marker=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if marker:
        marker_kw1 = {"marker": "o", "markersize": 5}
        marker_kw2 = {"marker": "x", "markersize": 5}
    else:
        marker_kw1 = {}
        marker_kw2 = {}

    ax.plot(predictions, y_coords, label="Predictions", color="green", **marker_kw1)  # type: ignore[arg-type]
    ax.plot(targets, y_coords, label="Targets", color="black", **marker_kw2)  # type: ignore[arg-type]
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def plot_probe_graph_fn(
    graphs,
    probe_graphs,
    node_positions,
    include_wt_nodes=True,
    include_wt_edges=True,
    include_probe_nodes=True,
    include_probe_edges=True,
    wt_color="b",
    probe_color="r",
    ax=None,
    edge_linewidth=0.5,
    wt_node_size=100,
    edge_color="g",
    edge_alpha=1.0,
    wt_marker="2",
    wt_edgecolor=None,
    wt_linewidth=0.5,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

    unique_wt_senders_receivers, unique_probe_receivers = get_wt_and_probe_idxs(
        graphs, probe_graphs
    )
    np.intersect1d(unique_wt_senders_receivers, unique_probe_receivers)

    if include_probe_edges:
        probe_edge_coordinates = [
            node_positions[probe_graphs.senders, :],
            node_positions[probe_graphs.receivers, :],
        ]

    wf_edge_coordinates = [
        node_positions[graphs.receivers, :] + graphs.edges[:, :-1],
        node_positions[graphs.receivers, :],
    ]

    if include_wt_edges:
        for i, (sender, receiver) in enumerate(zip(*wf_edge_coordinates)):
            ax.plot(
                [sender[0], receiver[0]],
                [sender[1], receiver[1]],
                c=edge_color,
                alpha=edge_alpha,
                ls="-",
                linewidth=edge_linewidth,
                label="WT edges" if i == 0 else "",
            )

    if include_probe_edges:
        for i, (sender, receiver) in enumerate(zip(*probe_edge_coordinates)):
            ax.plot(
                [sender[0], receiver[0]],
                [sender[1], receiver[1]],
                c="k",
                alpha=0.5,
                ls="-",
                linewidth=edge_linewidth,
                label="Probe edges" if i == 0 else "",
            )

    if include_wt_nodes:
        wt_positions = node_positions[unique_wt_senders_receivers, :]
        # If edge color specified for unfilled marker, plot two layers for outline effect
        if wt_edgecolor is not None and wt_marker in ["2", "1", "3", "4", "+", "x", "|", "_"]:
            # First layer: thicker marker in edge color (acts as outline)
            ax.scatter(
                wt_positions[:, 0],
                wt_positions[:, 1],
                c=wt_edgecolor,
                marker=wt_marker,
                s=wt_node_size,
                linewidths=wt_linewidth * 3,  # Thicker lines for outline effect
                zorder=9,
            )
            # Second layer: thinner marker in main color
            ax.scatter(
                wt_positions[:, 0],
                wt_positions[:, 1],
                c=wt_color,
                marker=wt_marker,
                s=wt_node_size,
                linewidths=wt_linewidth,
                label="WT nodes",
                zorder=10,
            )
        else:
            ax.scatter(
                wt_positions[:, 0],
                wt_positions[:, 1],
                c=wt_color,
                marker=wt_marker,
                s=wt_node_size,
                label="WT nodes",
                zorder=10,
                edgecolors=wt_edgecolor,
                linewidths=wt_linewidth,
            )
    if include_probe_nodes:
        probe_positions = node_positions[unique_probe_receivers, :]
        ax.scatter(
            probe_positions[:, 0],
            probe_positions[:, 1],
            c=probe_color,
            marker="o",
            s=20,
            label="Probe nodes",
            zorder=10,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def matplotlib_set_rcparams(publication: str = "paper"):
    def set_rcparamspaper():
        plt.style.use("classic")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.figsize"] = (3.347, 2.0)
        plt.rcParams["font.size"] = 11
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["axes.titlesize"] = 11
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        # Set the font size of the legend
        plt.rcParams["legend.fontsize"] = 9

        plt.rcParams["axes.grid"] = True  # Enable grid
        plt.rcParams["grid.alpha"] = 0.3  # Set grid transparency
        plt.rcParams["grid.color"] = "gray"  # Optional: grid color

    if publication == "paper":
        set_rcparamspaper()

    elif publication == "presentation":
        plt.style.use("classic")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 16
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 16
        plt.rcParams["xtick.labelsize"] = 16
        plt.rcParams["ytick.labelsize"] = 16
        # Set the font size of the legend
        plt.rcParams["legend.fontsize"] = 14

        plt.rcParams["axes.grid"] = True  # Enable grid
        plt.rcParams["grid.alpha"] = 0.3  # Set grid transparency
        plt.rcParams["grid.color"] = "gray"  # Optional: grid color

    else:
        raise ValueError(
            f"Publication style '{publication}' not recognized. Use 'paper' or 'presentation'."
        )


def plot_dataset_comparison(
    data,
    dataset_name="Dataset",
    turbine_diameter=178.3,
    ax=None,
    show_colorbar=True,
):
    """
    Plot a single graph datapoint showing turbine layout and flow field.

    Args:
        data: PyTorch Geometric Data object with attributes:
            - pos: Node positions (turbines + probes)
            - output_features: Flow field values at probe locations
            - trunk_inputs: Probe positions
            - global_features: [wind_speed, turbulence_intensity]
            - n_wt: Number of wind turbines (optional)
        dataset_name: Name of the dataset for title
        turbine_diameter: Turbine diameter in meters for normalization
        ax: Matplotlib axis to plot on (creates new if None)
        show_colorbar: Whether to show colorbar

    Returns:
        ax: Matplotlib axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    positions = data.pos.numpy() if hasattr(data.pos, "numpy") else data.pos
    trunk_inputs = (
        data.trunk_inputs.numpy() if hasattr(data.trunk_inputs, "numpy") else data.trunk_inputs
    )
    outputs = (
        data.output_features.numpy()
        if hasattr(data.output_features, "numpy")
        else data.output_features
    )
    global_features = (
        data.global_features.numpy()
        if hasattr(data.global_features, "numpy")
        else data.global_features
    )

    # Get number of turbines
    if hasattr(data, "n_wt"):
        n_wt = int(data.n_wt)
    else:
        # Assume first nodes are turbines, rest are probes
        n_wt = len(positions) - len(trunk_inputs)

    # Normalize positions by diameter
    D = turbine_diameter
    positions_norm = positions / D
    trunk_inputs_norm = trunk_inputs / D

    # Extract turbine and probe positions
    wt_positions = positions_norm[:n_wt]
    probe_positions = trunk_inputs_norm

    # Extract wind speed and TI from global features
    # Handle both 1D and 2D global features
    if global_features.ndim > 1:
        ws_inf = global_features[0, 0]
        ti_inf = global_features[0, 1]
    else:
        ws_inf = global_features[0]
        ti_inf = global_features[1]

    # Handle output features (could be velocity only or [velocity, tke])
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        velocity = outputs[:, 0]  # First channel is velocity
    else:
        velocity = outputs.flatten()

    # Plot flow field as scatter with colors
    scatter = ax.scatter(
        probe_positions[:, 0],
        probe_positions[:, 1],
        c=velocity,
        cmap="viridis",
        s=5,
        alpha=0.6,
        label="Probe locations",
    )

    # Plot wind turbines
    ax.scatter(
        wt_positions[:, 0],
        wt_positions[:, 1],
        c="red",
        marker="2",
        s=200,
        label=f"Wind turbines (n={n_wt})",
        zorder=10,
    )

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Velocity [m/s]")

    # Labels and title
    ax.set_xlabel(r"$x$ [D]")
    ax.set_ylabel(r"$y$ [D]")
    ax.set_title(
        f"{dataset_name}\nWS={ws_inf:.1f} m/s, TI={ti_inf:.2f}, n_probes={len(probe_positions)}"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return ax
