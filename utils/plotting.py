from typing import Dict, List

import jraph
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.path import Path


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def composite_marker():
    circle = Path.circle(radius=0.175)  # "o" shape
    vert_line = Path([[0, -1], [0, 1]], [Path.MOVETO, Path.LINETO])  # "|" shape

    # Combine the two markers
    vertices = np.concatenate([circle.vertices, vert_line.vertices])
    codes = np.concatenate([circle.codes, vert_line.codes])

    return Path(vertices, codes)


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:

    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)

    edge_labels = {
        x[:2]: np.round(nx_graph.get_edge_data(*x)["edge_feature"][0], 0)
        for x in nx_graph.edges
    }
    # node_labels = {
    #     n: np.round(nx_graph.nodes[n]["node_feature"][0], 0) for n in nx_graph.nodes
    # }
    # nx.draw(
    #     nx_graph,
    #     pos=pos,
    #     node_shape=None,
    #     with_labels=True,
    #     node_size=500,
    #     font_color="yellow",
    # )
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels)
    # Plot custom markers on nodes
    wt_marker = composite_marker()  # get the custom wt marker
    for node, (x, y) in pos.items():
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
    loss_hist: List,
    val_hist: List = None,
    val_epochs=None,
    additional_metrics: Dict = None,
):
    figure, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs = np.arange(len(loss_hist))
    ax.semilogy(epochs, loss_hist, label="Training Loss")
    if val_hist is not None:
        assert (
            val_epochs is not None
        ), "val_epochs must be provided if val_hist is provided"
        ax.semilogy(val_epochs, val_hist, label="Validation Loss")
    # create text box with additional metrics
    # textstr = "" # TODO FIX this after having separated the metrics into components
    # if additional_metrics is not None:
    #     for metric, values in additional_metrics.items():
    #         metric_textstr = "".join(
    #             [str(np.round(tx, 3)) + r"\t" for tx in list(values)]
    #         )
    #         textstr += f"{metric}: " + metric_textstr + "\n"
    #     ax.text(
    #         0.75,
    #         0.95,
    #         textstr,
    #         transform=ax.transAxes,
    #         fontsize=12,
    #         verticalalignment="top",
    #         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    #     )
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return figure, ax


def plot_pred_true_err_contour(
    trunk_input, prediction, output, normalize_distance=False
):

    error = output - prediction
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    if normalize_distance:
        D = 178.3
        trunk_input /= D

    # v_min = np.min([prediction.min(), output.min()])
    # v_max = np.max([prediction.max(), output.max()])
    # error_v_min = error.min()
    # error_v_max = error.max()

    for i, (ax, data, title) in enumerate(
        zip(axes, [prediction, output, error], ["Prediction", "Output", "Error"])
    ):
        # if title == "Error":
        #     vmin, vmax = error_v_min, error_v_max
        # else:
        #     vmin, vmax = v_min, v_max
        ax.tricontourf(
            trunk_input[:, 0],
            trunk_input[:, 1],
            data,
            # vmin=vmin,
            # vmax=vmax,
        )
        ax.set_title(title)
        ax.axis("equal")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("$y$[m/$D$]")
        # create colorbar pr ax
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Velocity [m/s]")

        # Format colorbar to disable scientific notation
        cbar.formatter = ticker.ScalarFormatter(useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()
    ax.set_xlabel("$x$[m/$D$]")

    # # add shared colorbar for the first two subplots
    # cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    # cbar1 = fig.colorbar(axes[0].collections[0], cax=cbar_ax1)
    # cbar1.set_label("Velocity [m/s]")

    # # add separate colorbar for the last subplot
    # cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.2])
    # cbar2 = fig.colorbar(axes[2].collections[0], cax=cbar_ax2)
    # cbar2.set_label("Velocity [m/s]")

    fig.tight_layout(rect=[0, 0, 0.9, 1])
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
    # add text box with metrics
    textstr = f"MSE: {metrics_dict['mse']:.3f}, \nRMSE: {metrics_dict['rmse']:.3f}\nMAE: {metrics_dict['mae']:.3f}\nMAPE: {metrics_dict['mape']:.3f}"
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    return fig, ax


def get_wt_and_probe_idxs(graphs, probe_graphs):
    unique_wt_senders_receivers = np.unique(
        np.concat([graphs.senders, graphs.receivers])
    )
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

    ax.plot(predictions, y_coords, label="Predictions", color="green", **marker_kw1)
    ax.plot(targets, y_coords, label="Targets", color="black", **marker_kw2)
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
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

    unique_wt_senders_receivers, unique_probe_receivers = get_wt_and_probe_idxs(
        graphs, probe_graphs
    )
    overlapping_values = np.intersect1d(
        unique_wt_senders_receivers, unique_probe_receivers
    )
    # print(overlapping_values)  # Should only be the padded node
    # print(unique_wt_senders_receivers)
    # print(unique_probe_receivers)

    # The two options below are equivalent and are only here as an example
    if include_probe_edges:
        # probe_edge_coordinates = [
        #     node_positions[probe_graphs.receivers, :] + probe_graphs.edges[:, :-1],
        #     node_positions[probe_graphs.receivers, :],
        # ]  # ]
        probe_edge_coordinates = [
            node_positions[probe_graphs.senders, :],
            node_positions[probe_graphs.receivers, :],
        ]

    wf_edge_coordinates = [
        node_positions[graphs.receivers, :] + graphs.edges[:, :-1],
        node_positions[graphs.receivers, :],
    ]
    # wf_edge_coordinates = [node_positions[graphs.senders, :], node_positions[graphs.receivers, :]]

    if include_wt_edges:
        for i, (sender, receiver) in enumerate(zip(*wf_edge_coordinates)):
            ax.plot(
                [sender[0], receiver[0]],
                [sender[1], receiver[1]],
                c="g",
                alpha=1,
                ls="-",
                linewidth=0.5,
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
                linewidth=0.5,
                label="Probe edges" if i == 0 else "",
            )

    if include_wt_nodes:
        wt_positions = node_positions[unique_wt_senders_receivers, :]
        ax.scatter(
            wt_positions[:, 0],
            wt_positions[:, 1],
            c=wt_color,
            marker="2",
            s=100,
            label="WT nodes",
            zorder=10,
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
    # ax.axis("equal")
    ax.legend()
    return ax


def matplotlib_set_rcparams(publication: str = "paper"):
    def set_rcparamspaper():
        plt.style.use("classic")
        # Override the background colors via rcParams
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.figsize"] = (3.347, 2.0)
        # Set the font size
        plt.rcParams["font.size"] = 11
        # Set the font family
        plt.rcParams["font.family"] = "serif"
        # Set the font weight
        # plt.rcParams["font.weight"] = "bold"
        # Set the font size of the axes
        plt.rcParams["axes.titlesize"] = 11
        # Set the font size of the axes labels
        plt.rcParams["axes.labelsize"] = 11
        # Set the font size of the ticks
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
        # Override the background colors via rcParams
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.figsize"] = (10, 6)
        # Set the font size
        plt.rcParams["font.size"] = 16
        # Set the font family
        plt.rcParams["font.family"] = "serif"
        # Set the font weight
        # plt.rcParams["font.weight"] = "bold"
        # Set the font size of the axes
        plt.rcParams["axes.titlesize"] = 16
        # Set the font size of the axes labels
        plt.rcParams["axes.labelsize"] = 16
        # Set the font size of the ticks
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
