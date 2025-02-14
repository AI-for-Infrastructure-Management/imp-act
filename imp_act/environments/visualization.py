import glob
import os

import igraph as ig
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utils_nx_ig as mu

from environments.config.environment_presets import small_environment_dict
from environments.road_env import RoadEnvironment
from PIL import Image

# plot dict for visualization with networkx
standard_dict = {"with_labels": True}

layout_dict = {
    "shell": nx.shell_layout,
    "planar": nx.planar_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
}

node_dict = {
    "node_size": 400,
    "node_color": "#1f78b4",
    "node_shape": "o",
    "alpha": 1.0,
    "edgecolors": "black",
}

edge_dict = {
    "width": 3,
    "edge_color": "black",
    "style": "solid",
    "alpha": 1,
    "edge_cmap": plt.cm.plasma,
    "edge_vmin": 0,
}

node_labels_dict = {
    "labels": None,
    "font_size": 12,
    "font_color": "k",
    "font_family": "sans-serif",
    "clip_on": True,
}

edge_labels_dict = {
    "edge_labels": None,
    "label_pos": 0.5,
    "font_size": 10,
    "font_color": "k",
    "horizontalalignment": "center",
    "verticalalignment": "center",
    "rotate": True,
    "clip_on": True,
}

matplotlib_dict = {"pad_inches": 0.1}

color_coding = {
    0: "tab:green",
    1: "tab:pink",
    2: "tab:orange",
    3: "tab:red",
}  # for maximum over states of the segments per edge


def update_dict(d: dict, my_dict: dict) -> dict:
    new_dict = d.copy()
    if my_dict != {}:
        new_dict.update(my_dict)
    return new_dict


def update_multiple_dicts(g: nx.Graph, str_list: list, dict_list: list) -> list:
    return_list = list()
    for dict_spec, my_dict in zip(str_list, dict_list):
        if dict_spec == "nodes":
            return_list.append(update_dict(d=node_dict, my_dict=my_dict))
        if dict_spec == "edges":
            return_list.append(update_dict(d=edge_dict, my_dict=my_dict))
        if dict_spec == "node_labels":
            return_list.append(
                update_dict(
                    d=update_dict(
                        d=node_labels_dict,
                        my_dict={"labels": {n: n for n in g.nodes()}},
                    ),
                    my_dict=my_dict,
                )
            )
        if dict_spec == "edge_labels":
            return_list.append(update_dict(d=edge_labels_dict, my_dict=my_dict))
    return return_list


def plot_prepare(g: nx.Graph, layout: str):
    if isinstance(g, ig.Graph):
        g = mu.convert_graph_to_nx(g=g)
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = layout_dict[layout](g)
    return fig, ax, pos, g


def plot_ending(title: str, show_plot: bool, save_plot: bool, filename: str) -> None:
    plt.title(title, fontsize=14) if title is not None else None
    plt.savefig(filename, pad_inches=0) if save_plot else None
    plt.show() if show_plot else None
    return


def only_graph_structure(
    g: nx.Graph,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
) -> [dict, dict, dict, dict]:
    # inject custom changes passed to the function
    str_list = ["nodes", "edges", "node_labels", "edge_labels"]
    dict_list = [my_node_dict, my_edge_dict, my_node_label_dict, my_edge_label_dict]
    (
        new_node_dict,
        new_edge_dict,
        new_node_label_dict,
        new_edge_label_dict,
    ) = update_multiple_dicts(g=g, str_list=str_list, dict_list=dict_list)
    return [new_node_dict, new_edge_dict, new_node_label_dict, new_edge_label_dict]


def only_edge_colors(
    g: nx.Graph, ax: plt.axes, use_cmap=False, my_edge_dict: dict = {}
) -> dict:
    num_states = (
        next(iter(next(iter(g.adjacency()))[1].values()))["road_edge"]
        .segments[0]
        .number_of_states
    )

    edge_colors = list()
    if use_cmap:
        for e in g.edges():
            edge_colors.append(max([s.state for s in g.edges[e]["road_edge"].segments]))
        # new_edge_dict = update_dict(d=new_edge_dict, my_dict={'edge_color': edge_colors, 'edge_vmax': num_states})
        new_edge_dict = update_dict(
            d=my_edge_dict, my_dict={"edge_color": edge_colors, "edge_vmax": num_states}
        )
        cmap = edge_dict["edge_cmap"]
    else:
        for e in g.edges():
            edge_colors.append(
                color_coding[max([s.state for s in g.edges[e]["road_edge"].segments])]
            )
        # new_edge_dict = update_dict(d=new_edge_dict, my_dict={'edge_color': edge_colors, 'edge_cmap':None, 'edge_vmin':0, 'edge_vmax': None})
        new_edge_dict = update_dict(
            d=my_edge_dict,
            my_dict={
                "edge_color": edge_colors,
                "edge_cmap": None,
                "edge_vmin": 0,
                "edge_vmax": None,
            },
        )
        cmap = plt.get_cmap("viridis", num_states)
        colors = cmap.colors
        for k, v in zip(color_coding.keys(), color_coding.values()):
            colors[k, :] = list(mpl.colors.to_rgba(v))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", colors, cmap.N
        )
    norm = mpl.colors.Normalize(vmin=0, vmax=num_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        ticks=np.linspace(0, num_states - 1, num_states),
        boundaries=np.arange(-0.5, num_states, 1),
    )
    return new_edge_dict


def only_edge_labels(g: nx.Graph, my_edge_label_dict: dict = {}) -> dict:
    num_states = (
        next(iter(next(iter(g.adjacency()))[1].values()))["road_edge"]
        .segments[0]
        .number_of_states
    )

    edge_states = list()
    edge_labels = {}
    for e in g.edges():
        edge_states.append([s.state for s in g.edges[e]["road_edge"].segments])
        el = [0] * num_states
        for s in edge_states[-1]:
            el[s] += 1
        label_string = ""
        for k in range(num_states):
            label_string = (
                label_string + rf"$S_{k}$:{el[k]} " if el[k] > 0 else label_string
            )
        edge_labels[e] = label_string[:-1]
    new_edge_label_dict = update_dict(
        d=update_dict(d=edge_labels_dict, my_dict={"edge_labels": edge_labels}),
        my_dict=my_edge_label_dict,
    )
    return new_edge_label_dict


def only_volumes(g: nx.Graph, my_edge_dict: dict = {}) -> dict:
    # go through edges and collect the assigned volumes
    width_list = list()
    width_dict = {}
    for e in g.edges():
        width_dict[e] = g.edges[e]["volume"]
        width_list.append(g.edges[e]["volume"] / 40)
    new_edge_dict = update_dict(d=my_edge_dict, my_dict={"width": width_list})
    return new_edge_dict


def general_plot(
    g: nx.Graph,
    layout="planar",
    with_color: bool = False,
    use_cmap: bool = False,
    with_edge_labels: bool = False,
    with_volumes: bool = False,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
    title: bool = None,
    show_plot: bool = True,
    save_plot: bool = False,
    filename: str = None,
    return_stuff: bool = False,
) -> None:

    fig, ax, pos, g = plot_prepare(g=g, layout=layout)
    (
        new_node_dict,
        new_edge_dict,
        new_node_label_dict,
        new_edge_label_dict,
    ) = only_graph_structure(
        g=g,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
    )

    # overwrite the respective dicts based on the desired inputs
    if with_color:
        new_edge_dict = only_edge_colors(
            g=g, ax=ax, use_cmap=use_cmap, my_edge_dict=new_edge_dict
        )
    if with_volumes:
        new_edge_dict = only_volumes(g=g, my_edge_dict=new_edge_dict)
    if with_edge_labels:
        new_edge_label_dict = only_edge_labels(
            g=g, my_edge_label_dict=my_edge_label_dict
        )

    nx.draw_networkx_nodes(G=g, pos=pos, **new_node_dict)
    nx.draw_networkx_labels(G=g, pos=pos, ax=ax, **new_node_label_dict)
    nx.draw_networkx_edges(G=g, pos=pos, ax=ax, **new_edge_dict)
    if new_edge_label_dict["edge_labels"] is not None:
        nx.draw_networkx_edge_labels(G=g, pos=pos, ax=ax, **new_edge_label_dict)

    plot_ending(
        title=title, show_plot=show_plot, save_plot=save_plot, filename=filename
    )
    return (
        [
            fig,
            ax,
            pos,
            g,
            new_node_dict,
            new_edge_dict,
            new_node_label_dict,
            new_edge_label_dict,
        ]
        if return_stuff
        else None
    )


def vis_one_episode(
    frame_folder: str = "./tmp_pic_folder", frame_type: str = ".png", delete=True
):
    if os.path.exists(frame_folder):
        delete_folder = False
    else:
        os.mkdir(frame_folder)
        delete_folder = True

    path_list = list()

    frame_type = "." + frame_type if "." not in frame_type else frame_type
    # find highest power of 10 in max_timesteps (for storing digits):
    digits = 1
    while 10**digits < small_environment_dict["max_timesteps"]:
        digits += 1

    # create env
    env = RoadEnvironment(**small_environment_dict)

    obs = env.reset()
    actions = [[1, 1] for _ in range(4)]
    time = 0
    pic_name = os.path.join(frame_folder, f"pic{time:0{digits}d}" + frame_type)
    path_list.append(pic_name)
    general_plot(
        g=env.graph,
        with_color=True,
        with_edge_labels=True,
        with_volumes=True,
        title="t: 0",
        show_plot=False,
        save_plot=True,
        filename=pic_name,
    )

    while time < small_environment_dict["max_timesteps"]:
        time += 1
        obs, cost, done, info = env.step(actions)
        pic_name = os.path.join(frame_folder, f"pic{time:0{digits}d}" + frame_type)
        path_list.append(pic_name)
        general_plot(
            g=env.graph,
            with_color=True,
            with_edge_labels=True,
            with_volumes=True,
            title=f"t: {time}",
            show_plot=False,
            save_plot=True,
            filename=pic_name,
        )
        if done:
            break

    save_frames_as_gif(frame_folder=frame_folder)
    if delete:
        [os.remove(path) for path in path_list]
    if delete_folder:
        os.rmdir(frame_folder)
    return


def save_frames_as_gif(frame_folder: str) -> None:
    frames = [
        Image.open(image) for image in glob.glob(os.path.join(frame_folder, "*.png"))
    ]
    frame_one = frames[0]
    frame_one.save(
        "one_traj.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=1000,
        loop=0,
    )
    return


# special plotting function which make use of general_plot. Not really necessary, but maybe someone wants to use these instead


def plot_only_graph_structure(
    g: nx.Graph,
    layout="planar",
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    title=None,
    show_plot=True,
    save_plot=False,
    filename=None,
    return_stuff=False,
) -> None:

    fig, ax, pos, g, nnd, ned, nnld, _ = general_plot(
        g=g,
        layout=layout,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
        title=title,
        show_plot=show_plot,
        save_plot=save_plot,
        filename=filename,
        return_stuff=True,
    )
    return [fig, ax, pos, g, nnd, ned, nnld] if return_stuff else None


def plot_only_states(
    g: nx.Graph,
    layout="planar",
    use_cmap=False,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    title=None,
    show_plot=True,
    save_plot=False,
    filename=None,
    return_stuff=False,
) -> None:

    fig, ax, pos, g, nnd, ned, nnld, _ = general_plot(
        g=g,
        layout=layout,
        with_color=True,
        use_cmap=use_cmap,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
        title=title,
        show_plot=show_plot,
        save_plot=save_plot,
        filename=filename,
        return_stuff=True,
    )
    return [fig, ax, pos, g, nnd, ned, nnld] if return_stuff else None


def plot_states_with_edge_labels(
    g: nx.Graph,
    layout="planar",
    use_cmap=False,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
    with_color: bool = True,
    title=None,
    show_plot=True,
    save_plot=False,
    filename=None,
    return_stuff=False,
) -> None:
    fig, ax, pos, g, nnd, ned, nnld, neld = general_plot(
        g=g,
        layout=layout,
        with_color=True,
        use_cmap=use_cmap,
        with_edge_labels=True,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
        my_edge_label_dict=my_edge_label_dict,
        title=title,
        show_plot=show_plot,
        save_plot=save_plot,
        filename=filename,
        return_stuff=True,
    )
    return [fig, ax, pos, g, nnd, ned, nnld, neld] if return_stuff else None


def plot_only_volumes(
    g: nx.Graph,
    layout="planar",
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
    title=None,
    show_plot=True,
    save_plot=False,
    filename=None,
    return_stuff=False,
) -> None:

    fig, ax, pos, g, nnd, ned, nnld, _ = general_plot(
        g=g,
        layout=layout,
        with_volumes=True,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
        my_edge_label_dict=my_edge_label_dict,
        title=title,
        show_plot=show_plot,
        save_plot=save_plot,
        filename=filename,
        return_stuff=True,
    )
    return [fig, ax, pos, g, nnd, ned, nnld, _] if return_stuff else None


def plot_states_labels_and_volumes(
    g: nx.Graph,
    layout="planar",
    with_color: bool = True,
    use_cmap: bool = False,
    with_edge_labels: bool = True,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
    title=None,
    show_plot=True,
    save_plot=False,
    filename=None,
    return_stuff=False,
) -> None:

    fig, ax, pos, g, nnd, ned, nnld, neld = general_plot(
        g=g,
        layout=layout,
        with_color=True,
        use_cmap=use_cmap,
        with_edge_labels=True,
        with_volumes=True,
        my_node_dict=my_node_dict,
        my_edge_dict=my_edge_dict,
        my_node_label_dict=my_node_label_dict,
        my_edge_label_dict=my_edge_label_dict,
        title=title,
        show_plot=show_plot,
        save_plot=save_plot,
        filename=filename,
        return_stuff=True,
    )
    return [fig, ax, pos, g, nnd, ned, nnld, neld] if return_stuff else None


# print graph with capacities, base travel times and assigned volumes
