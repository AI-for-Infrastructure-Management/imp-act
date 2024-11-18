import glob
import os
import igraph as ig
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from imp_act import make
from imp_act.environments.road_env import RoadEnvironment

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
    "label_pos": 0.3,
    "text_dict": {
        "fontsize": 10,
        "color": "k",
        "ha": "center",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.0",    # Rounded box with tight padding
            "ec": "none",                      # Edge color
            "fc": "white",                     # Background color
            "alpha": 0.8,                       # Transparency for better line visibility
        }
    }
}

budget_bar_dict = {
    "bar_shape_dict": { # in fractions of axis bbox
        "bottom": 1.03,
        "width": 0.6,
        "height": 0.03,
        "padding": 0.01,
    },
    "bar_fill_dict": {
        "color": "grey", 
        "alpha": 0.8,
    },
    "bar_border_dict": {
        "edgecolor": "black",
    },
    "percent_text_dict":{
        "fontsize": 10,
        "ha": "right", 
        "va": "center", 
    },
    "fraction_text_dict": {
        "fontsize": 10,
        "ha": "left", 
        "va": "center", 
    }
}

save_dict = {
    "bbox_inches": "tight", 
    "pad_inches": 0.2, 
}

matplotlib_dict = {"pad_inches": 0.1}

color_coding = {
    0: "tab:green",
    1: "tab:olive",
    2: "tab:orange",
    3: "tab:pink",
    4: "tab:red",
}  # for maximum over states of the segments per edge

def pickle_graph(g: ig.Graph, filename) -> None:
    g.write_pickle(fname=filename)
    return

def unpickle_graph(filename) -> ig.Graph:
    return ig.Graph.Read_Pickle(fname=filename)

def convert_graph_to_nx(g: ig.Graph) -> nx.Graph:
    return g.to_networkx()

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

def bezier_point(
        start: np.ndarray, end: np.ndarray, control: np.ndarray, t: float
    ) -> list[np.ndarray, np.ndarray]:
    """
    Calculate a point on a quadratic Bezier curve at position t.
    start: starting point (x, y)
    end: ending point (x, y)
    control: control point (x, y)
    t: position along the curve, between 0 and 1
    """
    point = (1 - t)**2 * start + 2 * (1 - t) * t * control + t**2 * end
    tangent = 2 * (1 - t) * (control - start) + 2 * t * (end - control)
    return point, tangent

def plot_prepare(g: nx.Graph, layout: str) -> list:
    if isinstance(g, ig.Graph):
        g = convert_graph_to_nx(g=g)
    fig, ax = plt.subplots(figsize=(10, 6))
    # if no predefined layout provided -> try node positions stored in graph
    if layout not in layout_dict.keys():
        # check if every node has position keys
        if all(
            [all(
                [p in g._node[k].keys() for p in ['position_x', 'position_y']]
                ) for k in g._node.keys()]
            ):
            # create dict of node positions
            pos = {k: np.array([g._node[k][p] for p in ['position_x', 'position_y']]) for k in g._node.keys()}
        else:
            layout = "planar"
            pos = layout_dict[layout](g)
    else:
        pos = layout_dict[layout](g)
    return fig, ax, pos, g


def plot_ending(
        fig: Figure,
        ax: Axes,
        title: str, 
        show_plot: bool=False, 
        equal_axis: bool=True,
        save_plot: bool=False, 
        filename: str='plot.png'
    ) -> None:
    plt.title(title, fontsize=14) if title is not None else None
    ax.axis('equal') if equal_axis else None
    plt.savefig(filename, dpi=fig.dpi, **save_dict) if save_plot else None
    plt.show() if show_plot else None
    return


def only_graph_structure(
    g: nx.Graph,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
) -> list[dict]:
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
    g: nx.Graph, ax: Axes, use_cmap=False, my_edge_dict: dict = {}
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
        d=update_dict(d=edge_labels_dict, my_dict=my_edge_label_dict),
        my_dict={"edge_labels": edge_labels},
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

# function that draws directed as well as undirected graphs
def draw_edges(
    g: nx.Graph, 
    pos: dict, 
    ax: Axes, 
    new_edge_dict: dict, 
    curve_factor: float
) -> None:
    if g.is_directed():
        # Draw edges with offsets for bidirectional edges
        for i, (u, v) in enumerate(g.edges()):
            # construct edge-specific dict
            sub_dict = {}
            for key, val in new_edge_dict.items():
                sub_dict[key] = val[i] if isinstance(val, list) else val

            # Check if there's a reverse edge
            if (v, u) in g.edges() and u < v:
                # Draw the two edges with curve
                nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={curve_factor}', arrows=True, **sub_dict)
                nx.draw_networkx_edges(g, pos, edgelist=[(v, u)], connectionstyle=f'arc3,rad={curve_factor}', arrows=True, **sub_dict)
            elif (v, u) not in g.edges():  # Draw normally if no bidirectional pair
                nx.draw_networkx_edges(g, pos, edgelist=[(v, u)], **sub_dict)
    else:
        nx.draw_networkx_edges(G=g, pos=pos, ax=ax, **new_edge_dict)
    return 

def draw_edge_labels(
    g: nx.Graph, 
    pos: dict, 
    ax: Axes,  
    new_edge_label_dict: dict, 
    curve_factor: float,
) -> None:
    for (u, v) in g.edges():
        t = 0.5
        cf = 0.0
        if g.is_directed():
            if (v, u) in g.edges():
                t = new_edge_label_dict["label_pos"]
                cf = curve_factor
        label = new_edge_label_dict['edge_labels'][(u,v)]
        node_pos_u = np.array(pos[u])
        node_pos_v = np.array(pos[v])
        midpoint = (node_pos_u + node_pos_v) / 2
        direction = np.array([node_pos_v[1] - node_pos_u[1], node_pos_u[0] - node_pos_v[0]])
        control_point = midpoint + cf * direction
        label_pos, tangent = bezier_point(node_pos_u, node_pos_v, control_point, t=t)
        # Calculate angle for text rotation
        angle = np.degrees(np.arctan2(tangent[1], tangent[0]))
        if angle < -90 or angle > 90:
            angle += 180
        plt.text(
            label_pos[0], 
            label_pos[1], 
            label,
            rotation=angle, 
            rotation_mode="anchor", 
            **new_edge_label_dict['text_dict'], 
        )   
    return


def draw_progress_bar(
    fig: Figure,  
    ax: Axes,
    current_budget: float, 
    total_budget: float,
    new_bar_dict: dict, 
):

    ## Create a new axis for the progress bar
    # get bounding box for graph axis
    ax_bbox = ax.get_position()
    # set coordinates for the bar (centered above graph axis)
    x_start = ax_bbox.x0 + (1-new_bar_dict["bar_shape_dict"]["width"]) * ax_bbox.width/2
    y_start = new_bar_dict["bar_shape_dict"]["bottom"] * ax_bbox.y1
    width = new_bar_dict["bar_shape_dict"]["width"] * ax_bbox.width
    height = new_bar_dict["bar_shape_dict"]["height"] * ax_bbox.y1
    shape_list = [x_start, y_start, width, height]

    """
    # Center the bar horizontally in figure
    left_margin = (1 - new_bar_dict["bar_shape_dict"]["width"]) / 2
    shape_list = [left_margin] + [new_bar_dict["bar_shape_dict"][key] for key in ["bottom", "width", "height"]]
    """

    bar_ax = fig.add_axes(shape_list)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')  # Turn off the axes

    ## Draw the bar
    # filled portion
    progress = current_budget / total_budget
    bar_ax.add_patch(
        mpl.patches.Rectangle(
            xy=(0, 0), 
            width=progress, 
            height=1, 
            transform=bar_ax.transAxes, 
            **new_bar_dict["bar_fill_dict"]
        )
    )

    # Draw the border of the bar
    bar_ax.add_patch(
        patches.Rectangle(
            xy=(0, 0), 
            width=1, 
            height=1, 
            fill=False, 
            transform=bar_ax.transAxes,
            **new_bar_dict["bar_border_dict"]
        )
    )

    text_height = y_start + height / 2
    # Add the percentage label (left of the bar)
    fig.text(
        x = x_start - new_bar_dict["bar_shape_dict"]["padding"] * width, 
        y = text_height,
        s = f"{int(progress*100)}%", 
        **new_bar_dict["percent_text_dict"]
    )

    # Add the value label (right of the bar)
    if total_budget > 1000:
        pc = int(np.floor(np.log10(current_budget)))
        fc = current_budget / (10**pc)
        pt = int(np.floor(np.log10(current_budget)))
        ft = current_budget / (10**pt)
        #s = rf"${fc:.1f}\cdot 10^{pc}$ / ${ft:.1f}\cdot 10^{pt}$"
        s = f"{fc:.1f}e{pc} / {ft:.1f}e{pt}"
    else:
        s = f"{int(current_budget)}/{int(total_budget)}"
    fig.text(
        x = x_start + (1 + new_bar_dict["bar_shape_dict"]["padding"]) * width, 
        y = text_height,
        s = s,
        **new_bar_dict["fraction_text_dict"]
    )
    return


def general_plot(
    env: RoadEnvironment,
    layout="positions",
    with_color: bool = False,
    use_cmap: bool = False,
    with_edge_labels: bool = False,
    with_volumes: bool = False,
    with_progress_bar: bool = False,
    my_node_dict: dict = {},
    my_edge_dict: dict = {},
    my_node_label_dict: dict = {},
    my_edge_label_dict: dict = {},
    my_bar_dict: dict = {},
    curve_factor: float=0.05,
    title: bool = None,
    show_plot: bool = True,
    equal_axis: bool = True,
    save_plot: bool = False,
    filename: str = None,
    return_stuff: bool = False,
) -> list | None:
    ## add here new bar dict
    fig, ax, pos, g = plot_prepare(g=env.graph, layout=layout)
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
    draw_edges(g=g, pos=pos, ax=ax, new_edge_dict=new_edge_dict, curve_factor=curve_factor)
    if new_edge_label_dict["edge_labels"] is not None:
        draw_edge_labels(
            g=g, 
            pos=pos, 
            ax=ax, 
            new_edge_label_dict=new_edge_label_dict, 
            curve_factor=curve_factor
        )
    if with_progress_bar:
        # check that env has budget
        if hasattr(env, "current_budget") and hasattr(env, "budget_amount"):
            new_bar_dict = update_dict(d=budget_bar_dict, my_dict=my_bar_dict)
            # check current and total budget
            draw_progress_bar(
                fig=fig,
                ax=ax,
                current_budget=env.current_budget,
                total_budget=env.budget_amount,
                new_bar_dict=new_bar_dict
            )
        else:
            print("Passed environment does not have attributes 'budget_amount' and/or 'current_budget")
    plot_ending(
        fig=fig,
        ax=ax,
        title=title, 
        show_plot=show_plot, 
        equal_axis=equal_axis, 
        save_plot=save_plot, 
        filename=filename
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
    env_name: str = "ToyExample-v2",
    frame_folder: str = "./tmp_pic_folder", 
    frame_type: str = ".png", 
    gif_name: str | None = None,
    delete=True,
    layout="positions",
    with_color: bool = True,
    with_edge_labels: bool = True,
    with_volumes: bool = False,
    curve_factor: float=0.05,
):
    if os.path.exists(frame_folder):
        delete_folder = False
    else:
        os.mkdir(frame_folder)
        delete_folder = True
    path_list = list()
    frame_type = "." + frame_type if "." not in frame_type else frame_type

    if gif_name is None:
        gif_name = '_'.join([env_name, "traj.gif"])

    # create env
    env = make(environment_name=env_name)
    obs = env.reset()

    # find highest power of 10 in max_timesteps (for storing digits):
    digits = 1
    while 10**digits < env.max_timesteps:
        digits += 1

    # select action 1 at every timestep for every segment
    act = 1
    actions = [
        [act for _ in edge["road_edge"].segments] for edge in env.graph.es
    ]
    
    time = 0
    pic_name = os.path.join(frame_folder, f"pic{time:0{digits}d}" + frame_type)
    path_list.append(pic_name)
    general_plot(
        g=env.graph,
        layout=layout,
        with_color=with_color,
        with_edge_labels=with_edge_labels,
        with_volumes=with_volumes,
        curve_factor=curve_factor,
        title="t: 0",
        show_plot=False,
        save_plot=True,
        filename=pic_name,
    )

    while time < env.max_timesteps:
        time += 1
        obs, cost, done, info = env.step(actions)
        pic_name = os.path.join(frame_folder, f"pic{time:0{digits}d}" + frame_type)
        path_list.append(pic_name)
        general_plot(
            g=env.graph,
            layout=layout,
            with_color=with_color,
            with_edge_labels=with_edge_labels,
            with_volumes=with_volumes,
            curve_factor=curve_factor,
            title=f"t: {time}",
            show_plot=False,
            save_plot=True,
            filename=pic_name,
        )
        if done:
            break

    save_frames_as_gif(frame_folder=frame_folder, savename=gif_name)
    if delete:
        [os.remove(path) for path in path_list]
    if delete_folder:
        os.rmdir(frame_folder)
    return


def save_frames_as_gif(frame_folder: str, savename: str) -> None:
    frames = [
        Image.open(image) for image in glob.glob(os.path.join(frame_folder, "*.png"))
    ]
    frame_one = frames[0]
    frame_one.save(
        os.path.join('./env_trajectories', savename),
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
