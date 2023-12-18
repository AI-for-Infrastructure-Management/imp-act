import numpy as np
import igraph as ig
from test_environment import *
from environment_presets import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import myutils as mu

from environment import RoadEnvironment

# plot dict for visualization with networkx
standard_dict = {'with_labels': True}

layout_dict = {'shell': nx.shell_layout, 'planar': nx.planar_layout, 'kamada_kawai': nx.kamada_kawai_layout}

node_dict = {'node_size': 400, 'node_color': '#1f78b4', 'node_shape': 'o', 'alpha': 1.0, 'edgecolors': 'black'}

edge_dict = {'width': 3, 'edge_color': 'black', 'style':'solid', 'alpha': 1, 'edge_cmap': plt.cm.plasma, 'edge_vmin': 0}

node_labels_dict = {'labels': None, 'font_size': 12, 'font_color': 'k', 'font_family': 'sans-serif', 'clip_on': True}

edge_labels_dict = {'edge_labels': None, 'label_pos': 0.5, 'font_size': 8, 'font_color': 'k', 'horizontalalignment': 'center', 
                    'verticalalignment': 'center', 'rotate': True, 'clip_on': True}

matplotlib_dict = {'pad_inches': 0.1}

color_coding = {0: 'tab:green', 1: 'tab:pink', 2: 'tab:orange', 3: 'tab:red'} # for maximum over states of the segments per edge


def update_dict(d: dict, my_dict: dict) -> dict:
    new_dict = d.copy()
    if my_dict != {}:
        new_dict.update(my_dict)
    return new_dict

def update_multiple_dicts(g: nx.Graph, str_list: list, dict_list: list) -> list:
    return_list = list()
    for dict_spec, my_dict in zip(str_list, dict_list):
        if dict_spec == 'nodes':
            return_list.append(update_dict(d=node_dict, my_dict=my_dict))
        if dict_spec == 'edges':
            return_list.append(update_dict(d=edge_dict, my_dict=my_dict))
        if dict_spec == 'node_labels':
            return_list.append(update_dict(d=update_dict(d=node_labels_dict, my_dict={'labels': {n:n for n in g.nodes()}}), my_dict=my_dict))
        if dict_spec == 'edge_labels':
            return_list.append(update_dict(d=edge_labels_dict, my_dict=my_dict))
    return return_list

def plot_prepare(g: nx.Graph, layout):
    if isinstance(g, ig.Graph):
        g = mu.convert_graph_to_nx(g=g)
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = layout_dict[layout](g)
    return fig, ax, pos, g

def graph_structure(g: nx.Graph, layout='planar', my_node_dict: dict={}, my_edge_dict: dict={}, my_node_label_dict: dict={}, 
                    show_plot=True, save_plot=False, filename=None, return_stuff=False) -> None:
    
    fig, ax, pos, g = plot_prepare(g=g, layout=layout)
    # inject custom changes passed to the function
    str_list = ['nodes', 'edges', 'node_labels']
    dict_list = [my_node_dict, my_edge_dict, my_node_label_dict]
    new_node_dict, new_edge_dict, new_node_label_dict = update_multiple_dicts(g=g, str_list=str_list, dict_list=dict_list)
    nx.draw_networkx_nodes(G=g, pos=pos, **new_node_dict)
    nx.draw_networkx_edges(G=g, pos=pos, ax=ax, **new_edge_dict)
    nx.draw_networkx_labels(G=g, pos=pos, ax=ax, **new_node_label_dict)
    plt.savefig(filename, pad_inches=0) if save_plot else None
    plt.show() if show_plot else None
    return [fig, ax, pos, g] if return_stuff else None

def graph_states(g: nx.Graph, layout='planar', use_cmap=False, my_node_dict: dict={}, my_edge_dict: dict={}, my_node_label_dict: dict={}, 
                 my_edge_label_dict: dict={}, show_plot=True, save_plot=False, filename=None, return_stuff=False) -> None:
    #fig, ax, pos = plot_prepare(g=g, layout=layout)

    #str_list = ['nodes', 'edges', 'node_labels']
    #dict_list = [my_node_dict, my_edge_dict, my_node_label_dict]
    #new_node_dict, new_edge_dict, new_node_label_dict = update_multiple_dicts(g=g, str_list=str_list, dict_list=dict_list)

    fig, ax, pos, g = graph_structure(g=g, layout=layout, my_node_dict=my_node_dict, my_edge_dict=my_edge_dict, 
                                      my_node_label_dict=my_node_label_dict, show_plot=False, save_plot=False, filename=None, return_stuff=True)

    num_states = next(iter(next(iter(g.adjacency()))[1].values()))['road_segments'].segments[0].number_of_states

    edge_colors = list()
    if use_cmap:
        for e in g.edges():
            edge_colors.append(max([s.state for s in g.edges[e]['road_segments'].segments]))
        #new_edge_dict = update_dict(d=new_edge_dict, my_dict={'edge_color': edge_colors, 'edge_vmax': num_states})
        new_edge_dict = update_dict(d=edge_dict, my_dict={'edge_color': edge_colors, 'edge_vmax': num_states})
        cmap = edge_dict['edge_cmap']
    else:
        for e in g.edges():
            edge_colors.append(color_coding[max([s.state for s in g.edges[e]['road_segments'].segments])])
        #new_edge_dict = update_dict(d=new_edge_dict, my_dict={'edge_color': edge_colors, 'edge_cmap':None, 'edge_vmin':0, 'edge_vmax': None})
        new_edge_dict = update_dict(d=edge_dict, my_dict={'edge_color': edge_colors, 'edge_cmap':None, 'edge_vmin':0, 'edge_vmax': None})
        cmap = plt.get_cmap('viridis', num_states)
        colors = cmap.colors
        for k, v in zip(color_coding.keys(), color_coding.values()):
            colors[k,:] = list(mpl.colors.to_rgba(v))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=num_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, ticks=np.linspace(0, num_states-1, num_states), boundaries=np.arange(-0.5, num_states, 1))

    #nx.draw_networkx_nodes(G=g, pos=pos, **new_node_dict)
    nx.draw_networkx_edges(G=g, pos=pos, ax=ax, **new_edge_dict)
    #nx.draw_networkx_labels(G=g, pos=pos, ax=ax, **new_node_label_dict)

    plt.savefig(filename, pad_inches=0) if save_plot else None
    plt.show() if show_plot else None
    return [fig, ax, pos, g] if return_stuff else None


def graph_states_with_edge_labels(g: nx.Graph, layout='planar', use_cmap=False, my_node_dict: dict={}, 
                                  my_edge_dict: dict={}, my_node_label_dict: dict={}, my_edge_label_dict: dict={}, 
                                  show_plot=True, save_plot=False, filename=None, return_stuff=False) -> None:
    # construct graph without edge labels
    fig, ax, pos, g = graph_states(g=g, layout=layout, use_cmap=use_cmap, my_node_dict=my_node_dict, my_edge_dict=my_edge_dict,
                                   my_node_label_dict=my_node_label_dict, my_edge_label_dict=my_edge_label_dict, show_plot=False, 
                                   save_plot=False, filename=None, return_stuff=True)
    
    num_states = next(iter(next(iter(g.adjacency()))[1].values()))['road_segments'].segments[0].number_of_states

    edge_states = list()
    edge_labels = {}
    for e in g.edges():
        edge_states.append([s.state for s in g.edges[e]['road_segments'].segments])
        el = [0]*num_states
        for s in edge_states[-1]:
            el[s] += 1
        label_string = ''
        for k in range(num_states):
            label_string = label_string + f'S{k}:{el[k]} ' if el[k] > 0 else label_string
        edge_labels[e] = label_string[:-1]

    new_edge_label_dict = update_dict(d=update_dict(d=edge_labels_dict, my_dict={'edge_labels': edge_labels}), my_dict=my_edge_label_dict)

    nx.draw_networkx_edge_labels(G=g, pos=pos, ax=ax, **new_edge_label_dict)

    plt.savefig(filename, pad_inches=0) if save_plot else None
    plt.show() if show_plot else None
    return [fig, ax, pos, g] if return_stuff else None



# print graph with capacities, base travel times and assigned volumes
    