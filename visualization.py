import igraph as ig
from igraph import Graph
from test_environment import *
from environment_presets import *
import matplotlib.pyplot as plt

from environment import RoadEnvironment

# vertex label size changes the shift not the size
standard_plot = {'vertex_size': 40, 'vertex_color': 'grey', 'vertex_shape': 'circle', 'vertex_label_size': 13,
                 'edge_width': 3, 'edge_color': 'black',
                 'autocurve': True, 'layout': 'auto', 'bbox': (0, 0, 800, 800)}

color_coding = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'} # for maximum over states of the segments per edge

def save_plot(drawing, save, filename) -> None:
    if save:
        # filetype inferred from filename ending, if no filetype specified: default to png
        if '.' not in filename:
            filename = filename + '.png'
        drawing.save(filename)
    return

def update_dict(my_dict: dict) -> dict:
    plot_dict = standard_plot.copy()
    if my_dict is not None:
        plot_dict.update(my_dict)
    return plot_dict


def graph_structure(g: Graph, my_dict: dict=None, save=False, filename=None) -> None:
    plot_dict = update_dict(my_dict=my_dict)
    drawing = ig.plot(obj=g, vertex_label=[e.index for e in g.es], **plot_dict)
    save_plot(drawing=drawing, save=save, filename=filename)
    return drawing

def graph_states(g: Graph, save:bool =False, filename:str =None) -> None:
    # green edge if no segment in states 1 or 2
    # yellow edge if at least one segment in state 1, no segment in state 2
    # orange edge if at least one segment in state 2
    # red edge if at least one segment in state 3
    edge_colors = list()
    for e in g.es:
        edge_colors.append(color_coding[max([s.state for s in e['road_segments'].segments])])
    plot_dict = standard_plot.copy()
    plot_dict.update({'edge_color': edge_colors})
    drawing = ig.plot(obj=g, vertex_label=[e.index for e in g.es], **plot_dict)
    # edge labels indicate state of segments
    return drawing


def graph_loads():
    # print graph with capacities, base travel times and assigned volumes
    return

## add export to other graph formats

    