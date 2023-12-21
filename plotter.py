import numpy as np
import igraph as ig
from test_environment import *
from environment_presets import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import myutils as mu

from environment import RoadEnvironment

class Plotter():

    def __init__(self, graph: nx.Graph) -> None:

        self.graph = graph

        self.standard_dict = {'with_labels': True}

        self.layout_dict = {'shell': nx.shell_layout, 
                            'planar': nx.planar_layout, 
                            'kamada_kawai': nx.kamada_kawai_layout}

        self.node_dict = {'node_size': 400, 'node_color': '#1f78b4', 
                          'node_shape': 'o', 'alpha': 1.0, 
                          'edgecolors': 'black'}

        self.edge_dict = {'width': 3, 'edge_color': 'black', 
                          'style':'solid', 'alpha': 1, 
                          'edge_cmap': plt.cm.plasma, 'edge_vmin': 0}

        self.node_labels_dict = {'labels': None, 'font_size': 12, 
                                 'font_color': 'k', 
                                 'font_family': 'sans-serif', 
                                 'clip_on': True}

        self.edge_labels_dict = {'edge_labels': None, 'label_pos': 0.5, 
                                 'font_size': 8, 'font_color': 'k', 
                                 'horizontalalignment': 'center', 
                                 'verticalalignment': 'center', 
                                 'rotate': True, 'clip_on': True}

        self.matplotlib_dict = {'pad_inches': 0.1}

        self.color_coding = {0: 'tab:green', 1: 'tab:pink', 2: 
                             'tab:orange', 3: 'tab:red'}

    def update_dict(self):
        pass

    def update_multiple_dicts(self):
        pass

    def plot_prepare(self):
        pass

    def graph_structure(self):
        pass

    def graph_states(self):
        pass

    def graph_states_with_edge_labels(self):
        pass


         

