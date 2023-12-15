import igraph as ig
from igraph import Graph
from test_environment import *
from environment_presets import *
import matplotlib.pyplot as plt

from environment import RoadEnvironment

def only_structure(g: Graph, save=False, filename=None) -> None:
    E =  RoadEnvironment(**small_environment_dict)
    ig.plot(obj=E.graph, bbox=(0, 0, 400, 400))
    return

def graph_with_cap(g: Graph, save:bool =False, filename:str =None) -> None:

## add export to other graph formats

    