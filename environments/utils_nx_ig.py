import igraph as ig
import networkx as nx


def pickle_graph(g: ig.Graph, filename) -> None:
    g.write_pickle(fname=filename)
    return


def unpickle_graph(filename) -> ig.Graph:
    return ig.Graph.Read_Pickle(fname=filename)


def convert_graph_to_nx(g: ig.Graph) -> nx.Graph:
    return g.to_networkx()
