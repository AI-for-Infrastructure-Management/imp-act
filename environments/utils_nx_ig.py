import igraph as ig
import networkx as nx


def pickle_graph(g: ig.Graph, filename) -> None:
    g.write_pickle(fname=filename)
    return


def unpickle_graph(filename) -> ig.Graph:
    return ig.Graph.Read_Pickle(fname=filename)


def convert_graph_to_nx(g: ig.Graph) -> nx.Graph:
    return g.to_networkx()


def check_for_positions(g: nx.Graph) -> bool:
    try:
        for n in g.nodes.data():
            assert ("position_x" in n[1].keys()) and ("position_y" in n[1].keys())
        return True
    except:
        return False


def get_pos_dict(g: nx.Graph) -> dict:
    pos_dict = {}
    for n in g.nodes.data():
        pos_dict[n[0]] = (n[1]["position_x"], n[1]["position_y"])
    return pos_dict
