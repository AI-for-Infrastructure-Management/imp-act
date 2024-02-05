import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


paper_url = "https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content"
data_url = "https://data.mendeley.com/datasets/py2zkrb65h/1"
truck_traffic_file = "01_Trucktrafficflow.csv"
nuts_regions_file = "02_NUTS-3-Regions.csv"
nodes_file_name = "03_network-nodes.csv"
edges_file_name = "04_network-edges.csv"


def plot_network(G, pos, title, path):
    # Plotting the network
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig(Path(path, f"{title}.png"))


def remove_nodes_with_degree_one_below_threshold(graph, threshold):
    nodes_to_remove = []
    for node in graph.nodes():
        if graph.degree(node) == 1:
            neighbor = list(graph.neighbors(node))[0]
            if graph.edges[node, neighbor]["Distance"] < threshold:
                nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def remove_nodes_and_merge_edges(graph, cleanup=False):
    for edge in graph.edges():
        if "original" not in graph.edges[edge]:
            graph.edges[edge]["original"] = True
            graph.edges[edge]["node_ids"] = []
            graph.edges[edge]["edge_ids"] = []

    # check that all edges have the original attribute
    assert all(["original" in graph.edges[edge] for edge in graph.edges()])

    while True:
        nodes_with_two_edges = [
            node for node, degree in dict(graph.degree()).items() if degree == 2
        ]
        if not nodes_with_two_edges:
            break

        for node in nodes_with_two_edges:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 2:
                edge_1 = graph.edges[node, neighbors[0]]
                edge_2 = graph.edges[node, neighbors[1]]

                nodes = [node]
                if not edge_1["original"]:
                    nodes += edge_1["node_ids"]
                if not edge_2["original"]:
                    nodes += edge_2["node_ids"]

                edges = []
                if not edge_1["original"]:
                    edges += edge_1["edge_ids"]
                else:
                    edges += [int(edge_1["Network_Edge_ID"])]

                if not edge_2["original"]:
                    edges += edge_2["edge_ids"]
                else:
                    edges += [int(edge_2["Network_Edge_ID"])]

                graph.remove_node(node)

                if not graph.has_edge(neighbors[0], neighbors[1]):
                    graph.add_edge(neighbors[0], neighbors[1])
                    graph.edges[neighbors[0], neighbors[1]]["Distance"] = float(
                        edge_1["Distance"] + edge_2["Distance"]
                    )
                    graph.edges[neighbors[0], neighbors[1]]["node_ids"] = nodes
                    graph.edges[neighbors[0], neighbors[1]]["edge_ids"] = edges
                    graph.edges[neighbors[0], neighbors[1]]["original"] = False
                else:
                    # edge already exists, leave the closer one
                    edge_3 = graph.edges[neighbors[0], neighbors[1]]
                    if edge_1["Distance"] + edge_2["Distance"] < edge_3["Distance"]:
                        graph.edges[neighbors[0], neighbors[1]]["Distance"] = (
                            edge_1["Distance"] + edge_2["Distance"]
                        )
                        graph.edges[neighbors[0], neighbors[1]]["node_ids"] = nodes
                        graph.edges[neighbors[0], neighbors[1]]["edge_ids"] = edges
                        graph.edges[neighbors[0], neighbors[1]]["original"] = False

    new_edge_info = []
    id = 5_000_000  # Highest id in dataset 2616216
    for edge in graph.edges():
        if not graph.edges[edge]["original"]:
            graph.edges[edge]["Network_Edge_ID"] = id
            edge_info = graph.edges[edge].copy()
            edge_info["Network_Node_A_ID"] = edge[0]
            edge_info["Network_Node_B_ID"] = edge[1]
            new_edge_info.append(edge_info)
            id += 1
        if cleanup:
            del graph.edges[edge]["node_ids"]
            del graph.edges[edge]["edge_ids"]

    return graph, new_edge_info


def parse_string_list_of_integer(string_list):
    if type(string_list) == str:
        if string_list == "[]":
            return []
        else:
            return [int(x) for x in string_list[1:-1].split(",")]
    else:
        pass


def export_country(args):
    # load data
    nodes_df = pd.read_csv(os.path.join(args.data_dir, nodes_file_name))
    edges_df = pd.read_csv(os.path.join(args.data_dir, edges_file_name))

    # Create folder for country
    country_output_path = Path(args.output_dir, f"countries/{args.country}")
    country_output_path.mkdir(parents=True, exist_ok=True)

    # Filtering for a Specific Country
    filtered_nodes = nodes_df[nodes_df["Country"] == args.country]
    filtered_edges = edges_df[
        edges_df["Network_Node_A_ID"].isin(filtered_nodes["Network_Node_ID"])
        & edges_df["Network_Node_B_ID"].isin(filtered_nodes["Network_Node_ID"])
    ]

    # Create a graph from the filtered edges
    G_filtered = nx.from_pandas_edgelist(
        filtered_edges, "Network_Node_A_ID", "Network_Node_B_ID"
    )

    # add edge information as attributes to graph
    edge_columns = [
        "Network_Edge_ID",
        "Manually_Added",
        "Distance",
        "Traffic_flow_trucks_2019",
        "Traffic_flow_trucks_2030",
    ]
    for index, row in filtered_edges.iterrows():
        for column in edge_columns:
            G_filtered.edges[row["Network_Node_A_ID"], row["Network_Node_B_ID"]][
                column
            ] = row[column]

    # Create a position dictionary from node coordinates
    pos_filtered = {
        row["Network_Node_ID"]: (row["Network_Node_X"], row["Network_Node_Y"])
        for index, row in filtered_nodes.iterrows()
    }

    # Plotting the network
    plot_network(
        G_filtered,
        pos_filtered,
        f"Network for {args.country} (N: {len(G_filtered.nodes)}, E: {len(G_filtered.edges)})",
        country_output_path,
    )

    # Merge nodes with only two edges
    G_reduced, new_edge_info = remove_nodes_and_merge_edges(G_filtered.copy())

    # Remove edges which are below threshold
    G_reduced_2 = remove_nodes_with_degree_one_below_threshold(
        G_reduced.copy(), args.pruning_threshold
    )

    # Merge nodes with only two edges again after removing edges below threshold
    G_reduced_3, new_edge_info = remove_nodes_and_merge_edges(
        G_reduced_2.copy(), cleanup=True
    )

    # Add position data for nodes to graph
    position_dict = {
        id: {"position_x": pos[0], "position_y": pos[1]}
        for id, pos in pos_filtered.items()
    }
    nx.set_node_attributes(G_reduced_3, position_dict)

    # Plotting the network
    plot_network(
        G_reduced_3,
        pos_filtered,
        f"Reduced Network for {args.country} (N: {len(G_reduced_3.nodes)}, E: {len(G_reduced_3.edges)})",
        country_output_path,
    )

    # rename attribute "Distance" to "distance" and make sure it is a float
    # drop attribute "original"
    for edge in G_reduced_3.edges():
        G_reduced_3.edges[edge]["distance"] = float(G_reduced_3.edges[edge]["Distance"])
        del G_reduced_3.edges[edge]["Distance"]
        del G_reduced_3.edges[edge]["original"]

    # rename attribute "Network_Edge_ID" to "id" and make sure it is an int
    # drop attribute "Manually_Added", "Traffic_flow_trucks_2019", "Traffic_flow_trucks_2030" from edge
    for edge in G_reduced_3.edges():
        G_reduced_3.edges[edge]["id"] = int(G_reduced_3.edges[edge]["Network_Edge_ID"])
        del G_reduced_3.edges[edge]["Network_Edge_ID"]
        if "Manually_Added" in G_reduced_3.edges[edge].keys():
            del G_reduced_3.edges[edge]["Manually_Added"]
        if "Traffic_flow_trucks_2019" in G_reduced_3.edges[edge].keys():
            del G_reduced_3.edges[edge]["Traffic_flow_trucks_2019"]
        if "Traffic_flow_trucks_2030" in G_reduced_3.edges[edge].keys():
            del G_reduced_3.edges[edge]["Traffic_flow_trucks_2030"]

    # Export graph to graphml
    nx.write_graphml_lxml(
        G_reduced_3, f"{country_output_path.absolute()}/graph.graphml"
    )

    # store new edge information as yaml
    with open(f"{country_output_path.absolute()}/new-edges.yaml", "w") as file:
        yaml.dump(new_edge_info, file)

    print(f"Exported graph for {args.country} to {country_output_path.absolute()}")
    print(f"Number of nodes: {len(G_reduced_3.nodes)}")
    print(f"Number of edges: {len(G_reduced_3.edges)}")

    travel_time = args.segment_length / args.segment_speed

    total_number_of_segments = 0
    segments = []
    for edge in G_reduced_3.edges():
        node_a = G_reduced_3.nodes()[edge[0]]
        node_b = G_reduced_3.nodes()[edge[1]]
        no_segments = int(
            np.ceil(G_reduced_3.edges[edge]["distance"] / args.segment_length)
        )

        # linear interpolation of coordinates
        x_coordinates = np.linspace(
            node_a["position_x"], node_b["position_x"], no_segments
        )
        y_coordinates = np.linspace(
            node_a["position_y"], node_b["position_y"], no_segments
        )

        for x, y in zip(x_coordinates, y_coordinates):
            segments.append(
                (edge[0], edge[1], x, y, args.segment_capacity, travel_time)
            )

        total_number_of_segments += no_segments
        # print(f'Edge {edge} has {no_segments} segments')

    segments_df = pd.DataFrame(
        segments,
        columns=[
            "source",
            "target",
            "position_x",
            "position_y",
            "capacity",
            "travel_time",
        ],
    )
    # save
    segments_df.to_csv(f"{country_output_path.absolute()}/segments.csv", index=False)

    print(f"Total number of segments: {total_number_of_segments}")

    # load NUTS-3 regions
    nuts_regions_df = pd.read_csv(os.path.join(args.data_dir, nuts_regions_file))

    # Filter so only regions which are in the graph or are part of the reduced edges are left
    nodes_in_graph = [
        row["Network_Node_ID"]
        for index, row in (
            nodes_df[nodes_df["Network_Node_ID"].isin(G_reduced_3.nodes())]
        ).iterrows()
    ]
    # add nodes which have been removed during reduction
    for edge in new_edge_info:
        nodes_in_graph += edge["node_ids"]

    # assert(len(nodes_in_graph) == len(set(nodes_in_graph)))
    # assert(len(nodes_in_graph) == len(G_filtered.nodes()))

    edges_in_graph = [
        row["Network_Edge_ID"]
        for index, row in (
            edges_df[edges_df["Network_Edge_ID"].isin(G_reduced_3.edges())]
        ).iterrows()
    ]

    for edge in new_edge_info:
        edges_in_graph += edge["edge_ids"]

    regions_in_graph = nuts_regions_df[
        nuts_regions_df["Network_Node_ID"].isin(nodes_in_graph)
    ]

    # generate lookup table to convert from Zone ID to Network Node ID
    zone_id_to_node_id_lookup = {
        row["ETISPlus_Zone_ID"]: row["Network_Node_ID"]
        for index, row in regions_in_graph.iterrows()
    }

    # load 01_Trucktrafficflow.csv
    truck_traffic_df = pd.read_csv(os.path.join(args.data_dir, truck_traffic_file))

    # filter trips to include only trips which have edges in the graph
    truck_traffic_df["remove"] = True
    truck_traffic_df["completly_in_graph"] = False

    region_ids_in_graph = regions_in_graph["ETISPlus_Zone_ID"].to_list()
    for index, row in tqdm(truck_traffic_df.iterrows(), total=len(truck_traffic_df)):
        path_edges = parse_string_list_of_integer(row["Edge_path_E_road"])
        found = False
        for edge in path_edges:
            if edge in edges_in_graph:
                truck_traffic_df.loc[index, "remove"] = False
                found = True
                break

        if not found:
            continue

        # check if both regions are in the graph
        if row["ID_origin_region"] in region_ids_in_graph:
            truck_traffic_df.loc[index, "origin_node"] = int(
                zone_id_to_node_id_lookup[row["ID_origin_region"]]
            )
        else:
            # find first edge in the region
            found = False
            for i, edge in enumerate(path_edges):
                if edge in edges_in_graph:
                    edge_in = edge
                    edge_out = path_edges[i - 1]
                    found = True
                    break
            if not found:
                raise Exception("No edge in graph found")

            # find the node between the two edges
            edges_df_filtered = edges_df[
                edges_df["Network_Edge_ID"].isin([edge_in, edge_out])
            ]
            nodes = [
                node
                for edge_node in ["Network_Node_A_ID", "Network_Node_B_ID"]
                for node in edges_df_filtered[edge_node]
            ]

            # find the node which is in the list twice
            for node in nodes:
                if nodes.count(node) == 2:
                    node_between = node
                    break

            truck_traffic_df.loc[index, "origin_node"] = int(node_between)

        if row["ID_destination_region"] in region_ids_in_graph:
            truck_traffic_df.loc[index, "destination_node"] = int(
                zone_id_to_node_id_lookup[row["ID_destination_region"]]
            )
        else:
            found = False
            for i, edge in enumerate(path_edges[::-1]):
                if edge in edges_in_graph:
                    edge_in = edge
                    edge_out = path_edges[::-1][i - 1]
                    found = True
                    break
            if not found:
                raise Exception("No edge in graph found")
            # find the node between the two edges
            edges_df_filtered = edges_df[
                edges_df["Network_Edge_ID"].isin([edge_in, edge_out])
            ]
            nodes = [
                node
                for edge_node in ["Network_Node_A_ID", "Network_Node_B_ID"]
                for node in edges_df_filtered[edge_node]
            ]

            # find the node which is in the list twice
            for node in nodes:
                if nodes.count(node) == 2:
                    node_between = node
                    break
            truck_traffic_df.loc[index, "destination_node"] = int(node_between)

    truck_traffic_df_filtered = truck_traffic_df[~truck_traffic_df["remove"]].copy()

    # Create reduced graph node lookup table
    new_edge_info_lookup = {}
    new_node_info_lookup = {}

    for edge in new_edge_info:
        for node in edge["node_ids"]:
            new_edge_info_lookup[node] = edge["Network_Edge_ID"]
            new_node_info_lookup[node] = [
                edge[ab] for ab in ["Network_Node_A_ID", "Network_Node_B_ID"]
            ]

    nodes_in_reduced_graph = [node for node in G_reduced_3.nodes()]

    # lookup new nodes in reduced graph
    for index, row in truck_traffic_df_filtered.iterrows():
        origin_node = int(row["origin_node"])
        destination_node = int(row["destination_node"])

        if origin_node not in nodes_in_reduced_graph:
            origin_node = new_node_info_lookup[origin_node][0]
        if destination_node not in nodes_in_reduced_graph:
            destination_node = new_node_info_lookup[destination_node][1]
        truck_traffic_df_filtered.loc[index, "origin_node_reduced"] = origin_node
        truck_traffic_df_filtered.loc[
            index, "destination_node_reduced"
        ] = destination_node

    # clean data before exporting
    # change datatype of [origin_node	destination_node	origin_node_reduced	destination_node_reduced] to int
    change = [
        "origin_node",
        "destination_node",
        "origin_node_reduced",
        "destination_node_reduced",
    ]
    truck_traffic_df_filtered = truck_traffic_df_filtered.astype(
        {column: int for column in change}
    )

    # drop columns
    drop = [
        "remove",
        "completly_in_graph",
        "Edge_path_E_road",
        "Distance_from_origin_region_to_E_road",
        "Distance_within_E_road",
        "Distance_from_E_road_to_destination_region",
        "Total_distance",
    ]
    truck_traffic_df_filtered = truck_traffic_df_filtered.drop(drop, axis=1)

    # sort origin_node_reduced and destination_node_reduced by id (The order of origin destination are irrelevant for the traffic assignment. So this reduces the number of duplicates)
    for index, row in truck_traffic_df_filtered.iterrows():
        if row["origin_node_reduced"] > row["destination_node_reduced"]:
            truck_traffic_df_filtered.loc[
                index, ["origin_node_reduced", "destination_node_reduced"]
            ] = (row["destination_node_reduced"], row["origin_node_reduced"])

    # aggregate duplicates (same origin and destination) by adding up the volume
    # Traffic_flow_trucks_2010	Traffic_flow_trucks_2019	Traffic_flow_trucks_2030	Traffic_flow_tons_2010	Traffic_flow_tons_2019	Traffic_flow_tons_2030
    truck_traffic_df_filtered = (
        truck_traffic_df_filtered.groupby(
            ["origin_node_reduced", "destination_node_reduced"]
        )
        .agg(
            {
                "Traffic_flow_trucks_2010": "sum",
                "Traffic_flow_trucks_2019": "sum",
                "Traffic_flow_trucks_2030": "sum",
                "Traffic_flow_tons_2010": "sum",
                "Traffic_flow_tons_2019": "sum",
                "Traffic_flow_tons_2030": "sum",
            }
        )
        .reset_index()
    )

    # remove rows where origin_node_reduced == destination_node_reduced
    truck_traffic_df_filtered = truck_traffic_df_filtered[
        truck_traffic_df_filtered["origin_node_reduced"]
        != truck_traffic_df_filtered["destination_node_reduced"]
    ]

    # remove rows where all Traffic_flow_trucks and Traffic_flow_tons are 0
    truck_traffic_df_filtered = truck_traffic_df_filtered[
        (
            truck_traffic_df_filtered[
                [
                    "Traffic_flow_trucks_2010",
                    "Traffic_flow_trucks_2019",
                    "Traffic_flow_trucks_2030",
                    "Traffic_flow_tons_2010",
                    "Traffic_flow_tons_2019",
                    "Traffic_flow_tons_2030",
                ]
            ]
            != 0
        ).any(axis=1)
    ]

    # rename origin_node_reduced and destination_node_reduced to origin and destination
    truck_traffic_df_filtered = truck_traffic_df_filtered.rename(
        columns={
            "origin_node_reduced": "origin",
            "destination_node_reduced": "destination",
        }
    )

    # make sure origin and destination are integers
    truck_traffic_df_filtered = truck_traffic_df_filtered.astype(
        {"origin": int, "destination": int}
    )

    # export to csv
    truck_traffic_df_filtered.to_csv(
        f"{country_output_path.absolute()}/traffic_full.csv", index=False
    )

    # drop everything except origin, destination, Traffic_flow_trucks_2019
    truck_traffic_df_filtered = truck_traffic_df_filtered[
        ["origin", "destination", "Traffic_flow_trucks_2019"]
    ]

    # rename Traffic_flow_trucks_2019 to volume
    truck_traffic_df_filtered = truck_traffic_df_filtered.rename(
        columns={"Traffic_flow_trucks_2019": "volume"}
    )

    print(f"Number of Trips: {len(truck_traffic_df_filtered)}")

    # export to csv
    truck_traffic_df_filtered.to_csv(
        f"{country_output_path.absolute()}/traffic.csv", index=False
    )

    # create config file
    config_dict = {
        "graph": {"type": "file", "path": "./graph.graphml"},
        "trips": {"type": "file", "path": "./traffic.csv"},
        "segments": {"type": "file", "path": "./segments.csv"},
    }

    with open(f"{country_output_path.absolute()}/network.yaml", "w") as file:
        yaml.dump(config_dict, file)

    # save info file for the reduced graph
    info = {
        "nodes": len(G_reduced_3.nodes),
        "edges": len(G_reduced_3.edges),
        "segments": total_number_of_segments,
        "trips": len(truck_traffic_df_filtered),
    }

    with open(f"{country_output_path.absolute()}/info.yaml", "w") as file:
        yaml.dump(info, file)


def main(args):
    # load data
    nodes_df = pd.read_csv(os.path.join(args.data_dir, nodes_file_name))
    edges_df = pd.read_csv(os.path.join(args.data_dir, edges_file_name))

    # Create a graph from the edges
    G = nx.from_pandas_edgelist(edges_df, "Network_Node_A_ID", "Network_Node_B_ID")

    # add edge information as attributes to graph
    edge_columns = [
        "Network_Edge_ID",
        "Manually_Added",
        "Distance",
        "Traffic_flow_trucks_2019",
        "Traffic_flow_trucks_2030",
    ]
    for index, row in edges_df.iterrows():
        for column in edge_columns:
            G.edges[row["Network_Node_A_ID"], row["Network_Node_B_ID"]][column] = row[
                column
            ]

    # Create a position dictionary from node coordinates
    pos = {
        row["Network_Node_ID"]: (row["Network_Node_X"], row["Network_Node_Y"])
        for index, row in nodes_df.iterrows()
    }

    # Plotting the network
    plot_network(
        G, pos, f"Europe (N: {len(G.nodes)}, E: {len(G.edges)})", args.output_dir
    )

    countries = nodes_df["Country"].unique()

    # ensure country code is correct
    if args.country == "ALL":
        for country in countries:
            args.country = country
            export_country(args)

    elif args.country in countries:
        export_country(args)

    else:
        raise ValueError(f"Country code {args.country} not found in data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", "-c", type=str, default="BE")
    parser.add_argument(
        "--segment_length",
        "-sl",
        type=float,
        default=10.0,
        help="Length of a segment in km",
    )
    parser.add_argument(
        "--segment_capacity",
        "-sc",
        type=float,
        default=9e6,
        help="Capacity of a segment in trucks per year",
    )
    parser.add_argument(
        "--segment_speed",
        "-ss",
        type=float,
        default=100.0,
        help="Travel speed on a segment in km/h",
    )
    parser.add_argument(
        "--pruning_threshold",
        "-p",
        type=float,
        default=10.0,
        help="Threshold for pruning edges in km",
    )
    parser.add_argument("--data-dir", "-d", type=str, default="environments/graph/data")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="environments/graph/output"
    )

    args = parser.parse_args()

    # check that data is in data directory
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")

    required_files = [
        truck_traffic_file,
        nuts_regions_file,
        nodes_file_name,
        edges_file_name,
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(args.data_dir, file)):
            raise ValueError(
                f"Data file {file} does not exist in {args.data_dir}. Please download the data from {data_url} and extract it to {args.data_dir}"
            )

    # create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
