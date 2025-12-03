import argparse
from pathlib import Path

import matplotlib.patches as patches

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fix_traffic_paths import analyze_and_fix_traffic
from validate_large_graph import validate_graph_structure, validate_trips_connectivity

paper_url = "https://publica-rest.fraunhofer.de/server/api/core/bitstreams/d4913d12-4cd1-473c-97cd-ed467ad19273/content"
data_url = "https://data.mendeley.com/datasets/py2zkrb65h/1"
truck_traffic_file = "01_Trucktrafficflow.csv"
truck_traffic_file_fixed = "01_Trucktrafficflow_fixed.csv"
nuts_regions_file = "02_NUTS-3-Regions.csv"
nodes_file_name = "03_network-nodes.csv"
edges_file_name = "04_network-edges.csv"

NEW_EDGE_ID_OFFSET = 5_000_000  # Highest id in dataset 2616216


def plot_network(G, pos, title, path, args):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Add grid
    ax.grid(True)

    if args.coordinate_range is not None:
        x_min, x_max, y_min, y_max = args.coordinate_range
        box_points = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min),
        ]
        # Create a Polygon patch
        polygon = patches.Polygon(
            box_points,
            closed=True,
            edgecolor="r",
            linewidth=2,
            facecolor="none",
            zorder=10,
        )

        # Add the patch to the Axes
        ax.add_patch(polygon)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Save the figure
    if path is not None:
        outdir = Path(path)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{title}.svg")


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
    id = NEW_EDGE_ID_OFFSET
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
    if string_list == "[]":
        return []
    else:
        return [int(x) for x in string_list[1:-1].split(",")]


def edge_nodes_get_shared_node(edge_nodes_1, edge_nodes_2):
    set_1 = set(edge_nodes_1)
    set_2 = set(edge_nodes_2)
    shared_nodes = set_1.intersection(set_2)
    if len(shared_nodes) == 1:
        return shared_nodes.pop()
    else:
        raise Exception(
            f"Could not determine shared node between edges. Edge 1 nodes: {edge_nodes_1}, Edge 2 nodes: {edge_nodes_2}"
        )


def export_coordinate_range(args):
    print(f"Exporting graph for coordinate range {args.coordinate_range}")
    # load data
    nodes_df = pd.read_csv(args.data_dir / nodes_file_name)
    edges_df = pd.read_csv(args.data_dir / edges_file_name)

    # Create folder for output
    folder_name = "_".join([str(c) for c in args.coordinate_range])
    output_path = args.output_dir / "coordinate_ranges" / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    min_x, max_x, min_y, max_y = args.coordinate_range
    filtered_nodes = nodes_df[
        (nodes_df["Network_Node_X"] > min_x)
        & (nodes_df["Network_Node_X"] < max_x)
        & (nodes_df["Network_Node_Y"] > min_y)
        & (nodes_df["Network_Node_Y"] < max_y)
    ]
    filtered_edges = edges_df[
        edges_df["Network_Node_A_ID"].isin(filtered_nodes["Network_Node_ID"])
        & edges_df["Network_Node_B_ID"].isin(filtered_nodes["Network_Node_ID"])
    ]

    export_graph(filtered_nodes, filtered_edges, output_path, args)


def export_country(args):
    print(f"Exporting graph for {args.country}")
    # load data
    nodes_df = pd.read_csv(args.data_dir / nodes_file_name)
    edges_df = pd.read_csv(args.data_dir / edges_file_name)

    # Create folder for country
    country_output_path = args.output_dir / "countries" / args.country
    country_output_path.mkdir(parents=True, exist_ok=True)

    # Filtering for a Specific Country
    filtered_nodes = nodes_df[nodes_df["Country"] == args.country]
    filtered_edges = edges_df[
        edges_df["Network_Node_A_ID"].isin(filtered_nodes["Network_Node_ID"])
        & edges_df["Network_Node_B_ID"].isin(filtered_nodes["Network_Node_ID"])
    ]

    export_graph(filtered_nodes, filtered_edges, country_output_path, args)


def export_graph(filtered_nodes, filtered_edges, output_path, args):
    nodes_df = pd.read_csv(args.data_dir / nodes_file_name)
    edges_df = pd.read_csv(args.data_dir / edges_file_name)

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
        output_path,
        args,
    )

    # Add position data for nodes to graph
    position_dict = {
        id: {"position_x": pos[0], "position_y": pos[1]}
        for id, pos in pos_filtered.items()
    }
    nx.set_node_attributes(G_filtered, position_dict)

    # Export Fully graph to graphml
    nx.write_graphml_lxml(G_filtered, output_path / "graph_full.graphml")

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

    nx.set_node_attributes(G_reduced_3, position_dict)

    # Plotting the network
    plot_network(
        G_reduced_3,
        pos_filtered,
        f"Reduced Network for {args.country} (N: {len(G_reduced_3.nodes)}, E: {len(G_reduced_3.edges)})",
        output_path,
        args,
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

    # Make the graph directed
    # Note: converting to a directed graph duplicates each undirected edge
    # into (u->v) and (v->u). Both directed edges retain the same 'id'.
    # This duplicate id across directions is expected.
    G_reduced_3 = G_reduced_3.to_directed()

    # Validate reduced graph structure before saving
    if not args.no_validate:
        print("\tValidating graph structure...")
        validate_graph_structure(G_reduced_3)
        print("\tGraph validation passed!")

    # Export graph to graphml
    nx.write_graphml_lxml(G_reduced_3, output_path / "graph.graphml")

    # store new edge information as yaml
    with open(output_path / "new-edges.yaml", "w") as file:
        yaml.dump(new_edge_info, file)

    print(f"\tNumber of nodes: {len(G_reduced_3.nodes)}")
    print(f"\tNumber of edges: {len(G_reduced_3.edges)}")

    total_number_of_segments = 0
    segments = []
    for edge in G_reduced_3.edges():
        node_a = G_reduced_3.nodes()[edge[0]]
        node_b = G_reduced_3.nodes()[edge[1]]
        if args.segment_length is not None:  # split edges into segments
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
                    (
                        edge[0],
                        edge[1],
                        x,
                        y,
                        args.segment_capacity,
                        args.segment_speed,
                        args.segment_length,
                    )
                )
        else:  # one segment per edge
            no_segments = 1
            (x, y) = (node_a["position_x"], node_a["position_y"])
            segments.append(
                (
                    edge[0],
                    edge[1],
                    x,
                    y,
                    args.segment_capacity,
                    args.segment_speed,
                    G_reduced_3.edges[edge]["distance"],
                )
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
            "travel_speed",
            "segment_length",
        ],
    )
    # save
    segments_df.to_csv(output_path / "segments.csv", index=False)

    print(f"\tTotal number of segments: {total_number_of_segments}")

    info = {
        "nodes": len(G_reduced_3.nodes),
        "edges": len(G_reduced_3.edges),
        "segments": total_number_of_segments,
    }

    if not args.skip_traffic:
        # Edge ID to nodes lookup
        edge_id_to_nodes = {}
        for _, row in edges_df.iterrows():
            eid = row["Network_Edge_ID"]
            u, v = row["Network_Node_A_ID"], row["Network_Node_B_ID"]
            edge_id_to_nodes[eid] = (u, v)

        nodes_in_graph = [
            row["Network_Node_ID"]
            for index, row in (
                nodes_df[nodes_df["Network_Node_ID"].isin(G_reduced_3.nodes())]
            ).iterrows()
        ]

        # add nodes which have been removed during reduction
        for edge in new_edge_info:
            nodes_in_graph += edge["node_ids"]

        all_edges_in_graph = [
            G_reduced_3.edges[edge]["id"] for edge in G_reduced_3.edges()
        ]
        edges_in_graph = [
            edge for edge in all_edges_in_graph if edge < NEW_EDGE_ID_OFFSET
        ]

        # Add edges which have been removed during reduction
        for edge in new_edge_info:
            edges_in_graph += edge["edge_ids"]

        edges_in_graph_set = set(edges_in_graph)

        # load truck traffic data (fixed version)
        fixed_traffic_path = args.data_dir / truck_traffic_file_fixed
        print(f"\tLoading truck traffic data from: {fixed_traffic_path}")
        truck_traffic_df = pd.read_csv(fixed_traffic_path)

        # Filtering trips (include only those that traverse the reduced graph)
        # Logic
        # 1) Each trip row contains Edge_path_E_road: an ordered
        #   list of edge IDs from origin to destination.
        # 2) Build edges_in_graph_set = IDs present in the reduced graph
        #    (including both original and newly created edges).
        # 3) Scan the path using a simple state machine:
        #    - ENTER a segment when we hit the first in-graph edge:
        #        origin := non-shared endpoint of the first two edges
        #        (via edge_nodes_get_shared_node on their endpoints).
        #    - STAY inside while edges remain in edges_in_graph_set.
        #    - EXIT the segment on the first out-of-graph edge after being inside:
        #        destination := shared endpoint between the last in-graph edge
        #        and the current out-of-graph edge → record OD.
        # 4) If we reach the end still inside a segment:
        #        destination := non-shared endpoint of the last two edges → record OD.
        # 5) Each detected in-graph segment yields one OD pair with traffic
        #    attributes copied from the trip row (keys in trip_export_keys).
        print("\tFinding trips which go through the filtered graph")
        found_trips = []
        trip_export_keys = [
            "Traffic_flow_trucks_2010",
            "Traffic_flow_trucks_2019",
            "Traffic_flow_trucks_2030",
            "Traffic_flow_tons_2010",
            "Traffic_flow_tons_2019",
            "Traffic_flow_tons_2030",
        ]

        for row in tqdm(
            truck_traffic_df.itertuples(index=False),
            total=len(truck_traffic_df),
        ):
            path_edges = parse_string_list_of_integer(row.Edge_path_E_road)

            if len(path_edges) < 2:
                continue

            state = "find_trip_start"
            start_node = None
            end_node = None

            first_edge = path_edges[0]
            if first_edge in edges_in_graph_set:
                state = "find_trip_end"
                first_edge_nodes = set(edge_id_to_nodes[first_edge])
                second_edge_nodes = set(edge_id_to_nodes[path_edges[1]])
                shared_node = edge_nodes_get_shared_node(
                    first_edge_nodes, second_edge_nodes
                )
                start_node = first_edge_nodes.difference(set([shared_node])).pop()

            for i, edge in enumerate(path_edges[1:]):
                if state == "find_trip_start":
                    if edge in edges_in_graph_set:
                        state = "find_trip_end"
                        edge_nodes = edge_id_to_nodes[edge]
                        last_edge_nodes = edge_id_to_nodes[path_edges[i]]
                        start_node = edge_nodes_get_shared_node(
                            edge_nodes, last_edge_nodes
                        )
                elif state == "find_trip_end":
                    if edge not in edges_in_graph_set:
                        state = "find_trip_start"
                        edge_nodes = edge_id_to_nodes[edge]
                        last_edge_nodes = edge_id_to_nodes[path_edges[i]]
                        end_node = edge_nodes_get_shared_node(
                            edge_nodes, last_edge_nodes
                        )
                        found_trips.append(
                            {
                                "origin_node": start_node,
                                "destination_node": end_node,
                                **{key: getattr(row, key) for key in trip_export_keys},
                            }
                        )
            if state == "find_trip_end":
                # trip ends at last edge
                last_edge_nodes = set(edge_id_to_nodes[path_edges[-1]])
                second_last_edge_nodes = set(edge_id_to_nodes[path_edges[-2]])
                shared_node = edge_nodes_get_shared_node(
                    last_edge_nodes, second_last_edge_nodes
                )
                end_node = last_edge_nodes.difference(set([shared_node])).pop()
                found_trips.append(
                    {
                        "origin_node": start_node,
                        "destination_node": end_node,
                        **{key: getattr(row, key) for key in trip_export_keys},
                    }
                )

        truck_traffic_df_filtered = pd.DataFrame(found_trips)

        # Create reduced graph node lookup table
        new_node_info_lookup = {}

        for edge in new_edge_info:
            for node in edge["node_ids"]:
                new_node_info_lookup[node] = [
                    edge[ab] for ab in ["Network_Node_A_ID", "Network_Node_B_ID"]
                ]

        # Create coordinate lookup for all nodes (including removed ones)
        all_node_coords = {
            row["Network_Node_ID"]: (row["Network_Node_X"], row["Network_Node_Y"])
            for _, row in nodes_df.iterrows()
        }

        def get_closest_endpoint(removed_node):
            """Return the endpoint node closest to the removed node using Euclidean distance."""
            endpoints = new_node_info_lookup[removed_node]
            removed_coords = all_node_coords[removed_node]
            endpoint_a_coords = all_node_coords[endpoints[0]]
            endpoint_b_coords = all_node_coords[endpoints[1]]
            dist_to_a = (removed_coords[0] - endpoint_a_coords[0]) ** 2 + (
                removed_coords[1] - endpoint_a_coords[1]
            ) ** 2
            dist_to_b = (removed_coords[0] - endpoint_b_coords[0]) ** 2 + (
                removed_coords[1] - endpoint_b_coords[1]
            ) ** 2
            return endpoints[0] if dist_to_a <= dist_to_b else endpoints[1]

        nodes_in_reduced_graph_set = set(G_reduced_3.nodes())

        # lookup new nodes in reduced graph
        for index, row in truck_traffic_df_filtered.iterrows():
            origin_node = int(row["origin_node"])
            destination_node = int(row["destination_node"])

            if origin_node not in nodes_in_reduced_graph_set:
                origin_node = get_closest_endpoint(origin_node)
            if destination_node not in nodes_in_reduced_graph_set:
                destination_node = get_closest_endpoint(destination_node)
            truck_traffic_df_filtered.loc[index, "origin_node_reduced"] = origin_node
            truck_traffic_df_filtered.loc[index, "destination_node_reduced"] = (
                destination_node
            )

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

        # Validate trips connectivity before saving full traffic
        if not args.no_validate:
            print("\tValidating trips connectivity...")
            validate_trips_connectivity(
                G_reduced_3,
                truck_traffic_df_filtered,
                origin_col="origin",
                destination_col="destination",
            )
            print("\tTrips validation passed!")

        # export to csv
        truck_traffic_df_filtered.to_csv(output_path / "traffic_full.csv", index=False)

        # drop everything except origin, destination, Traffic_flow_trucks_2019
        truck_traffic_df_filtered = truck_traffic_df_filtered[
            ["origin", "destination", "Traffic_flow_trucks_2019"]
        ]

        # rename Traffic_flow_trucks_2019 to volume
        truck_traffic_df_filtered = truck_traffic_df_filtered.rename(
            columns={"Traffic_flow_trucks_2019": "volume"}
        )

        print(f"\tNumber of Trips: {len(truck_traffic_df_filtered)}")

        # export to csv
        truck_traffic_df_filtered.to_csv(output_path / "traffic.csv", index=False)

        info["trips"] = len(truck_traffic_df_filtered)

    # store info
    with open(output_path / "info.yaml", "w") as file:
        yaml.dump(info, file)

    # create config file
    config_dict = {
        "graph": {"type": "file", "path": "./graph.graphml"},
        "trips": {"type": "file", "path": "./traffic.csv"},
        "segments": {"type": "file", "path": "./segments.csv"},
    }

    with open(output_path / "network.yaml", "w") as file:
        yaml.dump(config_dict, file)


def main(args):
    # Normalize to Path (in case called programmatically)
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    # load data
    nodes_df = pd.read_csv(args.data_dir / nodes_file_name)
    edges_df = pd.read_csv(args.data_dir / edges_file_name)

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
        G, pos, f"Europe (N: {len(G.nodes)}, E: {len(G.edges)})", args.output_dir, args
    )

    countries = nodes_df["Country"].unique()

    # ensure country code is correct
    if args.country == "ALL":
        for country in countries:
            args.country = country
            export_country(args)

    elif args.country in countries:
        export_country(args)

    elif args.coordinate_range is not None:
        export_coordinate_range(args)

    else:
        raise ValueError(
            "Use --country [CODE] or --coordinate_range [min_x max_x min_y max_y] to specify export area."
        )


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", "-c", type=str, default=None)
    parser.add_argument(
        "--coordinate-range",
        "-cr",
        type=float,
        nargs=4,
        default=None,
        help="Coordinate range to filter for. Format: min_x max_x min_y max_y",
    )
    parser.add_argument(
        "--segment-length",
        "-sl",
        type=float,
        default=None,
        help="Length of a segment in km",
    )
    parser.add_argument(
        "--segment-capacity",
        "-sc",
        type=float,
        default=9e6,
        help="Capacity of a segment in trucks per year",
    )
    parser.add_argument(
        "--segment-speed",
        "-ss",
        type=float,
        default=100.0,
        help="Travel speed on a segment in km/h",
    )
    parser.add_argument(
        "--pruning-threshold",
        "-p",
        type=float,
        default=1.0,
        help="Threshold for pruning edges in km",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data",
        help="Directory containing input data (default: <script_dir>/data)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory for output files (default: <script_dir>/output)",
    )

    parser.add_argument("--skip-traffic", action="store_true", default=False)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip validation of graph and trips before saving",
    )

    args = parser.parse_args()

    # check that data is in data directory
    if not args.data_dir.exists():
        raise ValueError(f"Data directory {args.data_dir} does not exist")

    required_files = [
        nuts_regions_file,
        nodes_file_name,
        edges_file_name,
    ]
    for file in required_files:
        if not (args.data_dir / file).exists():
            raise ValueError(
                f"Data file {file} does not exist in {args.data_dir}. Please download the data from {data_url} and extract it to {args.data_dir}"
            )

    if not args.skip_traffic:
        # Check if fixed traffic file exists, create it if not
        fixed_traffic_path = args.data_dir / truck_traffic_file_fixed
        if not fixed_traffic_path.exists():
            print(
                "Fixed traffic file not found, creating it... (This may take a few minutes)"
            )
            # Load required data files
            edges_df = pd.read_csv(args.data_dir / edges_file_name)
            regions_df = pd.read_csv(args.data_dir / nuts_regions_file)
            if not (args.data_dir / truck_traffic_file).exists():
                raise ValueError(
                    f"Truck traffic file {truck_traffic_file} does not exist in {args.data_dir}. Please download the data from {data_url} and extract it to {args.data_dir}"
                )
            traffic_df = pd.read_csv(args.data_dir / truck_traffic_file)

            fixed_traffic_df = analyze_and_fix_traffic(
                traffic_df, edges_df, regions_df, fix=True
            )
            fixed_traffic_df.to_csv(fixed_traffic_path, index=False)
            print(f"Saved fixed traffic data to: {fixed_traffic_path}")

    main(args)
