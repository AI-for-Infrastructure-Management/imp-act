from pathlib import Path

import pandas as pd
import yaml
from igraph import Graph

from environments.road_env import RoadEnvironment


class EnvironmentLoader:
    def __init__(self, filename):
        self.filename = filename
        self.root_path = Path(filename).parent
        self._load(filename)

    def _load(self, filename):
        """Load the environment from the config file"""
        config = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)

        config = self._handle_includes(config, root_path=self.root_path)

        self.config = self._check_params(config)

        # load general
        self.general = config["general"]
        self.max_timesteps = self.general["max_timesteps"]

        # load network
        network_config = config["network"]
        self.network = network_config

        # load graph
        graph_config = network_config["graph"]
        if graph_config["type"] == "file":
            path = Path(graph_config["path"])
            self.graph = Graph.Read_GraphML(open(path, "r"))
            self.graph.vs["id"] = [int(v["id"]) for v in self.graph.vs]
        elif graph_config["type"] == "list":
            self.graph = Graph(directed=False)

            nodes_list = graph_config["nodes"]
            nodes = [n["id"] for n in nodes_list]
            node_attributes = {
                key: [n[key] for n in nodes_list] for key in nodes_list[0].keys()
            }
            self.graph.add_vertices(nodes, attributes=node_attributes)

            edges_list = graph_config["edges"]
            edges = [(e["source"], e["target"]) for e in edges_list]
            edge_attributes = {
                key: [e[key] for e in edges_list]
                for key in edges_list[0].keys()
                if key not in ["source", "target"]
            }
            self.graph.add_edges(edges, attributes=edge_attributes)
        else:
            raise ValueError(f"Graph type {graph_config['type']} not supported")

        # load trips
        trips_config = network_config["trips"]
        if trips_config["type"] == "file":
            path = Path(trips_config["path"])
            self.trips = pd.read_csv(path)
            # ensure that origin, destination are integers
            self.trips = self.trips.astype(
                {"origin": int, "destination": int, "volume": float}
            )
        elif trips_config["type"] == "list":
            self.trips = pd.DataFrame(trips_config["list"])
        else:
            raise ValueError(f"Trips type {trips_config['type']} not supported")

        # load segments
        segments_config = network_config["segments"]
        if segments_config["type"] == "file":
            path = Path(segments_config["path"])
            self.segments = pd.read_csv(path)
        elif segments_config["type"] == "list":
            self.segments = pd.DataFrame(segments_config["list"])
        else:
            raise ValueError(f"Segments type {segments_config['type']} not supported")

        # group segments by origin, destination
        segments = {}
        for group, df in self.segments.groupby(["source", "target"]):
            segments[group] = df.to_dict("records")
        self.segments = segments

        # load model
        # TODO: load model data

    def _handle_includes(self, config, root_path):
        """Handle includes in the config dict by recursively loading them and updating the config."""
        self._handle_relative_paths(config, root_path)
        if "include" in config:
            include_path = config["include"]["path"]
            include_root_path = Path(include_path).parent
            include_config = yaml.load(open(include_path, "r"), Loader=yaml.FullLoader)
            include_config = self._handle_includes(include_config, include_root_path)
            config.update(include_config)
        for key in config.keys():
            if isinstance(config[key], dict):
                config[key] = self._handle_includes(config[key], root_path)
        return config

    def _handle_relative_paths(self, config, root_path):
        """
        Recursively handle relative paths in the config dict by converting them to absolute paths based
        on the given root path.
        """
        if "path" in config.keys():
            config["path"] = Path(root_path, config["path"]).absolute()
        for key in config.keys():
            if isinstance(config[key], dict):
                config[key] = self._handle_relative_paths(config[key], root_path)
        return config

    def _check_params(self, config):
        """Ensure that all required parameters are specified in the config file"""
        required_top_level_parameter = ["general", "model", "network"]
        for param in required_top_level_parameter:
            if param not in config.keys():
                raise ValueError("Missing required parameter: {}".format(param))
        return config

    def to_numpy(self):
        return RoadEnvironment.from_config(self)

    def to_jax(self):
        pass
