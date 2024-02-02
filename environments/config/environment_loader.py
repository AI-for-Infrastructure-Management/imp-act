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
        elif graph_config["type"] in ["dict", "random"]:
            raise NotImplementedError(
                f"Graph type {graph_config['type']} has not implemented yet"
            )
        else:
            raise ValueError(f"Graph type {graph_config['type']} not supported")

        # load trips
        trips_config = network_config["trips"]
        if trips_config["type"] == "file":
            path = Path(trips_config["path"])
            self.trips = pd.read_csv(path)
            # ensure that origin, destination are integers
            self.trips["origin"] = self.trips["origin"].astype(int)
            self.trips["destination"] = self.trips["destination"].astype(int)

        elif trips_config["type"] in ["dict", "random"]:
            raise NotImplementedError(
                f"Trips type {trips_config['type']} has not implemented yet"
            )
        else:
            raise ValueError(f"Trips type {trips_config['type']} not supported")

        # load segments
        segments_config = network_config["segments"]
        if segments_config["type"] == "file":
            path = Path(segments_config["path"])
            self.segments = pd.read_csv(path)
            # group segments by origin, destination
            segments = {}
            for group, df in self.segments.groupby(
                ["Network_Node_A_ID", "Network_Node_B_ID"]
            ):
                segments[group] = df.to_dict("records")
            self.segments = segments
        elif segments_config["type"] in ["dict", "random"]:
            raise NotImplementedError(
                f"Segments type {segments_config['type']} has not implemented yet"
            )
        else:
            raise ValueError(f"Segments type {segments_config['type']} not supported")

        # load model
        # TODO: load model data

    def _handle_includes(self, config, root_path):
        """Handle includes in the config file"""
        self._handle_relative_paths(config, root_path)
        if "include" in config.keys():
            include_path = config["include"]["path"]
            include_root_path = Path(include_path).parent
            include_config = yaml.load(open(include_path, "r"), Loader=yaml.FullLoader)
            include_config = self._handle_includes(include_config, include_root_path)
            config = {**config, **include_config}
        for key in config.keys():
            if isinstance(config[key], dict):
                config[key] = self._handle_includes(config[key], root_path)
        return config

    def _handle_relative_paths(self, config, root_path):
        """Handle relative paths in the config file"""
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
