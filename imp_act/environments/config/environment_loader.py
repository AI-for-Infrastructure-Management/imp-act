from pathlib import Path

import numpy as np

import pandas as pd
import yaml
from igraph import Graph

from ..road_env import RoadEnvironment


class EnvironmentLoader:
    def __init__(self, filename):
        self.filename = filename
        self.config = self._load(filename)

    def _load(self, filename):
        """Load the environment from the config file"""
        config = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)

        root_path = Path(filename).parent
        config = self._handle_includes(config, root_path=root_path)

        config = self._check_params(config)

        # load network
        network_config = config["network"]

        # load graph
        graph_config = network_config["graph"]
        if graph_config["type"] == "file":
            path = Path(graph_config["path"])
            graph = Graph.Read_GraphML(open(path, "r"))
            graph.vs["id"] = [int(v["id"]) for v in graph.vs]
        elif graph_config["type"] == "list":
            graph = Graph(directed=False)

            nodes_list = graph_config["nodes"]
            nodes = [n["id"] for n in nodes_list]
            node_attributes = {
                key: [n[key] for n in nodes_list] for key in nodes_list[0].keys()
            }
            graph.add_vertices(nodes, attributes=node_attributes)

            edges_list = graph_config["edges"]
            edges = [(e["source"], e["target"]) for e in edges_list]
            edge_attributes = {
                key: [e[key] for e in edges_list]
                for key in edges_list[0].keys()
                if key not in ["source", "target"]
            }
            graph.add_edges(edges, attributes=edge_attributes)
        else:
            raise ValueError(f"Graph type {graph_config['type']} not supported")

        config["network"]["graph"] = graph

        # load trips
        trips_config = network_config["trips"]
        if trips_config["type"] == "file":
            path = Path(trips_config["path"])
            trips = pd.read_csv(path)
            # ensure that origin, destination are integers
            trips = trips.astype({"origin": int, "destination": int, "volume": float})
        elif trips_config["type"] == "list":
            trips = pd.DataFrame(trips_config["list"])
        else:
            raise ValueError(f"Trips type {trips_config['type']} not supported")

        config["network"]["trips"] = trips

        # load segments
        segments_config = network_config["segments"]
        if segments_config["type"] == "file":
            path = Path(segments_config["path"])
            segments_df = pd.read_csv(path)
        elif segments_config["type"] == "list":
            segments_df = pd.DataFrame(segments_config["list"])
        else:
            raise ValueError(f"Segments type {segments_config['type']} not supported")

        # group segments by origin, destination
        segments = {}
        for group, df in segments_df.groupby(["source", "target"]):
            segments[group] = df.to_dict("records")

        config["network"]["segments"] = segments

        # load model
        segment = config["model"]["segment"]
        if segment["deterioration"]["type"] == "list":
            segment["deterioration"] = np.array(segment["deterioration"]["list"])
        else:
            raise ValueError(
                f"Deterioration type {segment['deterioration']['type']} not supported"
            )

        if segment["observation"]["type"] == "list":
            segment["observation"] = np.array(segment["observation"]["list"])
        else:
            raise ValueError(
                f"Deterioration type {segment['observation']['type']} not supported"
            )

        if segment["reward"]["state_action_reward"]["type"] == "list":
            segment["reward"]["state_action_reward"] = np.array(
                segment["reward"]["state_action_reward"]["list"]
            )
        else:
            raise ValueError(
                f"Deterioration type {segment['state_action_reward']['type']} not supported"
            )

        traffic = config["model"]["segment"]["traffic"]
        if traffic["base_travel_time_factors"]["type"] == "list":
            traffic["base_travel_time_factors"] = np.array(
                traffic["base_travel_time_factors"]["list"]
            )
        else:
            raise ValueError(
                f"Deterioration type {traffic['base_travel_time_factors']['type']} not supported"
            )

        if traffic["capacity_factors"]["type"] == "list":
            traffic["capacity_factors"] = np.array(traffic["capacity_factors"]["list"])
        else:
            raise ValueError(
                f"Deterioration type {traffic['capacity_factors']['type']} not supported"
            )

        self._check_model_values(config)

        return config

    def _handle_includes(self, config, root_path):
        """Handle includes in the config dict by recursively loading them and updating the config."""
        self._handle_relative_paths(config, root_path)
        if "include" in config:
            include_path = config["include"]["path"]
            include_root_path = Path(include_path).parent
            include_config = yaml.load(open(include_path, "r"), Loader=yaml.FullLoader)
            include_config = self._handle_includes(include_config, include_root_path)
            override = config["include"].get("override", True)
            self._recursive_update(config, include_config, override)
        for key in config.keys():
            if isinstance(config[key], dict):
                config[key] = self._handle_includes(config[key], root_path)
        return config

    def _recursive_update(self, first_dict, update_dict, override=True):
        """Recursively update the first_dict with the update_dict"""
        for key, value in update_dict.items():
            if key in first_dict:
                if isinstance(value, dict):
                    self._recursive_update(first_dict[key], value, override)
                else:
                    if override:
                        first_dict[key] = value
            else:
                first_dict[key] = value
        return first_dict

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

    def _check_model_values(self, config):
        # Ensure transition matrix values sum to 1
        # Shape: A x S x S
        deterioration_table = config["model"]["segment"]["deterioration"]
        if not np.allclose(deterioration_table.sum(axis=2), 1):
            raise ValueError("Transition matrix rows do not sum to 1")

        # Ensure do-nothing matrix is upper triangular
        # Shape: S x S
        if not np.allclose(np.triu(deterioration_table[0]), deterioration_table[0]):
            raise ValueError("Transition matrix is not upper triangular")
        if not np.allclose(deterioration_table[0], deterioration_table[1]):
            raise ValueError(
                "Transition for inspection and do-nothing are not the same"
            )

        # Ensure observation matrix values sum to 1
        # Shape: A x S x S
        observation_table = config["model"]["segment"]["observation"]
        if not np.allclose(observation_table.sum(axis=2), 1):
            raise ValueError("Observation matrix rows do not sum to 1")

        # Ensure reward matrix is valid
        # Shape: A
        reward_table = config["model"]["segment"]["reward"]["state_action_reward"]
        if np.any(reward_table > 0):
            raise ValueError("Reward matrix has values greater than 0")

        # Ensure base_travel_time_factors matrix is valid
        # Shape: A
        base_travel_time_factors = config["model"]["segment"]["traffic"][
            "base_travel_time_factors"
        ]
        if np.any(base_travel_time_factors < 1):
            raise ValueError("base_travel_time_factors vector has values less than 1")

        # Ensure traffic capacity_factors matrix is valid
        # Shape: A
        capacity_factors = config["model"]["segment"]["traffic"]["capacity_factors"]
        if np.any(capacity_factors > 1):
            raise ValueError("capacity_factors vector has values greater than 1")

    def to_numpy(self):
        return RoadEnvironment(self.config)

    def to_jax(self):
        pass
