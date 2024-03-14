from typing import Dict, Optional

import numpy as np


class RoadSegment:
    def __init__(
        self,
        config,
        random_generator,
        position_x,
        position_y,
        capacity,
        base_travel_time,
    ):
        self.random_generator = random_generator
        self.number_of_states = config["deterioration"].shape[1]
        self.initial_damage_prob = config["initial_damage_distribution"]
        self.position_x = position_x
        self.position_y = position_y

        self.capacity = capacity
        self.base_travel_time = base_travel_time

        # base travel time table
        # shape: A
        self.base_travel_time_table = (
            config["traffic"]["base_travel_time_factors"] * self.base_travel_time
        )

        # capacity table
        # shape: A
        self.capacity_table = config["traffic"]["capacity_factors"] * self.capacity

        # deterioration tables
        # shape: A x S x S
        self.deterioration_table = config["deterioration"]

        # observation tables
        # shape: A x S x O
        self.observation_tables = config["observation"]

        # Costs (negative rewards)
        # shape: S x A
        self.state_action_reward = config["reward"]["state_action_reward"]

        self.reset()

    def reset(self):
        self.get_initial_state()

    def step(self, action):
        # actions: [do-nothing, inspect, minor-repair, major-repair, replacement] = [0, 1, 2, 3, 4]

        # Corrective repair action if the worst condition is reached
        if self.state == self.number_of_states - 1:
            action = 4

        next_deterioration_state = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.deterioration_table[action][self.state],
        )

        self.base_travel_time = self.base_travel_time_table[action]
        self.capacity = self.capacity_table[action]

        reward = self.state_action_reward[action][self.state]
        self.state = next_deterioration_state

        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.observation_tables[action][self.state],
        )

        # Belief state computation
        self.belief = self.deterioration_table[action].T @ self.belief

        state_probs = self.observation_tables[action][
            :, self.observation
        ]  # likelihood of observation

        # Bayes' rule
        self.belief = state_probs * self.belief  # likelihood * prior
        self.belief /= np.sum(self.belief)  # normalize

        return reward

    def get_initial_state(self):
        # Computing initial state, observation, and belief

        self.belief = np.array(self.initial_damage_prob)
        self.initial_state = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.initial_damage_prob,
        )
        self.state = self.initial_state
        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.observation_tables[0][self.state],
        )


class RoadEdge:
    def __init__(
        self,
        segments,
        config,
        random_generator,
    ):

        self.segments = segments
        self.number_of_segments = len(segments)
        self.inspection_campaign_reward = config["reward"]["inspection_campaign_reward"]
        self.random_generator = random_generator
        self.bpr_alpha = config["traffic"]["bpr_alpha"]
        self.bpr_beta = config["traffic"]["bpr_beta"]
        self.reset(reset_segments=False)

    def calculate_bpr_capacity_factor(
        self, base_time_vec: np.array, capacity_vec: np.array
    ) -> np.array:
        return base_time_vec * self.bpr_alpha / (capacity_vec**self.bpr_beta)

    def update_edge_travel_time_factors(self) -> None:
        """Updates the edge travel time factors based on the current segment states."""
        btt_vec, cap_vec = np.hsplit(
            np.array([[seg.base_travel_time, seg.capacity] for seg in self.segments]), 2
        )
        self.base_time_factor = np.sum(btt_vec)
        self.capacity_factor = np.sum(
            self.calculate_bpr_capacity_factor(
                base_time_vec=btt_vec, capacity_vec=cap_vec
            )
        )
        return

    def compute_edge_travel_time(self, volume: float) -> float:
        """Computes the travel time for the edge based on the given current volume and precomputed capacity factor."""
        return self.base_time_factor + self.capacity_factor * (volume**self.bpr_beta)

    def step(self, actions):
        if len(self.segments) != len(actions):
            raise ValueError("self.segments and actions must have the same length")

        reward = 0
        for segment, action in zip(self.segments, actions):
            segment_reward = segment.step(action)
            reward += segment_reward

        if 1 in actions:
            reward += self.inspection_campaign_reward

        self.update_edge_travel_time_factors()

        return reward

    def reset(self, reset_segments=True):
        if reset_segments:
            for segment in self.segments:
                segment.reset()
        self.update_edge_travel_time_factors()

    def get_observation(self):
        return [segment.observation for segment in self.segments]

    def get_beliefs(self):
        return [segment.belief for segment in self.segments]

    def get_states(self):
        return [segment.state for segment in self.segments]


class RoadEnvironment:
    def __init__(
        self,
        config: Dict,
        seed: Optional[int] = None,
    ):
        self.random_generator = np.random.default_rng(seed)
        self.max_timesteps = config["general"]["max_timesteps"]

        self.graph = config["network"]["graph"]

        # Convert trips dataframe to list of tuples with correct vertex indices
        trips_df = config["network"]["trips"]
        trips = []
        for index in trips_df.index:
            vertex_1_list = self.graph.vs.select(id_eq=trips_df["origin"][index])
            vertex_2_list = self.graph.vs.select(id_eq=trips_df["destination"][index])
            if (len(vertex_1_list) == 0) or (len(vertex_2_list) == 0):
                raise ValueError(
                    f"Trip not in graph: {trips_df['origin'][index]} -> {trips_df['destination'][index]}"
                )
            trips.append(
                (
                    vertex_1_list[0].index,
                    vertex_2_list[0].index,
                    trips_df["volume"][index],
                )
            )

        self.trips = trips

        # Add road segments to graph edges
        for nodes, edge_segments in config["network"]["segments"].items():
            segments = []
            for segment in edge_segments:
                segments.append(
                    RoadSegment(
                        random_generator=self.random_generator,
                        position_x=segment["position_x"],
                        position_y=segment["position_y"],
                        capacity=segment["capacity"],
                        base_travel_time=segment["travel_time"],
                        config=config["model"]["segment"],
                    )
                )
            road_edge = RoadEdge(
                segments=segments,
                config=config["model"]["edge"],
                random_generator=self.random_generator,
            )

            vertex_1 = self.graph.vs.select(id_eq=nodes[0])
            vertex_2 = self.graph.vs.select(id_eq=nodes[1])
            graph_edge = self.graph.es.select(_between=(vertex_1, vertex_2))[0]
            graph_edge["road_segments"] = road_edge

        # Traffic assignment parameters
        ta_conf = config["network"]["traffic_assignment"]
        self.traffic_assignment_max_iterations = ta_conf["max_iterations"]
        self.traffic_assignment_convergence_threshold = ta_conf["convergence_threshold"]
        self.traffic_assignment_update_weight = ta_conf["update_weight"]

        self.travel_time_reward_factor = config["model"]["network"][
            "travel_time_reward_factor"
        ]

        self.reset(reset_edges=False)

        self.base_total_travel_time = self._get_total_travel_time()

    def reset(self, reset_edges=True):
        self.timestep = 0
        if reset_edges:
            for edge in self.graph.es:
                edge["road_segments"].reset()
        return self._get_observation()

    def _get_observation(self):
        adjacency_matrix = np.array(self.graph.get_adjacency().data)
        edge_observations = []
        edge_nodes = []
        edge_beliefs = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_segments"].get_observation())
            edge_beliefs.append(edge["road_segments"].get_beliefs())
            edge_nodes.append([edge.source, edge.target])

        observations = {
            "adjacency_matrix": adjacency_matrix,
            "edge_observations": edge_observations,
            "edge_beliefs": edge_beliefs,
            "edge_nodes": edge_nodes,
            "time_step": self.timestep,
        }

        return observations

    def _get_states(self):
        edge_states = []
        for edge in self.graph.es:
            edge_states.append(edge["road_segments"].get_states())

        return edge_states

    def _get_total_travel_time(self):
        # Initialize volumes
        self.graph.es["volume"] = 0

        # Initialize with all-or-nothing assignment
        self.graph.es["travel_time"] = [
            edge["road_segments"].compute_edge_travel_time(edge["volume"])
            for edge in self.graph.es
        ]

        for source, target, num_cars in self.trips:
            path = self.graph.get_shortest_paths(
                source, target, weights="travel_time", output="epath"
            )[0]
            for edge_id in path:
                self.graph.es[edge_id]["volume"] += num_cars

        for iteration in range(self.traffic_assignment_max_iterations):
            # Recalculate travel times with current volumes
            self.graph.es["travel_time"] = [
                edge["road_segments"].compute_edge_travel_time(edge["volume"])
                for edge in self.graph.es
            ]

            # Find the shortest paths using updated travel times
            new_volumes = np.zeros(len(self.graph.es))
            for source, target, num_cars in self.trips:
                path = self.graph.get_shortest_paths(
                    source, target, weights="travel_time", output="epath"
                )[0]
                for edge_id in path:
                    new_volumes[edge_id] += num_cars

            # Check for convergence by comparing volume changes
            volume_changes = np.abs(self.graph.es["volume"] - new_volumes)
            max_change = np.max(volume_changes)

            if max_change <= self.traffic_assignment_convergence_threshold:
                break

            # Update volumes by averaging
            self.graph.es["volume"] = (
                np.array(self.graph.es["volume"])
                * (1 - self.traffic_assignment_update_weight)
                + new_volumes * self.traffic_assignment_update_weight
            )

        return np.sum([edge["travel_time"] * edge["volume"] for edge in self.graph.es])

    def step(self, actions):
        maintenance_reward = 0
        for i, edge in enumerate(self.graph.es):
            maintenance_reward += edge["road_segments"].step(actions[i])

        total_travel_time = self._get_total_travel_time()

        travel_time_reward = self.travel_time_reward_factor * (
            total_travel_time - self.base_total_travel_time
        )

        reward = maintenance_reward + travel_time_reward

        observation = self._get_observation()

        self.timestep += 1

        info = {
            "states": self._get_states(),
            "total_travel_time": total_travel_time,
            "travel_times": self.graph.es["travel_time"],
            "volumes": self.graph.es["volume"],
            "reward_elements": [travel_time_reward, maintenance_reward],
        }

        return observation, reward, self.timestep >= self.max_timesteps, info

    def seed(self, seed):
        self.random_generator = np.random.default_rng(seed)
        for edge in self.graph.es:
            edge["road_segments"].random_generator = self.random_generator
            for segment in edge["road_segments"].segments:
                segment.random_generator = self.random_generator
