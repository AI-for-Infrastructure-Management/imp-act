from typing import Dict, Optional

import numpy as np

import igraph as ig


MILES_PER_KILOMETER = 0.621371
KILOMETERS_PER_MILE = 1.0 / MILES_PER_KILOMETER


class RoadSegment:
    def __init__(
        self,
        config,
        random_generator,
        position_x,
        position_y,
        capacity,
        segment_length,
        base_travel_time,
    ):
        self.random_generator = random_generator
        self.number_of_states = config["maintenance"]["deterioration"].shape[-1]
        self.initial_damage_prob = config["maintenance"]["initial_damage_distribution"]
        self.position_x = position_x
        self.position_y = position_y

        self.capacity = capacity
        self.segment_length = segment_length
        self.base_travel_time = base_travel_time

        # base travel time table
        # shape: A
        self.base_travel_time_table = (
            config["traffic"]["base_travel_time_factors"] * self.base_travel_time
        )

        # capacity table
        # shape: A
        self.capacity_table = config["traffic"]["capacity_factors"] * self.capacity

        # action durations
        # shape: A
        self.action_durations = config["maintenance"]["action_duration_factors"]

        # deterioration tables
        # shape: A x S x S or A x DR x S x S
        self.deterioration_table = config["maintenance"]["deterioration"]
        self.deterioration_rate_enabled = self.deterioration_table.ndim == 4
        if self.deterioration_rate_enabled:
            self.deterioration_rate_max = self.deterioration_table.shape[1]

        # observation tables
        # shape: A x S x O
        self.observation_tables = config["maintenance"]["observation"]

        # Costs (negative rewards)
        # shape: S x A
        self.state_action_reward = config["maintenance"]["reward"][
            "state_action_reward"
        ]

        # terminal state rewards
        # shape: S
        self.terminal_state_reward = config["maintenance"]["reward"][
            "terminal_state_reward"
        ]

        self.reset()

    def reset(self):
        self.forced_repair = False
        self.worst_observation_counter = 0
        self.action_duration = 0
        self.deterioration_rate = 0
        self.belief = np.array(self.initial_damage_prob)
        self.state = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.initial_damage_prob,
        )
        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.observation_tables[0][self.state],
        )

    def step(self, action):
        # actions: [do-nothing, inspect, minor-repair, major-repair, replacement] = [0, 1, 2, 3, 4]

        if self.deterioration_rate_enabled:
            transition_probabilities = self.deterioration_table[action][
                self.deterioration_rate
            ][self.state]
        else:
            transition_probabilities = self.deterioration_table[action][self.state]

        next_deterioration_state = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=transition_probabilities,
        )

        self.base_travel_time = self.base_travel_time_table[action]
        self.capacity = self.capacity_table[action]

        if self.forced_repair:
            reward = self.state_action_reward[-1][-1]
            self.forced_repair = False
            self.worst_observation_counter = 0
        else:
            reward = self.get_action_reward(action=action)

        self.state = next_deterioration_state

        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.observation_tables[action][self.state],
        )

        # Belief state computation
        if self.deterioration_rate_enabled:
            self.belief = (
                self.deterioration_table[action][self.deterioration_rate].T
                @ self.belief
            )
        else:
            self.belief = self.deterioration_table[action].T @ self.belief

        state_probs = self.observation_tables[action][
            :, self.observation
        ]  # likelihood of observation

        # Bayes' rule
        self.belief = state_probs * self.belief  # likelihood * prior
        self.belief /= np.sum(self.belief)  # normalize

        if self.deterioration_rate_enabled:
            if action == 4:
                self.deterioration_rate = 0
            else:
                self.deterioration_rate += 1
                if self.deterioration_rate > self.deterioration_rate_max:
                    raise ValueError(
                        f"Deterioration rate exceeded maximum value {self.deterioration_rate_max}"
                    )

        self.action_duration = self.action_durations[action]

        return reward

    def get_initial_state(self):
        # Computing initial state, observation, and belief
        self.deterioration_rate = 0
        self.forced_repair_interest_counter = 0
        self.belief = np.array(self.initial_damage_prob)
        self.initial_state = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.initial_damage_prob,
        )
        self.state = self.initial_state
        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states),
            p=self.observation_tables[self.ACTION_DO_NOTHING][self.state],
        )

    def get_action_reward(self, action):
        reward = self.state_action_reward[action][self.state]
        reward *= self.segment_length * MILES_PER_KILOMETER

        return reward

    def get_terminal_reward(self):
        return (
            np.sum(self.terminal_state_reward * self.belief)
            * self.segment_length
            * MILES_PER_KILOMETER
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
        self.inspection_campaign_reward = config["maintenance"]["reward"][
            "inspection_campaign_reward"
        ]
        if self.inspection_campaign_reward != 0:
            raise NotImplementedError(
                "Inspection campaign reward is not currently implemented with hard budget constraints."
            )
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

    def get_deterioration_rate(self):
        return [segment.deterioration_rate for segment in self.segments]

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
        self.max_timesteps = config["maintenance"]["max_timesteps"]
        self.forced_replace_worst_observation_count = config["maintenance"][
            "forced_replace_worst_observation_count"
        ]

        self.graph = config["topology"]["graph"]

        # Convert trips dataframe to list of tuples with correct vertex indices
        trips_df = config["traffic"]["trips"]
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
        for nodes, edge_segments in config["topology"]["segments"].items():
            segments = []
            for segment in edge_segments:

                if segment.get("segment_length") is None:
                    segment["segment_length"] = KILOMETERS_PER_MILE

                if segment.get("travel_time") is None:
                    base_travel_time = (
                        segment["segment_length"] / segment["travel_speed"]
                    )
                else:
                    base_travel_time = segment["travel_time"]

                segments.append(
                    RoadSegment(
                        random_generator=self.random_generator,
                        position_x=segment["position_x"],
                        position_y=segment["position_y"],
                        capacity=segment["capacity"],
                        segment_length=segment["segment_length"],
                        base_travel_time=base_travel_time,
                        config=config,
                    )
                )
            road_edge = RoadEdge(
                segments=segments,
                config=config,
                random_generator=self.random_generator,
            )

            edge_id = self.graph.get_eid(
                self.graph.vs.find(id=nodes[0]).index,
                self.graph.vs.find(id=nodes[1]).index,
            )
            graph_edge = self.graph.es[edge_id]
            graph_edge["road_edge"] = road_edge

        # Budget parameters
        self.budget_amount = config["maintenance"]["budget_amount"]
        assert type(self.budget_amount) in [int, float]
        self.budget_amount = float(self.budget_amount)
        self.budget_renewal_interval = config["maintenance"]["budget_renewal_interval"]
        assert type(self.budget_renewal_interval) == int

        # Traffic assignment parameters
        ta_conf = config["traffic"]["traffic_assignment"]
        self.traffic_assigmment_reuse_initial_volumes = ta_conf["reuse_initial_volumes"]
        self.traffic_assignment_initial_max_iterations = ta_conf[
            "initial_max_iterations"
        ]
        self.traffic_assignment_max_iterations = ta_conf["max_iterations"]
        self.traffic_assignment_convergence_threshold = ta_conf["convergence_threshold"]
        self.traffic_assignment_update_weight = ta_conf["update_weight"]

        self.travel_time_reward_factor = config["traffic"]["travel_time_reward_factor"]

        self.reset(reset_edges=False)

        self.base_traffic_factor = config["traffic"]["base_traffic_factor"]

        for edge in self.graph.es:
            edge["base_volume"] = self.base_traffic_factor * (
                min([seg.capacity for seg in edge["road_edge"].segments])
            )

        self.base_total_travel_time = self._get_total_travel_time(
            iterations=self.traffic_assignment_initial_max_iterations,
            set_initial_volumes=False,
        )
        self.initial_edge_volumes = np.array(self.graph.es["volume"])

    def reset(self, reset_edges=True):
        """ Resets the environment to the initial state.
        Args:
            reset_edges (bool): If True, the road segments will be reset.
        Returns:
            dict: The initial observation of the environment."""
        self.timestep = 0
        self.current_budget = self.budget_amount
        self.budget_constraint_applied = False
        if reset_edges:
            for edge in self.graph.es:
                edge["road_edge"].reset()
        return self._get_observation()

    def _get_observation(self):
        edge_observations = []
        edge_nodes = []
        edge_beliefs = []
        edge_deterioration_rates = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_edge"].get_observation())
            edge_deterioration_rates.append(edge["road_edge"].get_deterioration_rate())
            edge_beliefs.append(edge["road_edge"].get_beliefs())

        observations = {
            "edge_observations": edge_observations,
            "edge_deterioration_rates": edge_deterioration_rates,
            "edge_beliefs": edge_beliefs,
            "time_step": self.timestep,
            "budget_remaining": self.current_budget,
            "budget_time_until_renewal": self._get_budget_remaining_time(),
        }

        return observations

    def _get_states(self):
        edge_states = []
        for edge in self.graph.es:
            edge_states.append(edge["road_edge"].get_states())

        return edge_states

    def get_terminal_reward(self):
        """ Returns the total terminal reward for the environment in the current state."""

        total_terminal_reward = 0
        for edge in self.graph.es:
            for segment in edge["road_edge"].segments:
                total_terminal_reward += segment.get_terminal_reward()
        return total_terminal_reward

    def _get_total_travel_time(self, iterations=None, set_initial_volumes=False):
        if iterations is None:
            iterations = self.traffic_assignment_max_iterations

        if set_initial_volumes:
            self.graph.es["volume"] = self.initial_edge_volumes
        else:
            # Initialize with all-or-nothing assignment
            self.graph.es["volume"] = self.graph.es["base_volume"]

            self.graph.es["travel_time"] = [
                edge["road_edge"].compute_edge_travel_time(edge["volume"])
                for edge in self.graph.es
            ]

            for source, target, num_cars in self.trips:
                path = self.graph.get_shortest_paths(
                    source, target, weights="travel_time", output="epath"
                )[0]
                for edge_id in path:
                    self.graph.es[edge_id]["volume"] += num_cars

        for iteration in range(iterations):
            # Recalculate travel times with current volumes
            self.graph.es["travel_time"] = [
                edge["road_edge"].compute_edge_travel_time(edge["volume"])
                for edge in self.graph.es
            ]

            # Find the shortest paths using updated travel times
            new_volumes = np.array(self.graph.es["base_volume"])
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

        self.graph.es["travel_time"] = [
            edge["road_edge"].compute_edge_travel_time(edge["volume"])
            for edge in self.graph.es
        ]

        total_travel_time = np.sum(
            [edge["travel_time"] * edge["volume"] for edge in self.graph.es]
        )
        return total_travel_time

    def step(self, actions):
        """ Takes a step in the environment with the given actions.
        Args:
            actions (list): A list of lists containing the actions for each segment in the environment.
        
        Returns:
            tuple: A tuple containing the following elements:
                - observation (dict): The observation of the environment after the step.
                - reward (float): The reward received for the step.
                - done (bool): A boolean indicating if the episode is done.
                - info (dict): A dictionary containing additional information about the step.
        """
        actions = self._apply_action_constraints(actions)

        maintenance_reward = 0
        for i, edge in enumerate(self.graph.es):
            maintenance_reward += edge["road_edge"].step(actions[i])

        action_durations = []
        for edge in self.graph.es:
            for segment in edge["road_edge"].segments:
                action_durations.append(segment.action_duration)

        max_action_duration = max(action_durations)

        if max_action_duration > 0:
            worst_case_total_travel_time = self._get_total_travel_time(
                iterations=self.traffic_assignment_max_iterations,
                set_initial_volumes=self.traffic_assigmment_reuse_initial_volumes,
            )

            total_travel_time = (
                (1 - max_action_duration) * self.base_total_travel_time
                + max_action_duration * worst_case_total_travel_time
            )
        else:
            total_travel_time = self.base_total_travel_time

        travel_time_reward = self.travel_time_reward_factor * (
            total_travel_time - self.base_total_travel_time
        )

        reward = maintenance_reward + travel_time_reward

        # Update variables after step is complete for up to date observations

        self.timestep += 1

        if self.timestep % self.budget_renewal_interval == 0:
            self.current_budget = self.budget_amount

        observation = self._get_observation()

        done = self.timestep >= self.max_timesteps

        terminal_reward = self.get_terminal_reward() if done else 0
        reward += terminal_reward

        info = {
            "edge_states": self._get_states(),
            "total_travel_time": total_travel_time,
            "travel_times": self.graph.es["travel_time"],
            "traffic_volumes": self.graph.es["volume"],
            "reward_elements": {
                "travel_time_reward": travel_time_reward,
                "maintenance_reward": maintenance_reward,
                "terminal_reward": terminal_reward,
            },
            "budget_constraints_applied": self.budget_constraint_applied,
            "forced_replace_constraint_applied": self.forced_replace_constraint_applied,
            "applied_actions": actions,
        }

        return observation, reward, done, info

    def seed(self, seed):
        """ Seeds the random number generator of the environment.
        Args:
            seed (int): The seed value to use for seeding the random number generator.
        """
        self.random_generator = np.random.default_rng(seed)
        for edge in self.graph.es:
            edge["road_edge"].random_generator = self.random_generator
            for segment in edge["road_edge"].segments:
                segment.random_generator = self.random_generator

    def get_count_redundancies_summary(self, verbose: bool = True):
        """ Returns a string containing a summary of the redundancies in the environment.
        Args:
            verbose (bool): If True, the number of paths between each origin-destination pair will be printed.
        Returns:
            str: A string containing the summary of the redundancies.
        """
        vcount = self.graph.vcount()

        # number of paths between each origin-destination pair
        OD_num_paths = np.zeros((vcount, vcount))
        OD_matrix = np.zeros((vcount, vcount))

        string = ""

        string += "Summary | Network Trips\n"
        string += "=" * 23 + "\n\n"

        string += f"Total number of trips: {len(self.trips)}\n\n"

        for origin, destination, _ in self.trips:
            paths = self.graph.get_all_simple_paths(origin, destination)

            if verbose:
                string += f"O: {origin}, D: {destination} | # paths: {len(paths)}\n"

            OD_num_paths[origin, destination] = len(paths)
            OD_matrix[origin, destination] = 1

        string += "\n"
        string += "Summary | Network Redundancy\n"
        string += "=" * 28 + "\n\n"

        assert (OD_num_paths - OD_matrix >= 0).all(), "Some paths are missing"

        # count the number of redundancies
        # example: 'x' trips have 0 redundancies, 'y' trips have 1 redundancy, etc.
        _redundancy = OD_num_paths - OD_matrix
        redundancy_count = {}
        nonzero_indices = np.nonzero(OD_matrix)
        # loop through the non-zero elements of the OD matrix
        for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
            n_redundancy = int(_redundancy[i, j])
            if n_redundancy not in redundancy_count:
                redundancy_count[n_redundancy] = 1
            else:
                redundancy_count[n_redundancy] += 1

        # print the redundancy count
        for k, v in redundancy_count.items():
            string += f"{v} trips have {k} redundancies\n"
        return string

    def get_edge_traffic_summary(self):
        """ Returns a string containing a summary of the edge traffic in the environment.
        Returns:
            str: A string containing the summary of the edge traffic.
        """
        string = ""
        string += "Summary | Edge Traffic\n"
        string += "=" * 22 + "\n\n"
        string += f"{'Edge':^5} {'Volume (%)':^15} {'Travel Time':^5}\n"
        string += "-" * 30 + "\n"

        for edge in self.graph.es:
            id = edge.index
            volume = edge["volume"]
            travel_time = edge["travel_time"]
            capacity = sum([seg.capacity for seg in edge["road_edge"].segments])
            usage = volume / capacity * 100
            string += f"{id:^5} {int(volume):^5}({usage:^3.1f}%) {travel_time:^15.2f}\n"

        return string
    
    def get_topology_info(self):
        """ Returns a dictionary containing topology related information of the environment.
        Returns:
            dict: A dictionary containing the following keys:
                - adjacency_matrix (np.array): The adjacency matrix of the graph.
                - graph (igraph.Graph): A graph object with the same topology as the internal graph.
                - number_of_vertices (int): The number of vertices in the graph.
                - number_of_edges (int): The number of edges in the graph.
                - edges_origin_destination (list): A list of lists containing the source and target vertices of each edge.
                - segments_per_edge (list): A list containing the number of segments in each edge.
        """
        adjacency_matrix = np.array(self.graph.get_adjacency().data)

        
        edges_origin_destination = []
        segments_per_edge = []
        for edge in self.graph.es:
            segments_per_edge.append(len(edge["road_edge"].segments))
            edges_origin_destination.append([edge.source, edge.target])

        graph = ig.Graph(edges_origin_destination, directed=True)

        topology_info = {
            "adjacency_matrix": adjacency_matrix,
            "graph": graph,
            "number_of_vertices": graph.vcount(),
            "number_of_edges": graph.ecount(),
            "edges_origin_destination": edges_origin_destination,
            "segments_per_edge": segments_per_edge,
        }

        return topology_info
    
    def get_dimension_info(self):
        """ Returns a dictionary containing information about dimensions of the environment.
        
        Returns:
            dict: A dictionary containing the following keys:
                - actions (int): The number of actions available for each segment in the environment.
                - states (int): The number of states available for each segment in the environment.
                - observations (int): The number of observations available for each segment in the environment.
        """
        observation_tables = self.graph.es[0]["road_edge"].segments[0].observation_tables

        dimension_info = {
            "actions": observation_tables.shape[0],
            "states": observation_tables.shape[1],
            "observations": observation_tables.shape[2],
        }

        return dimension_info

    def _get_budget_remaining_time(self):
        return (
            self.budget_renewal_interval - self.timestep % self.budget_renewal_interval
        )

    def get_action_cost(self, actions):
        """ Returns the total cost of the given actions in the current state.
        Args:
            actions (list): A list of lists containing the actions for each segment in the environment.
            
        Returns:
            float: The total cost of the given actions in the current state.
        """
        total_cost = 0
        for i, edge in enumerate(self.graph.es):
            segments = edge["road_edge"].segments
            edge_actions = actions[i]
            for j, segment in enumerate(segments):
                segment_action = edge_actions[j]
                cost = -segment.get_action_reward(action=segment_action)
                total_cost += cost
        return total_cost

    def _apply_action_constraints(self, actions):
        actions = [action.copy() for action in actions]

        actions = self._apply_forced_repair_constraint(actions)
        actions = self._apply_budget_constraint(actions)

        return actions

    def _apply_forced_repair_constraint(self, actions):
        # Corrective replace action if the worst condition is observed
        self.forced_replace_constraint_applied = False
        for i, edge in enumerate(self.graph.es):
            for j, segment in enumerate(edge["road_edge"].segments):
                if segment.observation == segment.number_of_states - 1:
                    segment.worst_observation_counter += 1
                    if (
                        segment.worst_observation_counter
                        > self.forced_replace_worst_observation_count
                    ):
                        self.forced_replace_constraint_applied = True
                        segment.forced_repair = True
                        actions[i][j] = 4
                else:
                    segment.worst_observation_counter = 0
        return actions

    def _apply_budget_constraint(self, actions):
        """
        When actions cannot be taken due to budget constraints, we will
        set the actions to 0 (do-nothing). However, the cost associated
        with do-nothing is non-zero, which we will refer to as fallback cost.
        We will require those to be paid upfront for the budget cycle.
        When an action other than do-nothing is taken, the cost of
        that action will be adjusted to account for the fallback costs paid upfront.
        """
        self.budget_constraint_applied = False

        # Collect costs for each action
        edge_indices = []
        segment_indices = []
        adjusted_costs = []
        total_upfront_cost = 0  # total minimum cost for all segments
        total_future_upfront_cost = (
            0  # total minimum cost for all segments in the future
        )
        total_adjusted_cost = 0  # total cost (after adjusting for upfront costs)
        for i, edge in enumerate(self.graph.es):
            segments = edge["road_edge"].segments
            edge_actions = actions[i]

            for j, segment in enumerate(segments):
                edge_indices.append(i)
                segment_indices.append(j)
                if segment.forced_repair:  # forced repairs are not part of the budget
                    upfront_cost = 0
                    action_cost = 0
                else:
                    action = edge_actions[j]
                    upfront_cost = -segment.get_action_reward(action=0)
                    action_cost = -segment.get_action_reward(action=action)

                future_upfront_cost = -segment.get_action_reward(action=0)

                adjusted_cost = action_cost - upfront_cost

                total_upfront_cost += upfront_cost
                total_future_upfront_cost += future_upfront_cost
                total_adjusted_cost += adjusted_cost
                adjusted_costs.append(adjusted_cost)

        remaining_budget = (
            self.current_budget
            - total_upfront_cost
            - total_future_upfront_cost * (self._get_budget_remaining_time() - 1)
        )

        assert remaining_budget >= 0, "Remaining budget is negative"

        # if we do not have enough budget to take all actions,
        # we prioritize actions and select a random possible set of actions
        # that satisfies the budget
        if total_adjusted_cost > remaining_budget:

            self.budget_constraint_applied = True

            edge_indices = np.array(edge_indices)
            segment_indices = np.array(segment_indices)
            adjusted_costs = np.array(adjusted_costs)

            # Shuffle the costs to randomly select valid actions
            indices = np.arange(len(adjusted_costs))
            self.random_generator.shuffle(indices)

            shuffled_costs = adjusted_costs[indices]
            cumulative_costs = np.cumsum(shuffled_costs)

            # Find the index where the cumulative costs exceed the budget
            cutoff_index = np.searchsorted(
                cumulative_costs, remaining_budget, side="right"
            )

            # Set the actions that cannot be taken to 0
            zero_indices = indices[cutoff_index:]
            adjusted_costs[zero_indices] = 0
            for idx in zero_indices:
                if (
                    not self.graph.es[edge_indices[idx]]["road_edge"]
                    .segments[segment_indices[idx]]
                    .forced_repair
                ):
                    actions[edge_indices[idx]][segment_indices[idx]] = 0

        self.current_budget -= total_upfront_cost + np.sum(adjusted_costs)

        return actions
