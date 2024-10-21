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
        self.number_of_states = config["maintenance"]["deterioration"].shape[-1]
        self.initial_damage_prob = config["maintenance"]["initial_damage_distribution"]
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

        self.forced_repair = False

        self.reset()

    def reset(self):
        self.get_initial_state()

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
        else:
            reward = self.state_action_reward[action][self.state]

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

        return reward

    def get_initial_state(self):
        # Computing initial state, observation, and belief
        self.deterioration_rate = 0
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
                segments.append(
                    RoadSegment(
                        random_generator=self.random_generator,
                        position_x=segment["position_x"],
                        position_y=segment["position_y"],
                        capacity=segment["capacity"],
                        base_travel_time=segment["travel_time"],
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
            graph_edge["road_segments"] = road_edge

        # Budget parameters
        self.budget_amount = config["maintenance"]["budget_amount"]
        assert type(self.budget_amount) in [int, float]
        self.budget_renewal_interval = config["maintenance"]["budget_renewal_interval"]
        assert type(self.budget_renewal_interval) == int

        # Traffic assignment parameters
        ta_conf = config["traffic"]["traffic_assignment"]
        self.traffic_assignment_max_iterations = ta_conf["max_iterations"]
        self.traffic_assignment_convergence_threshold = ta_conf["convergence_threshold"]
        self.traffic_assignment_update_weight = ta_conf["update_weight"]

        self.travel_time_reward_factor = config["traffic"]["travel_time_reward_factor"]
        self.replacement_cost = config["maintenance"]["reward"]["state_action_reward"][-1][-1]

        self.reset(reset_edges=False)

        self.base_total_travel_time = self._get_total_travel_time()

    def reset(self, reset_edges=True):
        self.timestep = 0
        self.current_budget = self.budget_amount
        self.budget_constraint_applied = False
        if reset_edges:
            for edge in self.graph.es:
                edge["road_segments"].reset()
        return self._get_observation()

    def _get_observation(self):
        adjacency_matrix = np.array(self.graph.get_adjacency().data)
        edge_observations = []
        edge_nodes = []
        edge_beliefs = []
        edge_deterioration_rates = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_segments"].get_observation())
            edge_deterioration_rates.append(
                edge["road_segments"].get_deterioration_rate()
            )
            edge_beliefs.append(edge["road_segments"].get_beliefs())
            edge_nodes.append([edge.source, edge.target])

        observations = {
            "adjacency_matrix": adjacency_matrix,
            "edge_observations": edge_observations,
            "edge_deterioration_rates": edge_deterioration_rates,
            "edge_beliefs": edge_beliefs,
            "edge_nodes": edge_nodes,
            "time_step": self.timestep,
            "budget_remaining": self.current_budget,
            "budget_time_until_renewal": self._get_budget_remaining_time(),
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

        self.graph.es["travel_time"] = [
            edge["road_segments"].compute_edge_travel_time(edge["volume"])
            for edge in self.graph.es
        ]
        return np.sum([edge["travel_time"] * edge["volume"] for edge in self.graph.es])

    def step(self, actions):
        actions = self._apply_action_constraints(actions)

        maintenance_reward = 0
        for i, edge in enumerate(self.graph.es):
            maintenance_reward += edge["road_segments"].step(actions[i])

        self.current_budget += maintenance_reward

        total_travel_time = self._get_total_travel_time()

        travel_time_reward = self.travel_time_reward_factor * (
            total_travel_time - self.base_total_travel_time
        )

        reward = maintenance_reward + travel_time_reward

        # Update variables after step is complete for up to date observations

        self.timestep += 1

        if self.timestep % self.budget_renewal_interval == 0:
            self.current_budget = self.budget_amount

        observation = self._get_observation()

        info = {
            "states": self._get_states(),
            "total_travel_time": total_travel_time,
            "travel_times": self.graph.es["travel_time"],
            "volumes": self.graph.es["volume"],
            "reward_elements": [travel_time_reward, maintenance_reward],
            "budget_constraints_applied": self.budget_constraint_applied,
        }

        return observation, reward, self.timestep >= self.max_timesteps, info

    def seed(self, seed):
        self.random_generator = np.random.default_rng(seed)
        for edge in self.graph.es:
            edge["road_segments"].random_generator = self.random_generator
            for segment in edge["road_segments"].segments:
                segment.random_generator = self.random_generator

    def get_count_redundancies_summary(self, verbose: bool = True):
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
        string = ""
        string += "Summary | Edge Traffic\n"
        string += "=" * 22 + "\n\n"
        string += f"{'Edge':^5} {'Volume (%)':^15} {'Travel Time':^5}\n"
        string += "-" * 30 + "\n"

        for edge in self.graph.es:
            id = edge.index
            volume = edge["volume"]
            travel_time = edge["travel_time"]
            capacity = sum([seg.capacity for seg in edge["road_segments"].segments])
            usage = volume / capacity * 100
            string += f"{id:^5} {int(volume):^5}({usage:^3.1f}%) {travel_time:^15.2f}\n"

        return string

    def _get_budget_remaining_time(self):
        return (
            self.budget_renewal_interval - self.timestep % self.budget_renewal_interval
        )

    def get_action_cost(self, actions):
        total_cost = 0
        for i, edge in enumerate(self.graph.es):
            segments = edge["road_segments"].segments
            edge_actions = actions[i]
            for j, segment in enumerate(segments):
                segment_action = edge_actions[j]
                cost = -segment.state_action_reward[segment_action][segment.state]
                total_cost += cost
        return total_cost

    def _apply_action_constraints(self, actions):
        actions = [action.copy() for action in actions]

        actions = self._apply_budget_constraint(actions)
        actions = self._apply_forced_repair_constraint(actions)
        return actions

    def _apply_forced_repair_constraint(self, actions):
        # Corrective replace action if the worst condition is observed
        for i, edge in enumerate(self.graph.es):
            for j, segment in enumerate(edge["road_segments"].segments):
                if segment.observation == segment.number_of_states - 1:
                    actions[i][j] = 4
                    self.current_budget -= self.replacement_cost
                    segment.forced_repair = True
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
        total_upfront_cost = 0  # total cost of do-nothing action for all segments
        total_adjusted_cost = 0  # total cost (after adjusting for upfront costs)
        for i, edge in enumerate(self.graph.es):
            segments = edge["road_segments"].segments
            edge_actions = actions[i]

            for j, segment in enumerate(segments):
                action = edge_actions[j]
                edge_indices.append(i)
                segment_indices.append(j)
                upfront_cost = -segment.state_action_reward[0, segment.state]
                action_cost = -segment.state_action_reward[action][segment.state]
                adjusted_cost = action_cost - upfront_cost

                total_upfront_cost += upfront_cost
                total_adjusted_cost += adjusted_cost
                adjusted_costs.append(adjusted_cost)

        remaining_budget = (
            self.current_budget - total_upfront_cost * self._get_budget_remaining_time()
        )

        assert remaining_budget >= 0, "Remaining budget is negative"

        # if we do not have enough budget to take all actions,
        # we prioritize actions and select the largest set of actions
        # that satisfy the budget
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
            for idx in zero_indices:
                actions[edge_indices[idx]][segment_indices[idx]] = 0

        return actions
