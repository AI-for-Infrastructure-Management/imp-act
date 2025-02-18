from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import vmap

# jax.config.update("jax_disable_jit", True)


@struct.dataclass
class EnvState:
    # Properties of segments
    damage_state: jnp.array
    observation: jnp.array
    belief: jnp.array
    base_travel_time: jnp.array
    capacity: jnp.array
    worst_obs_counter: jnp.array
    deterioration_rate: jnp.array
    timestep: int
    episode_return: float = 0.0


MILES_PER_KILOMETER = 0.621371
KILOMETERS_PER_MILE = 1.0 / MILES_PER_KILOMETER


class JaxRoadEnvironment(environment.Environment):
    """
    JAX implementation of the Road Environment.
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Episode time horizon
        self.max_timesteps = config["maintenance"]["max_timesteps"]

        ## 1) Network modeling

        # 1.1) Topology
        self.graph = config["topology"]["graph"]
        self.num_nodes, self.num_edges = self.graph.vcount(), self.graph.ecount()
        self.edges = jnp.array(self.graph.get_edgelist())
        # Adjacency matrix (with edge indices)
        # A1: vertex pairs contain edge index ("id"), 0 otherwise
        # A2: if no edge between vertex pairs, value is -1, 0 otherwise
        A1 = np.array(self.graph.get_adjacency(attribute="id"))
        A2 = np.array(self.graph.get_adjacency(eids=True))
        self.adjacency_matrix = jnp.array(A1 + A2, dtype=jnp.int32)

        # 1.2) Road Segments
        (
            self.segments_list,
            self.initial_btts,
            self.initial_capacities,
            self.segment_lengths,
            self.total_num_segments,
        ) = self._extract_segments_info(config)
        self.idxs_map = self._compute_idxs_map(self.segments_list)

        ## 2) Traffic modeling
        self.travel_time_reward_factor = config["traffic"]["travel_time_reward_factor"]

        # 2.1) BPR function parameters, capacity, and base travel times
        self.traffic_alpha = config["traffic"]["bpr_alpha"]
        self.traffic_beta = config["traffic"]["bpr_beta"]
        self.base_traffic_factor = config["traffic"]["base_traffic_factor"]
        # fractions of the initial capacities, multiplied by the initial capacities later
        self.capacity_table = jnp.array(config["traffic"]["capacity_factors"])
        # fractions of the initial btts, multiplied by the initial btts later
        self.btt_table = jnp.array(config["traffic"]["base_travel_time_factors"])

        # 2.2) Network traffic
        self.trips = self._extract_trip_info(config)
        self.trip_sources, self.trip_destinations = jnp.nonzero(self.trips)

        # 2.3) Traffic assignment
        ta_conf = config["traffic"]["traffic_assignment"]
        self.traffic_assigmment_reuse_initial_volumes = ta_conf[
            "reuse_initial_volumes"
        ]  # TODO: implement if needed, for speedups
        if self.traffic_assigmment_reuse_initial_volumes:
            print(
                "Warning: Reusing initial volumes in traffic assignment is not implemented yet."
            )
        self.traffic_assignment_initial_max_iterations = ta_conf[
            "initial_max_iterations"
        ]
        self.traffic_assignment_max_iterations = ta_conf["max_iterations"]
        self.traffic_assignment_convergence_threshold = ta_conf["convergence_threshold"]
        self.traffic_assignment_update_weight = ta_conf["update_weight"]

        ## 3) Inspection and maintenance modeling
        imp_conf = config["maintenance"]
        self.action_map = {
            "do-nothing": 0,
            "inspect": 1,
            "minor-repair": 2,
            "major-repair": 3,
            "replace": 4,
        }

        # 3.1) Damage states and observations
        self.initial_damage_prob = jnp.array(imp_conf["initial_damage_distribution"])
        self.num_damage_states = imp_conf["deterioration"].shape[-1]
        self.num_observations = imp_conf["observation"].shape[-1]
        self.forced_replace_worst_observation_count = imp_conf[
            "forced_replace_worst_observation_count"
        ]

        # 3.2) Action space
        # action durations (shape: A)
        self.action_durations = jnp.array(imp_conf["action_duration_factors"])

        # 3.3) Deterioration and observation models
        # Deterioration model (shape: A x S x S or A x DR x S x S)
        self.deterioration_table = jnp.array(imp_conf["deterioration"])
        self.deterioration_rate_enabled = self.deterioration_table.ndim == 4
        if self.deterioration_rate_enabled:
            self.deterioration_rate_max = self.deterioration_table.shape[1]
        # Observation model (shape: A x S x O)
        self.observation_table = jnp.array(imp_conf["observation"])

        # 3.4) Budget and rewards
        self.inspection_campaign_reward = imp_conf["reward"][
            "inspection_campaign_reward"
        ]
        # rewards_table (shape: S x A)
        self.rewards_table = jnp.array(imp_conf["reward"]["state_action_reward"])
        # terminal state rewards (shape: S)
        self.terminal_state_reward = jnp.array(
            imp_conf["reward"]["terminal_state_reward"]
        )
        # Budget parameters
        self.budget_amount = float(imp_conf["budget_amount"])
        self.budget_renewal_interval = imp_conf["budget_renewal_interval"]

        ## Environment properties
        key = jax.random.PRNGKey(9898)  # dummy key, doesn't matter
        _, state = self.reset(key)

        # Base total travel time
        self.base_volumes = (
            self._gather(state.capacity, fill_value=jnp.inf).min(axis=1)
            * self.base_traffic_factor
        )
        self.base_total_travel_time = self._get_total_travel_time(
            state, self.traffic_assignment_initial_max_iterations
        )

    def _extract_segments_info(self, config: Dict):
        """Extract segments information from the configuration file.
        (Only used in the constructor)
        """

        # get all edge ids from the graph
        igraph_edge_ids = self.graph.es["id"]

        total_num_segments = 0
        for edge_segments in config["topology"]["segments"].values():
            total_num_segments += len(edge_segments)

        # idxs_list: list of segment indices for each edge
        segments_idxs_list = [[] for _ in range(self.num_edges)]
        segment_initial_btt = np.empty(total_num_segments)
        segment_initial_capacity = np.empty(total_num_segments)
        segment_lengths = np.empty(total_num_segments)
        idx = 0
        for nodes, edge_segments in config["topology"]["segments"].items():
            _indices = []
            for segment in edge_segments:
                # get edge index from graph using nodes
                edge_id = self.graph.get_eid(
                    self.graph.vs.find(id=nodes[0]).index,
                    self.graph.vs.find(id=nodes[1]).index,
                )
                # get equivalent JAX edge index
                edge_id = igraph_edge_ids.index(edge_id)
                _indices.append(idx)

                # segment length
                if segment.get("segment_length") is None:
                    _segment_length = KILOMETERS_PER_MILE
                else:
                    _segment_length = segment["segment_length"]

                # travel time
                if segment.get("travel_time") is None:
                    base_travel_time = (
                        segment["segment_length"] / segment["travel_speed"]
                    )
                else:
                    base_travel_time = segment["travel_time"]

                segment_initial_capacity[idx] = segment["capacity"]
                segment_initial_btt[idx] = base_travel_time
                segment_lengths[idx] = _segment_length

                idx += 1

            segments_idxs_list[edge_id] = _indices

        return (
            segments_idxs_list,
            jnp.array(segment_initial_btt),
            jnp.array(segment_initial_capacity),
            jnp.array(segment_lengths),
            total_num_segments,
        )

    def _extract_trip_info(self, config: Dict):
        """
        Store trip information from the config file into the trips matrix.
        trips[i, j] is the volume of trips from node i to node j.
        (Only used in the constructor)
        """

        trips = np.zeros((self.num_nodes, self.num_nodes))

        trips_df = config["traffic"]["trips"]
        for index in trips_df.index:

            vertex_1_list = self.graph.vs.select(id_eq=trips_df["origin"][index])
            vertex_2_list = self.graph.vs.select(id_eq=trips_df["destination"][index])

            if (len(vertex_1_list) == 0) or (len(vertex_2_list) == 0):
                raise ValueError(
                    f"Trip not in graph: {trips_df['origin'][index]} -> {trips_df['destination'][index]}"
                )

            vertex_1 = vertex_1_list[0].index
            vertex_2 = vertex_2_list[0].index

            trips[vertex_1, vertex_2] = trips_df["volume"][index]

        return jnp.array(trips)

    def _compute_idxs_map(self, segments_list):
        """
        Compute the idxs_map used to gather the properties of all
        segments belonging to an edge.

        Parameters
        ----------
        segments_list : list
            List of segments indices belonging to each edge.

        Returns
        -------
        idxs_map : np.array
            Matrix of indices. idxs_map[i,j] is the index of the jth
            segment belonging to the ith edge.
        """

        # fill value to ensure that values greater than the total number
        # of segments return 0 when used as indices in self._gather
        _fill = 100_000

        assert (
            _fill > self.total_num_segments
        ), "Fill value must be greater than the total number of segments."

        # maximum number of segments per edge
        max_length = max(len(segment) for segment in segments_list)

        # create a np.array and pad with _fill
        idxs_map = np.full((self.num_edges, max_length), _fill)
        for i, segment in enumerate(segments_list):
            idxs_map[i, : len(segment)] = segment

        return idxs_map

    @partial(jax.jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, 0, 0, 0, None))
    def _get_next(
        self, key: chex.PRNGKey, dam_state: int, action: int, table: jnp.array
    ) -> int:
        """
        Abstract method to get the next state/observation given the
        current state, action, and transition/observation table.
        """
        # sample
        next_dam_state = jax.random.choice(
            key, self.num_damage_states, p=table[action, dam_state]
        )

        return next_dam_state

    @partial(jax.jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _get_next_damage_state(self, key, dam_state, action, det_rate):

        # sample
        next_dam_state = jax.random.choice(
            key,
            self.num_damage_states,
            p=self.deterioration_table[action, det_rate, dam_state],
        )
        return next_dam_state

    @partial(jax.jit, static_argnums=(0,))
    @partial(vmap, in_axes=(None, 0, 0, None))
    def _get_rewards_from_table(
        self, dam_state: int, action: int, rewards_table: jnp.array
    ) -> float:
        return rewards_table[dam_state, action]

    @partial(jax.jit, static_argnums=(0,))
    def _get_campaign_reward(self, action: jnp.array) -> float:

        _gathered = self._gather(action)

        # check which segments were inspected,
        # return 1 if at least one segment was inspected, 0 otherwise
        # max: avoid 2 in case of multiple inspections on the same edge
        # sum: total number of inspections
        total_num_inspections = jnp.where(_gathered == 1, 1, 0).max(axis=1).sum()

        return total_num_inspections * self.inspection_campaign_reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_maintenance_reward(
        self, damage_state: jnp.array, action: jnp.array
    ) -> float:

        maintenance_reward = self._get_rewards_from_table(
            damage_state, action, self.rewards_table
        ).sum()

        campaign_reward = self._get_campaign_reward(action)

        maintenance_reward += campaign_reward

        return maintenance_reward

    @partial(jax.jit, static_argnums=0)
    def compute_edge_travel_time(self, state: EnvState, edge_volumes: jnp.array):
        """
        Compute the travel time of each edge given the current volumes
        of each edge.

        BPR function:
        travel_time = btt * (1 + alpha * (v / c) ** beta)

        btt: base travel time
        alpha, beta: parameters
        v: volume of cars on the edge
        c: capacity of the edge

        Since we have edge volumes, we aggregate the btt and capacity
        of all segments belonging to an edge and multiply capacity by
        the edge volume.

        Parameters
        ----------
        state : Environment state

        edge_volumes : Vector of volumes of each edge
                       shape: (num_edges)

        Returns
        -------
        edge_travel_times : Vector of travel times of each edge
                            shape: (num_edges, 1)
        """

        # gather base travel time and capacity of each edge
        btt_factor = self._gather(state.base_travel_time).sum(axis=1)

        # compute capacity factor for each segment
        capacity_factor = (
            state.base_travel_time
            * self.traffic_alpha
            / (state.capacity) ** self.traffic_beta
        )
        # gather capacity factor of each edge
        capacity_factor = self._gather(capacity_factor).sum(axis=1)

        # calculate travel time on each edge
        return btt_factor + capacity_factor * edge_volumes**self.traffic_beta

    @partial(jax.jit, static_argnums=(0,))
    def _get_weight_matrix(self, weights: jnp.array):
        """
        Get the weight matrix for the shortest path algorithm from all
        nodes to the given destination node.

        Parameters
        -------
        weights: Vector of weights (example: travel time) for each edge

        Returns
        -------
        weights_matrix:
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf
        """
        edges = self.edges
        weights_matrix = jnp.full((self.num_nodes, self.num_nodes), jnp.inf)

        # set diagonal to 0
        weights_matrix = weights_matrix.at[
            jnp.arange(self.num_nodes), jnp.arange(self.num_nodes)
        ].set(0)

        # set weights (uses jax.lax.scatter behind the scenes)
        weights_matrix = weights_matrix.at[edges[:, 0], edges[:, 1]].set(weights)

        return weights_matrix

    @partial(jax.jit, static_argnums=(0,))
    def _get_cost_to_go(self, weights_matrix: jnp.array):
        """
        Get the cost-to-go from all nodes to all nodes using the
        Floyd-Warshall algorithm.

        Parameters
        ----------
        weights_matrix :
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf
            The diagonal of the matrix is set to 0.

        Returns
        -------
        cost_to_go_matrix : Matrix of cost-to-go from all nodes to all nodes
        """

        cost_to_go_matrix = jnp.copy(weights_matrix)

        def body_fun(k, cost_to_go_matrix):
            # Dynamically slice the k-th column and row
            kth_col = jax.lax.dynamic_slice(
                cost_to_go_matrix, (0, k), (self.num_nodes, 1)
            )  # Slice the k-th column
            kth_row = jax.lax.dynamic_slice(
                cost_to_go_matrix, (k, 0), (1, self.num_nodes)
            )  # Slice the k-th row

            # Update dist using the current intermediate vertex k
            cost_to_go_matrix = jnp.minimum(cost_to_go_matrix, kth_col + kth_row)
            return cost_to_go_matrix

        return jax.lax.fori_loop(
            0, self.num_nodes, body_fun, init_val=(cost_to_go_matrix)
        )

    @partial(jax.jit, static_argnums=0)
    @partial(vmap, in_axes=(None, 0, 0, None, None))
    def _get_volumes(
        self,
        source: int,
        destination: int,
        weights_matrix: jnp.array,
        cost_to_go_matrix: jnp.array,
    ):
        """
        Get a vector containing the volume of each edge in the shortest
        path from the given source to the given destination.

        Parameters
        ----------
        source : Source node

        destination : Destination node

        weights_matrix : jnp.array
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf

        cost_to_go_matrix : jnp.array
            Matrix of cost-to-go from all nodes to all nodes. The cost_to_go_matrix[i,j]
            is the cost-to-go from node i to node j using the shortest available path.

        """

        trip = self.trips[source][destination]

        volumes = jnp.zeros(self.num_edges)

        def body_fun(val):
            step, current_node, volumes, _ = val

            # reminder: returns the first index
            next_node = jnp.argmin(
                weights_matrix[current_node, :] + cost_to_go_matrix[:, destination]
            )
            # find edge index given current_node and next_node
            edge_index = self.adjacency_matrix[current_node, next_node]
            volumes = volumes.at[edge_index].set(volumes[edge_index] + trip)
            return step + 1, next_node, volumes, destination != next_node

        def cond_fun(val):
            _, _, _, break_cond = val
            return break_cond

        return jax.lax.while_loop(
            cond_fun, body_fun, init_val=(0, source, volumes, True)
        )[2]

    @partial(jax.jit, static_argnums=0)
    def _get_volumes_shortest_path(self, weights: jnp.array, base_volumes: jnp.array):
        """
        Compute the volumes of each edge in the shortest path from the
        given source to the given destination.

        source: https://jax.quantecon.org/short_path.html

        Parameters
        ----------
        weights : Vector of weights (example: travel time) for each edge

        Returns
        -------
        volumes : Vector of volumes of each edge in the shortest path
                  from the given source to the given destination.
                  shape: (trips, num_edges)
        """

        weight_matrix = self._get_weight_matrix(weights)
        cost_to_go_matrix = self._get_cost_to_go(weight_matrix)

        # for volumes set diagonal to jnp.inf
        weight_matrix = weight_matrix.at[
            jnp.arange(self.num_nodes), jnp.arange(self.num_nodes)
        ].set(jnp.inf)

        trips_volumes = self._get_volumes(
            self.trip_sources,
            self.trip_destinations,
            weight_matrix,
            cost_to_go_matrix,
        ).sum(axis=0)

        return base_volumes + trips_volumes

    @partial(jax.jit, static_argnums=0)
    def _get_total_travel_time(self, state, max_iterations):
        """
        Get the total travel time of all trucks in the network.
        We assume that there is a base volume of traffic from cars,
        We only consider the travel time for trucks (trips) in the network.

        """

        # 0.1 Initialize volumes
        # base_volumes: base traffic (such as cars, always on the road)
        edge_volumes = self.base_volumes

        # 0.2 Calculate initial travel times
        edge_travel_times = self.compute_edge_travel_time(state, edge_volumes)

        # 0.3 Find the shortest paths using all-or-nothing assignment
        edge_volumes = self._get_volumes_shortest_path(
            edge_travel_times, self.base_volumes
        )

        # repeat until convergence
        def body_fun(val):
            edge_volumes, _, i, _ = val

            # 1. Recalculate travel times with current volumes
            edge_travel_times = self.compute_edge_travel_time(state, edge_volumes)

            # 2. Find the shortest paths using updated travel times
            #    (recalculates edge volumes)
            new_edge_volumes = self._get_volumes_shortest_path(
                edge_travel_times, self.base_volumes
            )

            # 3. Check for convergence by comparing volume changes
            volume_changes = jnp.abs(edge_volumes - new_edge_volumes)
            max_volume_change = jnp.max(volume_changes)

            # 4. Update edge volumes
            edge_volumes = (
                self.traffic_assignment_update_weight * new_edge_volumes
                + (1 - self.traffic_assignment_update_weight) * edge_volumes
            )

            return edge_volumes, max_volume_change, i + 1, edge_travel_times

        def cond_fun(val):
            _, max_volume_change, i, _ = val
            return (
                max_volume_change > self.traffic_assignment_convergence_threshold
            ) & (i < max_iterations)

        edge_volumes, _, _, edge_travel_times = jax.lax.while_loop(
            cond_fun, body_fun, init_val=(edge_volumes, jnp.inf, 0, edge_travel_times)
        )

        # 5. Calculate total travel time
        return jnp.sum(edge_travel_times * edge_volumes)

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _get_next_belief(
        self, belief: jnp.array, obs: int, action: int, det_rate: int
    ) -> jnp.array:
        """Update belief for a single segment."""

        next_belief = self.deterioration_table[action, det_rate].T @ belief
        state_probs = self.observation_table[action][:, obs]
        next_belief = state_probs * next_belief
        next_belief /= next_belief.sum()
        return next_belief

    @partial(jax.jit, static_argnums=0)
    @partial(vmap, in_axes=(None, 0, 0))
    def _get_next_deterioration_rate(self, action: int, det_rate: int) -> int:
        det_rate = jax.lax.cond(
            action == self.action_map["replace"],
            lambda x: 0,
            lambda x: jnp.minimum(x + 1, self.deterioration_rate_max),
            det_rate,
        )
        return det_rate

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self,
        keys: chex.PRNGKey,
        state: EnvState,
        action: jnp.array,
    ) -> Tuple[chex.Array, list, float, bool, dict]:
        """Move the environment one timestep forward."""

        # split keys into keys for damage transitions and observations
        keys_transition, keys_obs = jnp.split(keys, 2, axis=0)


        ## Maintenance modeling
        damage_state = self._get_next_damage_state(
            keys_transition,
            state.damage_state,
            action,
            state.deterioration_rate,
        )

        # deterioration rate update
        deterioration_rate = self._get_next_deterioration_rate(
            action, state.deterioration_rate
        )

        # observation
        obs = self._get_next(keys_obs, damage_state, action, self.observation_table)

        # maintenance reward
        maintenance_reward = self._get_maintenance_reward(state.damage_state, action)

        # belief update
        belief = self._get_next_belief(
            state.belief, obs, action, state.deterioration_rate
        )

        ## Traffic modeling
        base_travel_time = self.btt_table[action] * self.initial_btts
        capacity = self.capacity_table[action] * self.initial_capacities
        )

        total_travel_time = self._get_total_travel_time(state)
        travel_time_reward = self.travel_time_reward_factor * (
            total_travel_time - self.base_total_travel_time
        )

        # reward
        reward = maintenance_reward + travel_time_reward

        # done
        timestep = state.timestep + 1
        done = self.is_terminal(timestep)


        # returns
        returns = state.episode_return + reward

        # info
        info = {
            "total_travel_time": total_travel_time,
            "reward_elements": {
                "travel_time_reward": travel_time_reward,
                "maintenance_reward": maintenance_reward,
                "terminal_reward": terminal_reward,
            },
            "returns": returns,
        }

        next_state = EnvState(
            damage_state=damage_state,
            observation=obs,
            belief=belief,
            base_travel_time=base_travel_time,
            capacity=capacity,
            deterioration_rate=deterioration_rate,
            timestep=state.timestep + 1,
            episode_return=returns * jnp.logical_not(done),
        )

        return obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):

        key, subkeys = self.split_key(key)

        # environment step
        obs_st, state_st, reward, done, info = self.step_env(subkeys, state, action)

        # reset env
        key, sub_key = jax.random.split(key)
        obs_re, state_re = self.reset(sub_key)

        # Auto-reset environment based on done
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )

        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def reset(self, key) -> Tuple[chex.Array, EnvState]:
        # initial damage state
        key, subkey = jax.random.split(key)
        damage_state = jax.random.choice(
            subkey,
            self.num_damage_states,
            p=self.initial_damage_prob,
            shape=(self.total_num_segments,),
        )

        # initial observation
        keys = jax.random.split(key, self.total_num_segments + 1)
        key, subkeys = keys[0], keys[1:]
        _actions = jnp.zeros(self.total_num_segments, dtype=jnp.int32)
        obs = self._get_next(subkeys, damage_state, _actions, self.observation_table)

        # initial belief
        belief = jnp.array([self.initial_damage_prob] * self.total_num_segments)

        # worst observation counter (for forced repair)
        worst_observation_counter = jnp.zeros(self.total_num_segments, dtype=jnp.int32)

        # deterioration rate
        deterioration_rate = jnp.zeros(self.total_num_segments, dtype=jnp.int32)

        env_state = EnvState(
            damage_state=damage_state,
            observation=obs,
            belief=belief,
            base_travel_time=self.initial_btts,
            capacity=self.initial_capacities,
            worst_obs_counter=worst_observation_counter,
            deterioration_rate=deterioration_rate,
            timestep=0,
        )
        return self.get_obs(env_state), env_state

    @partial(jax.jit, static_argnums=(0, 2))
    def _gather(self, x: jnp.array, fill_value: float = 0.0) -> jnp.array:
        """
        Gather the properties (example: base travel time) of all
        segments belonging to an edge.

        We gather the properties using idxs_map to map segment indices
        to edge indices. We pad the gathered values with 0.0 to make get
        equal number of columns using jnp.take.

        Parameters
        ----------
        x : jnp.array
            Array of properties of all segments.
            shape: (num_segments)

        Returns
        -------
        x_gathered : jnp.array
            Array of properties of all edges.
            shape: (max_num_segments_per_edge, num_edges)
        """

        return jnp.take(x, self.idxs_map, fill_value=fill_value)

    def get_obs(self, state: EnvState) -> chex.Array:
        return state.observation

    def is_terminal(self, timestep: int) -> bool:
        return timestep >= self.max_timesteps

    def action_space(self) -> spaces.Discrete:
        pass

    def state_space(self):
        pass

    def observation_space(self) -> spaces.Discrete:
        pass

    def belief_space(self):
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def num_actions(self) -> int:
        return self.total_num_segments

    @partial(jax.jit, static_argnums=(0))
    def split_key(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, chex.PRNGKey]:
        """
        #! The rule of thumb is: never reuse keys
        (unless you want identical outputs)

        Split key into keys for each random variable:

        - keys for damage transitions of each segment (#segments)
        - keys for observations of each segment (#segments)
        - key for next timestep (1)

        """

        keys = jax.random.split(key, self.total_num_segments * 2 + 1)
        subkeys = keys[: self.total_num_segments * 2, :]
        key = keys[-1, :]

        return key, subkeys

    def _get_shortest_path(
        self,
        source: int,
        destination: int,
        weights_matrix: jnp.array,
        cost_to_go_matrix: jnp.array,
    ) -> list:
        """
        #! only used in tests (cannot jit since output can have variable size)
        Get the shortest path from the source to the destination.

        Parameters
        ----------
        source : int
            Source node

        destination : int
            Destination node

        weights_matrix : jnp.array
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf

        cost_to_go_matrix : jnp.array
            Matrix of cost-to-go from all nodes to all nodes.
            The cost_to_go_matrix[i,j] is the cost-to-go from node i to
            node j using the shortest available path.

        Returns
        -------
        shortest_path : List edge ids of the shortest path from the
                        source to the destination.
        """

        current_node = source
        edges_path = []

        while current_node != destination:
            next_node = jnp.argmin(
                weights_matrix[current_node, :] + cost_to_go_matrix[:, destination]
            )
            edge_index = self.adjacency_matrix[current_node, next_node]
            edges_path.append(edge_index)
            current_node = next_node

        return edges_path
