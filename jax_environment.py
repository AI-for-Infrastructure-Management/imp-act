from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import vmap
from params import EnvParams


@struct.dataclass
class EnvState:
    # Properties of segments
    damage_state: jnp.array
    observation: jnp.array
    belief: jnp.array
    base_travel_time: jnp.array
    capacity: jnp.array
    timestep: int


class JaxRoadEnvironment(environment.Environment):
    """
    JAX implementation of the Road Environment.
    """

    def __init__(self, params: EnvParams):
        super().__init__()

        # Horizon parameters
        self.max_timesteps = params.max_timesteps

        # Reward parameters
        self.travel_time_reward_factor = params.travel_time_reward_factor

        # Graph parameters
        self.num_nodes = params.num_vertices
        self.edges = params.edges
        self.num_edges = params.num_edges
        self.edge_segments_numbers = params.edge_segments_numbers
        self.total_num_segments = int(jnp.sum(self.edge_segments_numbers))

        # Traffic assignment parameters
        self.shortest_path_max_iterations = params.shortest_path_max_iterations
        self.traffic_assignment_max_iterations = (
            params.traffic_assignment_max_iterations
        )
        self.traffic_assignment_convergence_threshold = (
            params.traffic_assignment_convergence_threshold
        )
        self.traffic_assignment_update_weight = params.traffic_assignment_update_weight
        self.traffic_alpha = params.traffic_alpha
        self.traffic_beta = params.traffic_beta

        # Road Network parameters
        self.trip_sources, self.trip_destinations = jnp.nonzero(params.trips)
        self.trips = params.trips

        self.btt_table = params.btt_table
        self.capacity_table = params.capacity_table

        # Damage parameters
        self.num_damage_states = params.num_dam_states
        self.initial_dam_state = params.initial_dam_state
        self.initial_obs = params.initial_obs
        self.initial_belief = params.initial_belief

        self.deterioration_table = params.deterioration_table
        self.observation_table = params.observation_table

        self.rewards_table = params.rewards_table

        self.idxs_map = self._compute_idxs_map()

        # reset
        _obs, state = self.reset_env()

        # total base travel time
        self.total_base_travel_time = self._get_total_travel_time(state)

    def _compute_idxs_map(self):
        """
        Compute the idxs_map used to gather the properties of all
        segments belonging to an edge.

        Returns
        -------
        idxs_map : np.array
            Matrix of indices. idxs_map[i,j] is the index of the jth
            segment belonging to the ith edge.
        """

        idxs_map = np.full((self.num_edges, self.edge_segments_numbers.max()), 1_0000)
        idx = 0
        for i, num_segments in enumerate(self.edge_segments_numbers):
            idxs_map[i, :num_segments] = np.arange(idx, idx + num_segments)

            idx += num_segments

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
    @partial(vmap, in_axes=(None, 0, 0, None))
    def _get_maintenance_reward(
        self, dam_state: int, action: int, rewards_table: jnp.array
    ) -> float:
        return rewards_table[dam_state, action]

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

        params : Environment parameters

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

    @partial(jax.jit, static_argnums=0)
    def _get_weight_matrix(
        self, weights: jnp.array, edges: jnp.array, destination: int
    ):
        """
        Get the weight matrix for the shortest path algorithm from all
        nodes to the given destination node.

        Parameters
        -------
        weights: Vector of weights (example: travel time) for each edge

        num_nodes: Number of nodes in the graph

        edges: Vector of tuples (example: [(0, 1), (1,3)]) representing the
            nodes of each edge.

        destination: Destination node

        Returns
        -------
        weights_matrix:
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf
        """

        weights_matrix = jnp.full((self.num_nodes, self.num_nodes), jnp.inf)
        # set destination node to 0
        weights_matrix = weights_matrix.at[destination, destination].set(0)

        # set weights (uses jax.lax.scatter behind the scenes)
        weights_matrix = weights_matrix.at[edges[:, 0], edges[:, 1]].set(weights)
        weights_matrix = weights_matrix.at[edges[:, 1], edges[:, 0]].set(weights)

        return weights_matrix

    @partial(jax.jit, static_argnums=(0, 2))
    def _get_cost_to_go(self, weights_matrix: jnp.array, max_iter: int):
        """
        Get the cost-to-go from all nodes to the given destination node.

        Parameters
        ----------
        weights_matrix :
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf

        num_nodes : Number of nodes in the graph
        max_iter : Maximum number of iterations for the while loop

        Returns
        -------
        J : Vector of cost-to-go from all nodes to the given destination
            node. J[i] is the cost from node i to the destination node.
        """

        J = jnp.zeros(self.num_nodes)  # Initial guess

        def body_fun(values):
            # Define the body function of while loop
            i, J, _ = values

            # Update J and break condition
            next_J = jnp.min(weights_matrix + J, axis=1)
            break_condition = jnp.allclose(next_J, J, 0.01, 0.01)

            # Return next iteration values
            return i + 1, next_J, break_condition

        def cond_fun(values):
            i, _, break_condition = values
            return ~break_condition & (i < max_iter)

        return jax.lax.while_loop(cond_fun, body_fun, init_val=(0, J, False))[1]

    @partial(jax.jit, static_argnums=0)
    def _get_volumes(
        self,
        source: int,
        destination: int,
        weights_matrix: jnp.array,
        J: jnp.array,
    ):
        """
        Get a vector containing the volume of each edge in the shortest
        path from the given source to the given destination.

        Access this function through _get_volumes_shortest_path

        Parameters
        ----------
        source : Source node

        destination : Destination node

        weights_matrix : jnp.array
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf

        J : Vector of cost-to-go from all nodes to the given destination
            node. J[i] is the cost from node i to the destination node.

        params : Environment parameters

        """
        edges = self.edges
        trips = self.trips
        num_edges = len(edges)
        # this need to be float32 since we multiply by
        # params.traffic_assignment_update_weight (float) in the traffic
        # assignment loop
        volumes = jnp.full((num_edges,), 0.0)

        def body_fun(val):
            step, current_node, volumes, _ = val

            # TODO: ties: usually returns the first index, should we randomize?
            next_node = jnp.argmin(weights_matrix[current_node, :] + J)

            # find edge index given current_node and next_node
            edge_index = (
                jnp.asarray(edges == jnp.array([current_node, next_node]))
                .all(axis=1)
                .nonzero(size=1)[0]
            )
            trip = trips[source][destination]
            volumes = volumes.at[edge_index].set(volumes[edge_index] + trip)
            return step + 1, next_node, volumes, destination != next_node

        def cond_fun(val):  # we might want to add a max_depth condition if too slow
            _, _, _, break_cond = val
            return break_cond

        return jax.lax.while_loop(
            cond_fun, body_fun, init_val=(0, source, volumes, True)
        )[2]

    @partial(jax.jit, static_argnums=0)
    @partial(vmap, in_axes=(None, 0, 0, None))
    def _get_volumes_shortest_path(
        self, source: int, destination: int, weights: jnp.array
    ):
        """
        Compute the volumes of each edge in the shortest path from the
        given source to the given destination.

        source: https://jax.quantecon.org/short_path.html

        Parameters
        ----------
        source : source node

        destination : destination node

        weights : Vector of weights (example: travel time) for each edge

        params : Environment parameters

        Returns
        -------
        volumes : Vector of volumes of each edge in the shortest path
                  from the given source to the given destination.
                  shape: (num_edges,)
        """

        edges = self.edges

        weight_matrix = self._get_weight_matrix(weights, edges, destination)
        cost_to_go = self._get_cost_to_go(
            weight_matrix, max_iter=self.shortest_path_max_iterations
        )
        volumes = self._get_volumes(source, destination, weight_matrix, cost_to_go)
        return volumes

    @partial(jax.jit, static_argnums=0)
    def _get_total_travel_time(self, state):
        """Get the total travel time of all cars in the network."""

        # 0.1 Initialize volumes
        edge_volumes = jnp.full((len(self.edges)), 0)

        # 0.2 Calculate initial travel times
        edge_travel_times = self.compute_edge_travel_time(state, edge_volumes)

        # 0.3 Find the shortest paths using all-or-nothing assignment
        edge_volumes = self._get_volumes_shortest_path(
            self.trip_sources, self.trip_destinations, edge_travel_times
        ).sum(axis=0)

        # repeat until convergence
        def body_fun(val):
            edge_volumes, _ = val

            # 1. Recalculate travel times with current volumes
            edge_travel_times = self.compute_edge_travel_time(state, edge_volumes)

            # 2. Find the shortest paths using updated travel times
            #    (recalculates edge volumes)
            new_edge_volumes = self._get_volumes_shortest_path(
                self.trip_sources, self.trip_destinations, edge_travel_times
            ).sum(axis=0)

            # 3. Check for convergence by comparing volume changes
            volume_changes = jnp.abs(edge_volumes - new_edge_volumes)
            max_volume_change = jnp.max(volume_changes)

            # 4. Update edge volumes
            edge_volumes = (
                self.traffic_assignment_update_weight * new_edge_volumes
                + (1 - self.traffic_assignment_update_weight) * edge_volumes
            )

            return edge_volumes, max_volume_change

        def cond_fun(val):
            _, max_volume_change = val
            return max_volume_change > self.traffic_assignment_convergence_threshold

        edge_volumes, _ = jax.lax.while_loop(
            cond_fun, body_fun, init_val=(edge_volumes, jnp.inf)
        )

        edge_travel_times = self.compute_edge_travel_time(state, edge_volumes)

        # 5. Calculate total travel time
        return jnp.sum(edge_travel_times * edge_volumes)

    @partial(vmap, in_axes=(None, 0, 0, 0))
    def _get_next_belief(self, belief: jnp.array, obs: int, action: int) -> jnp.array:
        """Update belief for a single segment."""

        next_belief = self.deterioration_table[action].T @ belief
        state_probs = self.observation_table[action][:, obs]
        next_belief = state_probs * next_belief
        next_belief /= next_belief.sum()
        return next_belief

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

        # next state
        next_state = self._get_next(
            keys_transition, state.damage_state, action, self.deterioration_table
        )

        # observation
        obs = self._get_next(keys_obs, next_state, action, self.observation_table)

        # maintenance reward
        maintenance_reward = self._get_maintenance_reward(
            state.damage_state, action, self.rewards_table
        ).sum()

        # belief update
        belief = self._get_next_belief(state.belief, obs, action)

        base_travel_time = self.btt_table[action, state.damage_state]
        capacity = self.capacity_table[action, state.damage_state]

        total_travel_time = self._get_total_travel_time(state)
        travel_time_reward = self.travel_time_reward_factor * (
            total_travel_time - self.total_base_travel_time
        )

        # reward
        reward = maintenance_reward + travel_time_reward

        next_state = EnvState(
            damage_state=next_state,
            observation=obs,
            belief=belief,
            base_travel_time=base_travel_time,
            capacity=capacity,
            timestep=state.timestep + 1,
        )

        # done
        done = self.is_terminal(next_state)

        # info
        info = {
            "total_travel_time": total_travel_time,
            "maintenance_reward": maintenance_reward,
            "travel_time_reward": travel_time_reward,
        }

        return obs, reward, done, info, next_state

    @partial(jax.jit, static_argnums=0)
    def reset_env(self) -> Tuple[chex.Array, EnvState]:
        # initial damage state
        damage_state = jnp.zeros(self.total_num_segments, dtype=jnp.uint8)

        # initial belief
        belief = jnp.array([self.initial_belief] * self.total_num_segments)

        # initial base travel times
        initial_btt = jnp.ones(self.total_num_segments) * self.btt_table[0, 0]
        # initial capacity
        initial_capacity = jnp.ones(self.total_num_segments) * self.capacity_table[0, 0]

        env_state = EnvState(
            damage_state=damage_state,
            observation=damage_state,
            belief=belief,
            base_travel_time=initial_btt,
            capacity=initial_capacity,
            timestep=0,
        )
        return self.get_obs(env_state), env_state

    @partial(jax.jit, static_argnums=0)
    def _gather(self, x: jnp.array):
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

        return jnp.take(x, self.idxs_map, fill_value=0.0)

    def get_obs(self, state: EnvState) -> chex.Array:
        return state.observation

    def is_terminal(self, state: EnvState) -> bool:
        return state.timestep >= self.max_timesteps

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

        return subkeys, key


if __name__ == "__main__":
    params = EnvParams()
    env = JaxRoadEnvironment(params)

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    key = jax.random.PRNGKey(442)
    step_keys, key = env.split_key(key)

    # reset
    obs, state = env.reset_env()

    total_rewards = 0.0

    # rollout
    for _ in range(params.max_timesteps):
        # step
        obs, reward, done, info, state = env.step_env(step_keys, state, action)

        # generate keys for next timestep
        step_keys, key = env.split_key(key)

        # update total rewards
        total_rewards += reward

    print(f"Total rewards: {total_rewards}")
    print(f"Total travel time: {info['total_travel_time']}")
