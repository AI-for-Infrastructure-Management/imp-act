from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import vmap
from params import EnvParams
from functools import cached_property


@struct.dataclass
class EnvState:
    damage_state: jnp.array
    observation: jnp.array
    belief: jnp.array
    base_travel_time: jnp.array
    capacity: jnp.array
    timestep: int


class RoadEnvironment(environment.Environment):

    """

    JAX implementation of the Road Environment.

    """

    def __init__(self):
        super().__init__()

    def _get_next(
        self, key: chex.PRNGKey, dam_state: int, action: int, table: jnp.array
    ) -> int:
        # sample
        next_dam_state = jax.random.choice(key, 4, p=table[action, dam_state])

        return next_dam_state

    def _vmap_get_next(self):
        # get next state or observation for all segments
        return vmap(self._get_next, in_axes=(0, 0, 0, None))

    def _get_maintenance_reward(
        self, dam_state: int, action: int, rewards_table: jnp.array
    ) -> float:
        return rewards_table[action, dam_state]

    def _vmap_get_maintenance_reward(self):
        return vmap(self._get_maintenance_reward, in_axes=(0, 0, None))

    def compute_edge_travel_time(
        self, state: EnvState, edge_volumes: jnp.array, params: EnvParams
    ):
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
        state : EnvState
            Environment state

        edge_volumes : jnp.array
            Vector of volumes of each edge in the graph.
            shape: (num_edges)

        params : EnvParams
            Environment parameters

        Returns
        -------
        edge_travel_times : jnp.array
            Vector of travel times of each edge in the graph.
            shape: (num_edges, 1)
        """

        # gather base travel time and capacity of each edge
        btt_factor = self._gather(state.base_travel_time).sum(axis=1)

        # compute capacity factor for each segment
        capacity_factor = (
            state.base_travel_time
            * params.traffic_alpha
            / (state.capacity) ** params.traffic_beta
        )
        # gather capacity factor of each edge
        capacity_factor = self._gather(capacity_factor).sum(axis=1)

        # calculate travel time on each edge
        return btt_factor + capacity_factor * edge_volumes**params.traffic_beta

    def _get_weight_matrix(self, weights, num_nodes, edges, destination):
        """
        Get the weight matrix for the shortest path algorithm from all
        nodes to the given destination node.

        Parameters
        -------
        weights: jnp.array
            Vector of weights (example: travel time) for each edge

        num_nodes: int
            Number of nodes in the graph

        edges: jnp.array
            Vector of tuples (example: [(0, 1), (1,3)]) representing the
            nodes of each edge.

        destination: int
            Destination node

        Returns
        -------
        weights_matrix: jnp.array
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf
        """

        weights_matrix = jnp.full((num_nodes, num_nodes), jnp.inf)

        # set destination node to 0
        weights_matrix = weights_matrix.at[destination, destination].set(0)

        # TODO: can we do this without a for loop? or jax.fori_loop?
        for edge, w in zip(edges, weights):
            i, j = edge  # node indices
            # undirected graph, so we need to set both directions
            weights_matrix = weights_matrix.at[i, j].set(w)
            weights_matrix = weights_matrix.at[j, i].set(w)

        return weights_matrix

    def _get_cost_to_go(self, weights_matrix, num_nodes, max_iter):
        """
        Get the cost-to-go from all nodes to the given destination node.

        Parameters
        ----------
        weights_matrix : jnp.array
            Matrix of weights (example: travel time) between each pair
            of nodes. The weights_matrix[i,j] is the weight of the edge
            from node i to node j. If there is no edge between node i and
            node j, then weights_matrix[i,j] = jnp.inf

        num_nodes : int
            Number of nodes in the graph
        max_iter : int
            Maximum number of iterations for the while loop

        Returns
        -------
        J : jnp.array
            Vector of cost-to-go from all nodes to the given destination
            node. J[i] is the cost from node i to the destination node.
        """

        J = jnp.zeros(num_nodes)  # Initial guess

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

    def _get_volumes(self, source, destination, weights_matrix, J, params):
        """
        Get a vector containing the volume of each edge in the shortest
        path from the given source to the given destination.

        Access this function through _get_volumes_shortest_path

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

        J : jnp.array
            Vector of cost-to-go from all nodes to the given destination
            node. J[i] is the cost from node i to the destination node.

        params : EnvParams
            Environment parameters

        """
        edges = params.edges
        trips = params.trips
        num_edges = len(edges)
        volumes = jnp.full((num_edges,), 0)

        def body_fun(val):
            step, current_node, volumes, _ = val
            # TODO: ties: usually returns the first index, should we randomize?
            next_node = jnp.argmin(weights_matrix[current_node, :] + J)
            # path = path.at[step+1].set(next_node)
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

    # def print_best_path(self, source, destination, weights_matrix, J, params):

    #     current_node = source
    #     edges = params.edges
    #     trips = params.trips
    #     num_edges = len(edges)
    #     volumes = jnp.full((num_edges,), 0)

    #     while current_node != destination:
    #         next_node = jnp.argmin(weights_matrix[current_node, :] + J)

    #         edge_index = (
    #             jnp.asarray(edges == jnp.array([current_node, next_node]))
    #             .all(axis=1)
    #             .nonzero(size=1)[0]
    #         )

    #         trip = trips[source][destination]

    #         print(f"{current_node} -> {next_node} ({edge_index}) ({trip})")

    #         volumes = volumes.at[edge_index].set(volumes[edge_index] + trip)

    #         print(volumes)

    #         current_node = next_node
    #     return volumes

    def _get_volumes_shortest_path(self, source, destination, weights, params):
        """

        Compute the volumes of each edge in the shortest path from the
        given source to the given destination.

        source: https://jax.quantecon.org/short_path.html

        Parameters
        ----------
        source : int
            source node
        destination : int
            destination node
        weights : jnp.array
            Vector of weights (example: travel time) for each edge in the
            graph.
        params : EnvParams
            Environment parameters

        Returns
        -------
        volumes : jnp.array
            Vector of volumes of each edge in the shortest path from the
            given source to the given destination.
            shape: (num_edges,)
        """

        num_nodes = params.num_vertices
        edges = params.edges

        weight_matrix = self._get_weight_matrix(weights, num_nodes, edges, destination)
        cost_to_go = self._get_cost_to_go(weight_matrix, num_nodes, max_iter=500)
        volumes = self._get_volumes(
            source, destination, weight_matrix, cost_to_go, params
        )
        return volumes

    def _vmap_get_volumes_shortest_path(self):
        return vmap(self._get_volumes_shortest_path, in_axes=(0, 0, None, None))

    # @cached_property
    def _get_initial_volumes(self, params):
        """
        Get the initial volumes of each edge in the graph with
        all-or-nothing assignment. @cached_property is used to ensure
        that this function is only run the first time. Subsequent calls
        will return the cached result.

        Parameters
        ----------
        params : EnvParams
            Environment parameters

        Returns
        -------
        initial_volumes : jnp.array
            Vector of initial volumes of each edge in the graph.
            shape: (num_edges,)
        """

        # TODO: what if there are no direct paths from source to destination?
        num_edges = len(params.edges)
        edge_volumes = jnp.full((num_edges,), 0)
        for k, (i, j) in enumerate(params.edges):
            num_cars = params.trips[i, j]
            edge_volumes = edge_volumes.at[k].set(num_cars)

        return edge_volumes

    def _get_total_travel_time(self, state, params):
        # 0.1 get initial edge volumes
        edge_volumes = self._get_initial_volumes(params)

        _sources, _destinations = jnp.nonzero(params.trips, size=4)

        # repeat until convergence
        for _ in range(params.traffic_assignment_max_iterations):
            # 1. Recalculate travel times with current volumes
            edge_travel_times = self.compute_edge_travel_time(
                state, edge_volumes, params
            )

            # 2. Find the shortest paths using updated travel times
            #    (recalculates edge volumes)
            new_edge_volumes = self._vmap_get_volumes_shortest_path()(
                _sources, _destinations, edge_travel_times, params
            ).sum(axis=0)

            # 3. Check for convergence by comparing volume changes
            volume_changes = jnp.abs(edge_volumes - new_edge_volumes)
            max_volume_change = jnp.max(volume_changes)

            if max_volume_change < params.traffic_assignment_convergence_threshold:
                break

            # 4. Update edge volumes
            edge_volumes = (
                params.traffic_assignment_update_weight * new_edge_volumes
                + (1 - params.traffic_assignment_update_weight) * edge_volumes
            )

        # 5. Calculate total travel time
        return jnp.sum(edge_travel_times * edge_volumes)

    def _get_next_belief(
        self, belief: jnp.array, obs: int, action: int, params: EnvParams
    ) -> jnp.array:
        next_belief = params.deterioration_table[action].T @ belief
        state_probs = params.observation_table[action][:, obs]
        next_belief = state_probs * next_belief
        next_belief /= next_belief.sum()
        return next_belief

    def _vmap_get_next_belief(self):
        return vmap(self._get_next_belief, in_axes=(0, 0, 0, None))

    def step_env(
        self, keys: chex.PRNGKey, state: EnvState, action: jnp.array, params: EnvParams
    ) -> Tuple[chex.Array, list, float, bool, dict]:
        # next state
        next_state = self._vmap_get_next()(
            keys, state.damage_state, action, params.deterioration_table
        )

        # observation
        obs = self._vmap_get_next()(keys, next_state, action, params.observation_table)

        # maintenance reward
        maintenance_reward = self._vmap_get_maintenance_reward()(
            state.damage_state, action, params.rewards_table
        ).sum()

        # belief update
        belief = self._vmap_get_next_belief()(state.belief, obs, action, params)

        base_travel_time = params.btt_table[action, state.damage_state]
        capacity = params.capacity_table[action, state.damage_state]

        total_travel_time = self._get_total_travel_time(state, params)
        # total_travel_time = 0.0
        travel_time_reward = params.travel_time_reward_factor * total_travel_time

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
        done = self.is_terminal(next_state, params)

        # info
        info = {
            "total_travel_time": total_travel_time,
            "maintenance_reward": maintenance_reward,
            "travel_time_reward": travel_time_reward,
        }

        return obs, reward, done, info, next_state

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        # initial damage state
        damage_state = jnp.zeros(params.total_num_segments, dtype=jnp.uint8)

        # initial belief
        belief = jnp.array([params.initial_belief] * params.total_num_segments)

        # initial base travel times (using pytree)
        initial_btt = jnp.ones(params.total_num_segments) * params.btt_table[0, 0]
        # initial capacity
        initial_capacity = (
            jnp.ones(params.total_num_segments) * params.capacity_table[0, 0]
        )

        env_state = EnvState(
            damage_state=damage_state,
            observation=damage_state,
            belief=belief,
            base_travel_time=initial_btt,
            capacity=initial_capacity,
            timestep=0,
        )

        return self.get_obs(env_state), env_state

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

        idxs_map = jnp.array(
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ]
        )

        return jnp.take(x, idxs_map, fill_value=0.0)

    def get_obs(self, state: EnvState) -> chex.Array:
        return state.observation

    def is_terminal(self, state: EnvParams, params: EnvParams) -> bool:
        return state.timestep >= params.max_timesteps

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        pass

    def state_space(self, params: EnvParams):
        pass

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        pass

    def belief_space(self, params: EnvParams):
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def num_actions(self) -> int:
        pass


if __name__ == "__main__":
    params = EnvParams()
    env = RoadEnvironment()

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    key = jax.random.PRNGKey(442)
    keys = jax.random.split(key, params.total_num_segments + 1)  # keys for all segments
    subkeys = keys[: params.total_num_segments, :]  # subkeys for each segment
    keys = keys[params.total_num_segments, :]  # key for next timestep

    # reset
    obs, state = env.reset_env(key, params)

    total_rewards = 0.0

    # rollout
    for _ in range(params.max_timesteps):
        # step
        obs, reward, done, info, state = env.step_env(subkeys, state, action, params)

        # generate keys for next timestep
        # keys for all segments
        keys = jax.random.split(
            keys,
            params.total_num_segments + 1,
        )
        subkeys = keys[: params.total_num_segments, :]  # subkeys for each segment
        keys = keys[params.total_num_segments, :]  # subkeys for each segment

        # update total rewards
        total_rewards += reward

    print(f"Total rewards: {total_rewards}")
