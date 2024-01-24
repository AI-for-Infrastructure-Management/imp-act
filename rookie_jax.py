from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import vmap
from jax.tree_util import tree_flatten, tree_unflatten
from params import EnvParams


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

    def calculate_bpr_travel_time(
        volume: int, capacity: int, base_time: float, alpha: float, beta: int
    ):
        return base_time * (1 + alpha * (volume / capacity) ** beta)

    def _vmap_calculate_bpr_travel_time(self):
        return vmap(self.calculate_bpr_travel_time, in_axes=(0, 0, 0, None, None))

    def compute_edge_base_travel_time(self, state: EnvState):
        # map segments to edges, gather base travel times and sum over segments
        return self._gather(state.base_travel_time).sum(axis=0)

    def compute_edge_travel_time(
        self, state: EnvState, edge_volumes: jnp.array, params: EnvParams
    ):
        # get edge base travel times
        edge_base_travel_time = self.compute_edge_base_travel_time(state)

        # get edge travel times
        edge_travel_times = self._vmap_calculate_bpr_travel_time()(
            edge_volumes,
            state.capacity,
            edge_base_travel_time,
            params.traffic_alpha,
            params.traffic_beta,
        )

        return edge_travel_times

    def _get_shortest_paths(self, state, action, params):
        #! Cannot use igraph in JAX if we want to jit this function
        # update edge volumes
        pass

    def _get_total_travel_time(self, state, action, params):
        # 0.1 get edge volumes: Initialize with all-or-nothing assignment

        # 0.2 get edge travel times

        # repeat until convergence

        # 1. Recalculate travel times with current volumes

        # 2. Find the shortest paths using updated travel times
        #    (recalculates edge volumes)

        # 3. Check for convergence by comparing volume changes

        return 0.0
    
    def _get_next_belief(
            self, belief: jnp.array, obs: int, action: int, params: EnvParams
    ) -> jnp.array:
        next_belief = params.deterioration_table[action].T @ belief
        state_probs = params.observation_table[action][:, obs]
        next_belief = state_probs*next_belief
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

        # TODO: traffic assignment (returning 0.0 for now)
        total_travel_time = self._get_total_travel_time(state, action, params)
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
        damage_state = [ # TODO: what is this exactly?
            {"0": [0, 1]},
            {"1": [0, 3, 0]},
            {"2": [1, 1, 2]},
            {"3": [3, 1, 2]},
        ]

        # flatten pytree and convert to jnp.array
        damage_state = jnp.array(
            jax.tree_util.tree_leaves(damage_state), dtype=jnp.uint8
        )

        # initial belief
        belief = jnp.array(
            [params.initial_belief]*params.total_num_segments
        )

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

    def _to_pytree(self, x: jnp.array):
        # example pytree
        py_tree = [ # TODO: what is this exactly?
            {"0": np.array([0, 1], dtype=np.uint8)},
            {"1": np.array([0, 3, 0], dtype=np.uint8)},
            {"2": np.array([1, 1, 2], dtype=np.uint8)},
            {"3": np.array([3, 1, 2], dtype=np.uint8)},
        ]

        # flatten pytree
        _, treedef = tree_flatten(py_tree)

        # put x into pytree
        py_tree = tree_unflatten(treedef, x.tolist())

        return py_tree

    def _gather(self, x: jnp.array):
        # map segment indices to edge indices
        # and pad gathered values with 0.0
        # to make get equal number of columns
        # TODO: precompute idxs_map

        # map of segment indices to segment ids
        idxs_map = jnp.array( # TODO: what is this exactly?
            [
                [100, 0, 1],
                [2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
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

    _action = [{"0": [0, 0]}, {"1": [0, 0, 0]}, {"2": [0, 0, 0]}, {"3": [0, 0, 0]}]
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
        keys = jax.random.split(
            keys, params.total_num_segments * 2 # TODO: we actually only need (total_num_segments+1) keys since we can use the last one for new splits. Check if the current implementation take too long for large graphs and evantually change to this other implementation
        )  # keys for all segments
        subkeys = keys[: params.total_num_segments, :]  # subkeys for each segment
        keys = keys[params.total_num_segments, :]  # subkeys for each segment

        # update total rewards
        total_rewards += reward

    print(f"Total rewards: {total_rewards}")

    ################ Speed test #################
    # Check speed of step_env with and without jit
    # run it to check if jit is working properly
    # after making changes(takes ~30s)
    # check speed to run 1000 steps
    import timeit

    number, repeat = 1000, 3

    jit_step_env = jax.jit(env.step_env)

    # TODO: I do not think that the following timeit test with jit is reliable 

    # python documentation suggests using min
    # step w/o jit (best of 3)
    output_nj = min(
        timeit.repeat(
            "env.step_env(subkeys, state, action, params)",
            number=number,
            repeat=repeat,
            globals=globals(),
        )
    )
    print(f"Non-jit per step: {output_nj / number:.3e}s")

    # step w/ jit (best of 3)
    output_j = min(
        timeit.repeat(
            "jit_step_env(subkeys, state, action, params)",
            number=number,
            repeat=repeat,
            globals=globals(),
        )
    )
    print(f"Jit per step: {output_j / number:.3e}s")

    print(f"Speedup: {output_nj / output_j:.2f}x")
