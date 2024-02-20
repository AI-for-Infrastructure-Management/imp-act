import itertools

import jax
import jax.numpy as jnp
import pytest

from environments.jax_environment import EnvState, JaxRoadEnvironment
from igraph import Graph
from wrappers.jax_env_wrapper import JaxRoadEnvironmentWrapper


def test_total_base_travel_time(toy_environment_numpy, toy_environment_jax):
    _, _ = toy_environment_jax.reset_env()
    _jax = toy_environment_jax.total_base_travel_time

    _numpy = toy_environment_numpy.base_total_travel_time

    assert _jax == _numpy


def test_shortest_path_computation(toy_environment_jax):
    """Test shortest path computation."""

    _num_vertices = 4
    edges_list = jnp.array(
        [(0, 1), (1, 2), (2, 3), (3, 0)]
    )

    weights_list = [2, 6, 5, 8]

    # create graph using igraph
    graph = Graph()
    graph.add_vertices(_num_vertices)
    graph.add_edges(edges_list)

    sources = [0, 1, 2, 3]
    targets = [0, 1, 2, 3]

    for source, target in itertools.product(sources, targets):

        print(f"source {source} target {target}")

        # Find shortest path using igraph
        shortest_path = graph.get_shortest_paths(
            source, target, weights=weights_list, mode="out", output="epath"
        )
        # get cost to travel from source to target using shortest path
        cost_1 = float(sum([weights_list[i] for i in shortest_path[0]]))
        print(cost_1)

        # get cost to travel from source to target
        weights_matrix = toy_environment_jax._get_weight_matrix(weights_list, edges_list, target)
        print(weights_matrix)
        cost_2 = toy_environment_jax._get_cost_to_go(weights_matrix, 100)[source]
        print(cost_2)

        assert cost_1 == cost_2


def test_get_travel_time(toy_environment_numpy, toy_environment_jax):
    "Test total travel time is the same for jax and numpy env"
    actions = [
        [1, 1] for edge in toy_environment_numpy.graph.es
    ]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = toy_environment_numpy.step(actions)
        dam_state = jnp.array(toy_environment_numpy._get_states()).flatten()
        belief = jnp.array(obs["edge_beliefs"]).flatten()
        total_travel_time_np = info["total_travel_time"]
        base_travel_times = []
        capacities = []
        for edge in toy_environment_numpy.graph.es["road_segments"]:
            for segment in edge.segments:
                base_travel_times.append(segment.base_travel_time)
                capacities.append(segment.capacity)
        jax_state = EnvState(
            damage_state=dam_state,
            observation=jnp.array(obs["edge_observations"]).flatten(),
            belief=belief,
            base_travel_time=jnp.asarray(base_travel_times),
            capacity=jnp.asarray(capacities),
            timestep=timestep,
        )
        total_travel_time_jax = toy_environment_jax._get_total_travel_time(jax_state)
        print(total_travel_time_np, total_travel_time_jax)
        assert total_travel_time_np.round() == total_travel_time_jax.round()


def test_jax_keys(toy_environment_jax):

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    # rollout
    key = jax.random.PRNGKey(442)
    step_keys, key = toy_environment_jax.split_key(key)

    # environment reset
    _, state = toy_environment_jax.reset_env()

    done = False

    while not done:
        _, _, done, _, state = toy_environment_jax.step_env(step_keys, state, action)

        # generate keys for next timestep
        step_keys, key = toy_environment_jax.split_key(key)

    _rollout_key = key

    # check if keys are the same
    assert (jnp.array([3808878501, 3829080728]) == _rollout_key).all()


def test_jax_wrapper_keys(toy_environment_jax_wrapper):
    jax_env = toy_environment_jax_wrapper
    step_keys, key = jax_env.step_keys, jax_env.key

    _ = jax_env.reset()
    done = False

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    while not done:
        _, _, done, _ = jax_env.step(action)
        step_keys_new, key_new = jax_env.step_keys, jax_env.key
        assert (step_keys != step_keys_new).all()
        assert (key != key_new).all()
        step_keys, key = step_keys_new, key_new


def test_belief_computation(toy_environment_jax, toy_environment_numpy):
    action_np = [
        [1, 1] for edge in toy_environment_numpy.graph.es
    ]
    action_jax = [{"0": [1, 1]}, {"1": [1, 1]}, {"2": [1, 1]}, {"3": [1, 1]}]
    action_jax = jax.tree_util.tree_leaves(action_jax)
    action_jax = jnp.array(action_jax, dtype=jnp.uint8)

    observation = toy_environment_numpy.reset()
    belief = jnp.array(observation["edge_beliefs"]).reshape(-1, 4)
    done = False

    while not done:
        observation, _, done, _ = toy_environment_numpy.step(action_np)
        obs = jnp.array(observation["edge_observations"]).flatten()
        print(jnp.array(observation["edge_beliefs"]), obs, action_jax)
        belief_jax = toy_environment_jax._get_next_belief(belief, obs, action_jax)
        belief = jnp.array(observation["edge_beliefs"]).reshape(-1, 4)

        assert jnp.allclose(belief, belief_jax, atol=1e-3)


def test_idxs_map(toy_environment_jax): # TODO: hardcoded for graph param, rewrite for toy_environment_jax
    compute_idxs_map = toy_environment_jax._compute_idxs_map()

    f = 1_000_000

    true_idxs_map = jnp.array(
        [
            [0, f, f, f],
            [1, 2, f, f],
            [3, 4, 5, f],
            [6, 7, 8, 9],
            [10, f, f, f],
            [11, 12, f, f],
            [13, 14, 15, f],
            [16, 17, 18, 19],
        ]
    )

    assert jnp.allclose(true_idxs_map, compute_idxs_map)


def test_gather(toy_environment_jax): # TODO: hardcoded for graph param, rewrite for toy_environment_jax

    edge_values = jnp.array(
        [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
        ]
    )

    computed_values = toy_environment_jax._gather(edge_values)

    true_values = jnp.array(
        [
            [101, 0.0, 0.0, 0.0],
            [102, 103, 0.0, 0.0],
            [104, 105, 106, 0.0],
            [107, 108, 109, 110],
            [111, 0.0, 0.0, 0.0],
            [112, 113, 0.0, 0.0],
            [114, 115, 116, 0.0],
            [117, 118, 119, 120],
        ]
    )

    assert jnp.allclose(true_values, computed_values)


def test_campaign_reward(toy_environment_jax):
    """Test campaign reward computation."""

    # inspections on 0 edges => campaign reward: 0
    _action = [{"0": [0, 2]}, {"1": [2, 0]}, {"2": [2, 2]}, {"3": [2, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    campaign_reward = toy_environment_jax._get_campaign_reward(action)

    assert campaign_reward == 0

    # inspections on 3 edges => campaign reward: -3 * 5 = -15
    _action = [{"0": [1, 2]}, {"1": [1, 1]}, {"2": [1, 0]}, {"3": [2, 3]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    campaign_reward = toy_environment_jax._get_campaign_reward(action)

    assert campaign_reward == -15
