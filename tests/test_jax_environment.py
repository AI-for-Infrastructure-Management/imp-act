import itertools

import jax
import jax.numpy as jnp
import numpy as np

import pytest
from igraph import Graph

from imp_act.environments.jax_environment import EnvState

environment_fixtures_jax = [
    "toy_environment_2_jax",
    "cologne_environment_jax",
]


def test_total_base_travel_time_toy(toy_environment_2, toy_environment_2_jax):
    """Test if total base travel time is the same for jax and numpy env"""
    assert jnp.allclose(
        toy_environment_2.base_total_travel_time,
        toy_environment_2_jax.base_total_travel_time,
        rtol=1e-2,
    )


@pytest.mark.skip(reason="Old implementation references.")
def test_compute_edge_travel_time(cologne_environment, cologne_environment_jax):
    """Compare edge travel time computation for different volumes."""

    numpy_env = cologne_environment
    jax_env = cologne_environment_jax

    for volume in [0, 100, 250]:

        # Numpy environment
        numpy_env.graph.es["volume"] = volume
        numpy_edge_travel_times = [
            edge["road_segments"].compute_edge_travel_time(edge["volume"])
            for edge in numpy_env.graph.es
        ]
        numpy_edge_travel_times = jnp.array(numpy_edge_travel_times)

        # Jax environment
        _, state = jax_env.reset_env()
        edge_volumes = jnp.full((jax_env.num_edges), volume)
        edge_travel_times = jax_env.compute_edge_travel_time(state, edge_volumes)

        assert jnp.allclose(edge_travel_times, numpy_edge_travel_times, rtol=1e-2)


@pytest.mark.skip(reason="Old implementation references.")
def test_shortest_path_computation_toy(toy_environment_2_jax):
    """Test shortest path computation."""

    _num_vertices = 4
    edges_list = jnp.array([(0, 1), (1, 2), (2, 3), (3, 0)])

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

        # get cost to travel from source to target
        weights_matrix = toy_environment_2_jax._get_weight_matrix(weights_list)
        cost_2 = toy_environment_2_jax._get_cost_to_go(weights_matrix)[source][target]

        assert cost_1 == cost_2


def test_trips_initialization(cologne_environment, cologne_environment_jax):
    """Test trips initialization."""

    jax_env = cologne_environment_jax
    numpy_env = cologne_environment

    for source, target, num_cars in numpy_env.trips:
        assert jax_env.trips[source, target] == num_cars


@pytest.mark.skip(reason="Old implementation references.")
def test_shortest_path_cost_equivalence_large(
    cologne_environment,
    cologne_environment_jax,
):
    """
    When computing the cost to travel from source to target using the
    shortest path, are the costs the same for jax and numpy env?
    In general, the shortest path is not unique, so we compare the cost.
    """

    jax_env = cologne_environment_jax
    numpy_env = cologne_environment

    trips = numpy_env.trips

    # Numpy environment
    numpy_env.graph.es["volume"] = 0
    numpy_env.graph.es["travel_time"] = [
        edge["road_segments"].compute_edge_travel_time(edge["volume"])
        for edge in numpy_env.graph.es
    ]

    # Jax environment
    _, state = jax_env.reset_env()
    edge_volumes = jnp.full((jax_env.num_edges), 0)
    edge_travel_times = jax_env.compute_edge_travel_time(state, edge_volumes)

    for source, target, _ in trips:

        shortest_path = numpy_env.graph.get_shortest_paths(
            source, target, weights="travel_time", mode="out", output="epath"
        )[0]

        # get cost to travel from source to target using shortest path
        cost_numpy = float(
            sum([numpy_env.graph.es["travel_time"][i] for i in shortest_path])
        )

        # Jax environment
        weights_matrix = jax_env._get_weight_matrix(edge_travel_times)
        cost_jax = jax_env._get_cost_to_go(weights_matrix)[source][target]

        assert jnp.allclose(cost_numpy, cost_jax, rtol=1e-3)


@pytest.mark.skip(reason="Old implementation references.")
def test_shortest_paths_large(cologne_environment, cologne_environment_jax):
    """
    Does shortest path compute by jax environment match the shortest path(s)
    computed by numpy environment?

    Use the same trips for both environments, get all shortest paths for
    each trip. Check if JAX finds at least one of the numpy paths.
    """
    jax_env = cologne_environment_jax
    numpy_env = cologne_environment

    trips = numpy_env.trips

    # Numpy environment
    numpy_env.graph.es["volume"] = 0
    numpy_env.graph.es["travel_time"] = [
        edge["road_segments"].compute_edge_travel_time(edge["volume"])
        for edge in numpy_env.graph.es
    ]

    # Jax environment
    _, state = jax_env.reset_env()
    edge_volumes = jnp.full((jax_env.num_edges), 0)
    edge_travel_times = jax_env.compute_edge_travel_time(state, edge_volumes)

    for source, target, _ in trips:

        # Numpy environment

        # list of nodes in the shortest paths
        # length of the list is the number of unique shortest paths
        # and each element is a list of nodes in the shortest path
        all_shortest_paths_nodes = numpy_env.graph.get_all_shortest_paths(
            source,
            target,
            weights="travel_time",
            mode="out",
        )

        # get edge ids of the shortest path
        all_shortest_paths_numpy = [
            [
                numpy_env.graph.get_eid(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]
            for path in all_shortest_paths_nodes
        ]

        # Jax environment
        weights_matrix = jax_env._get_weight_matrix(edge_travel_times)
        cost_to_go_matrix = jax_env._get_cost_to_go(weights_matrix)

        weights_matrix = weights_matrix.at[
            jnp.arange(jax_env.num_nodes), jnp.arange(jax_env.num_nodes)
        ].set(jnp.inf)

        shortest_path_jax = jax_env._get_shortest_path(
            source, target, weights_matrix, cost_to_go_matrix
        )

        assert shortest_path_jax in all_shortest_paths_numpy


@pytest.mark.skip(reason="Old implementation references.")
def test_get_travel_time(toy_environment_2, toy_environment_2_jax):
    "Test total travel time is the same for jax and numpy env"
    actions = [
        [1 for _ in edge["road_edge"].segments] for edge in toy_environment_2.graph.es
    ]
    timestep = 0
    done = False
    total_num_segments = sum(
        [len(edge.segments) for edge in toy_environment_2.graph.es["road_edge"]]
    )

    while not done:
        timestep += 1
        obs, _, done, info = toy_environment_2.step(actions)
        dam_state = jnp.array(toy_environment_2._get_states()).flatten()
        belief = jnp.array(obs["edge_beliefs"]).flatten()
        total_travel_time_np = info["total_travel_time"]
        base_travel_times = np.empty(total_num_segments)
        capacities = np.empty(total_num_segments)
        id_counter = 0
        for edge in toy_environment_2.graph.es["road_edge"]:
            for segment in edge.segments:
                base_travel_times[id_counter] = segment.base_travel_time
                capacities[id_counter] = segment.capacity
                id_counter += 1
        jax_state = EnvState(
            damage_state=dam_state,
            observation=jnp.array(obs["edge_observations"]).flatten(),
            belief=belief,
            base_travel_time=jnp.asarray(base_travel_times),
            capacity=jnp.asarray(capacities),
            timestep=timestep,
        )
        total_travel_time_jax = toy_environment_2_jax._get_total_travel_time(jax_state)
        print(total_travel_time_np, total_travel_time_jax)
        assert jnp.allclose(total_travel_time_np, total_travel_time_jax, rtol=1e-2)


@pytest.mark.skip(reason="Old implementation references.")
def test_belief_computation(toy_environment_2, toy_environment_2_jax):
    action_np = [[1, 1] for edge in toy_environment_2.graph.es]
    action_jax = [{"0": [1, 1]}, {"1": [1, 1]}, {"2": [1, 1]}, {"3": [1, 1]}]
    action_jax = jax.tree_util.tree_leaves(action_jax)
    action_jax = jnp.array(action_jax, dtype=jnp.uint8)

    observation = toy_environment_2.reset()
    belief = jnp.array(observation["edge_beliefs"]).reshape(-1, 4)
    done = False

    while not done:
        observation, _, done, _ = toy_environment_2.step(action_np)
        obs = jnp.array(observation["edge_observations"]).flatten()
        print(jnp.array(observation["edge_beliefs"]), obs, action_jax)
        belief_jax = toy_environment_2_jax._get_next_belief(belief, obs, action_jax)
        belief = jnp.array(observation["edge_beliefs"]).reshape(-1, 4)

        assert jnp.allclose(belief, belief_jax, rtol=1e-3)


@pytest.mark.skip(reason="Old implementation references.")
def test_campaign_reward(toy_environment_2_jax):
    """Test inspection campaign reward computation."""

    # inspections on 0 edges => campaign reward: 0
    _action = [{"0": [0, 2]}, {"1": [2, 0]}, {"2": [2, 2]}, {"3": [2, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    campaign_reward = toy_environment_2_jax._get_campaign_reward(action)

    assert campaign_reward == 0

    # inspections on 3 edges => campaign reward: -3 * 5 = -15
    _action = [{"0": [1, 2]}, {"1": [1, 1]}, {"2": [1, 0]}, {"3": [2, 3]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    campaign_reward = toy_environment_2_jax._get_campaign_reward(action)

    assert campaign_reward == -15


@pytest.mark.skip(reason="Old implementation references.")
def test_jax_keys(toy_environment_2_jax):

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    # rollout
    key = jax.random.PRNGKey(442)
    step_keys, key = toy_environment_2_jax.split_key(key)

    # environment reset
    _, state = toy_environment_2_jax.reset_env()

    done = False

    while not done:
        _, _, done, _, state = toy_environment_2_jax.step_env(step_keys, state, action)

        # generate keys for next timestep
        step_keys, key = toy_environment_2_jax.split_key(key)

    _rollout_key = key

    # check if keys are the same
    assert (jnp.array([3808878501, 3829080728]) == _rollout_key).all()


@pytest.mark.skip(reason="Old implementation references.")
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
