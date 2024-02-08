import itertools

import jax
import jax.numpy as jnp
import pytest

from environment import RoadEnvironment as NumPyRoadEnvironment
from environment_presets import small_environment_dict
from igraph import Graph

from jax_environment import EnvState, JaxRoadEnvironment
from params import EnvParams


@pytest.fixture
def params():
    return EnvParams()


@pytest.fixture
def graph_params():
    class GraphParams:
        # Horizon parameters
        max_timesteps: int = 50
        discount_factor: float = 1.0

        # Reward parameters
        travel_time_reward_factor: float = -0.01

        # Graph parameters
        num_vertices: int = 7
        edges: jnp.array = jnp.array(
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
        )
        num_edges: int = 8
        edge_segments_numbers: jnp.array = jnp.array([2] * num_edges)
        edge_weights = [2, 6, 5, 8, 10, 15, 2, 6]

        shortest_path_max_iterations: int = 500
        traffic_assignment_max_iterations: int = 15
        traffic_assignment_convergence_threshold: float = 0.01
        traffic_assignment_update_weight: float = 0.5
        traffic_alpha: float = 0.15
        traffic_beta: float = 4.0

        # Road Network parameters
        trips: jnp.array = jnp.array(
            [[0, 200, 0, 0], [0, 0, 0, 200], [200, 0, 0, 0], [0, 0, 200, 0]],
            dtype=jnp.int32,
        )

        btt_table: jnp.array = (
            jnp.array(
                [
                    [1.00, 1.10, 1.40, 1.60],
                    [1.00, 1.10, 1.40, 1.60],
                    [1.00, 1.05, 1.15, 1.45],
                    [1.50, 1.50, 1.50, 1.50],
                ]
            )
            * 50.0
        )

        capacity_table: jnp.array = (
            jnp.array(
                [
                    [1.00, 1.00, 1.00, 1.00],
                    [1.00, 1.00, 1.00, 1.00],
                    [0.80, 0.80, 0.80, 0.80],
                    [0.50, 0.50, 0.50, 0.50],
                ]
            )
            * 500.0
        )

        # Damage parameters
        num_dam_states: int = 4
        initial_dam_state: int = 0
        initial_obs: int = 0
        initial_belief: jnp.array = jnp.array([1.0, 0.0, 0.0, 0.0])

        deterioration_table: jnp.array = jnp.array(
            [
                [  # Action 0: do-nothing
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                    [0.0, 0.0, 0.9, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [  # Action 1: inspect
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                    [0.0, 0.0, 0.9, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [  # Action 2: minor repair
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.8, 0.2, 0.0, 0.0],
                    [0.7, 0.2, 0.1, 0.0],
                ],
                [  # Action 3: major repair (replacement)
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        observation_table: jnp.array = jnp.array(
            [
                [  # Action 0: do-nothing
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [  # Action 1: inspect
                    [0.8, 0.2, 0.0, 0.0],
                    [0.1, 0.8, 0.1, 0.0],
                    [0.0, 0.1, 0.9, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [  # Action 2: minor repair
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [  # Action 3: major repair (replacement)
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ]
        )

        rewards_table: jnp.array = jnp.array(
            [
                [0, -1, -20, -150],
                [0, -1, -25, -150],
                [0, -1, -30, -150],
                [0, -1, -40, -150],
            ]
        )

    return GraphParams()


@pytest.fixture
def small_jax_environment(params):
    return JaxRoadEnvironment(params)


@pytest.fixture
def small_numpy_environment():
    """Create a small environment for testing."""
    small_environment_dict["seed"] = 42
    env = NumPyRoadEnvironment(**small_environment_dict)
    return env


def test_total_base_travel_time(small_numpy_environment, small_jax_environment):
    _, _ = small_jax_environment.reset_env()
    _jax = small_jax_environment.total_base_travel_time

    _numpy = small_numpy_environment.base_total_travel_time

    assert _jax == _numpy


@pytest.mark.skip(reason="Takes too long")
def test_shortest_path_computation(graph_params):
    """Test shortest path computation."""

    _num_vertices = 7
    edges_list = jnp.array(
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    )

    weights_list = [2, 6, 5, 8, 10, 15, 2, 6]

    # create graph using igraph
    graph = Graph()
    graph.add_vertices(_num_vertices)
    graph.add_edges(edges_list)

    jax_env = JaxRoadEnvironment(graph_params)

    sources = [0, 1, 2, 3, 4, 5, 6]
    targets = [0, 1, 2, 3, 4, 5, 6]

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
        weights_matrix = jax_env._get_weight_matrix(weights_list, edges_list, target)
        print(weights_matrix)
        cost_2 = jax_env._get_cost_to_go(weights_matrix, 100)[source]
        print(cost_2)

        assert cost_1 == cost_2


def test_get_travel_time(small_numpy_environment, small_jax_environment):
    "Test total travel time is the same for jax and numpy env"
    actions = [
        [1, 1] for _ in range(len(small_numpy_environment.edge_segments_numbers))
    ]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = small_numpy_environment.step(actions)
        dam_state = jnp.array(small_numpy_environment._get_states()).flatten()
        belief = jnp.array(obs["edge_beliefs"]).flatten()
        total_travel_time_np = info["total_travel_time"]
        base_travel_times = []
        capacities = []
        for edge in small_numpy_environment.graph.es["road_segments"]:
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
        total_travel_time_jax = small_jax_environment._get_total_travel_time(jax_state)
        print(total_travel_time_np, total_travel_time_jax)
        assert total_travel_time_np.round() == total_travel_time_jax.round()


def test_jax_keys(small_jax_environment):

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    # rollout
    key = jax.random.PRNGKey(442)
    step_keys, key = small_jax_environment.split_key(key)

    # environment reset
    _, state = small_jax_environment.reset_env()

    done = False

    while not done:
        _, _, done, _, state = small_jax_environment.step_env(step_keys, state, action)

        # generate keys for next timestep
        step_keys, key = small_jax_environment.split_key(key)

    _rollout_key = key

    # check if keys are the same
    assert (jnp.array([3808878501, 3829080728]) == _rollout_key).all()
