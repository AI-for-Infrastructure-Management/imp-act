import jax.numpy as jnp
import pytest

from environment import RoadEnvironment as NumPyRoadEnvironment
from environment_presets import small_environment_dict
from igraph import Graph

from jax_environment import JaxRoadEnvironment
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
        edge_segments_numbers: jnp.array = jnp.array([2, 2, 2, 2])
        total_num_segments: int = 8
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


def test_total_base_travel_time(small_numpy_environment, small_jax_environment, params):
    _, _ = small_jax_environment.reset_env()
    _jax = small_jax_environment.total_base_travel_time

    _numpy = small_numpy_environment.base_total_travel_time

    assert _jax == _numpy


def test_shortest_path_computation(graph_params):
    """Test shortest path computation."""

    _num_vertices = 7
    edges_list = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    weights_list = [2, 6, 5, 8, 10, 15, 2, 6]

    source = 0
    target = 6

    # create graph using igraph
    graph = Graph()
    graph.add_vertices(_num_vertices)
    graph.add_edges(edges_list)

    # Find shortest path using igraph
    shortest_path = graph.get_shortest_paths(
        source, target, weights=weights_list, mode="out", output="epath"
    )
    # get cost to travel from 0 to 6 using shortest path
    cost_1 = sum([weights_list[i] for i in shortest_path[0]])

    jax_env = JaxRoadEnvironment(graph_params)

    # get cost to travel from 0 to 6
    weights_matrix = jax_env._get_weight_matrix(
        graph_params.edge_weights, graph_params.edges, target
    )
    cost_2 = jax_env._get_cost_to_go(weights_matrix, 100)[source]

    assert cost_1 == cost_2
