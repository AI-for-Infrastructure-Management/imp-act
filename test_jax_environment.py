import pytest
import numpy as np
import pytest

import jax
from igraph import Graph
import jax.numpy as jnp
from flax import struct

from environment import RoadEnvironment
from environment_presets import small_environment_dict

from rookie_jax import RoadEnvironment as JAXRoadEnvironment
from params import EnvParams


@pytest.fixture
def params():
    return EnvParams()


@pytest.fixture
def small_jax_environment(params):
    return JAXRoadEnvironment(params)


@pytest.fixture
def small_numpy_environment():
    """Create a small environment for testing."""
    small_environment_dict["seed"] = 42
    env = RoadEnvironment(**small_environment_dict)
    return env


def test_total_base_travel_time(small_numpy_environment, small_jax_environment, params):
    key = jax.random.PRNGKey(442)
    _, _ = small_jax_environment.reset_env(key, params)
    _jax = small_jax_environment.total_base_travel_time

    _numpy = small_numpy_environment.base_total_travel_time

    assert _jax == _numpy


def test_shortest_path_computation():
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

    # create graph using rookie_jax
    @struct.dataclass
    class EnvParams:
        # Graph parameters
        num_vertices: int = _num_vertices
        edges: jnp.array = jnp.array(edges_list)
        edge_weights: jnp.array = jnp.array(weights_list)

        # unnecessary parameters (only needed for jax_env constructor)
        trips: jnp.array = jnp.array(
            [[0, 200, 0, 0], [0, 0, 0, 200], [200, 0, 0, 0], [0, 0, 200, 0]],
            dtype=jnp.int32,
        )
        edge_segments_numbers: jnp.array = jnp.array([2, 2, 2, 2])
        num_edges: int = 1000
        num_dam_states: int = 4

    params = EnvParams()
    jax_env = JAXRoadEnvironment(params)

    # get cost to travel from 0 to 6
    weights_matrix = jax_env._get_weight_matrix(
        params.edge_weights, params.edges, target
    )
    cost_2 = jax_env._get_cost_to_go(weights_matrix, 100)[source]

    assert cost_1 == cost_2

