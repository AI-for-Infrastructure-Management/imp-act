import pytest

from environment import RoadEnvironment
from environment_presets import *
import igraph as ig

@pytest.fixture
def small_environment():
    """Create a small environment for testing."""
    env = RoadEnvironment(**small_environment_dict)
    return env

@pytest.fixture
def large_environment():
    """Create a large environment for testing."""
    graph = ig.Graph.Read_GraphML("germany.graphml")
    
    edge_segments_numbers = [2 for _ in range(len(graph.es))]
    trips = [(a,b, 200) for a,b in graph.get_edgelist()]
    env = RoadEnvironment(None, None, edge_segments_numbers, trips, max_timesteps=50, graph=graph)

    return env

def test_observation_keys(small_environment):
    """Test if the observation dictionary has the correct keys in reset and step functions."""

    env = small_environment
    obs = env.reset()

    keys =  ["adjacency_matrix", 
            "edge_observations",
            "edge_beliefs",
            "edge_nodes",]
    
    for key in keys:
        assert key in obs.keys()

    actions = [[1,1] for _ in range(len(env.edge_segments_numbers))]
    obs, cost, done, info = env.step(actions)

    for key in keys:
        assert key in obs.keys()

def test_one_episode(small_environment):
    """Test if the environment can run one episode."""
    env = small_environment

    _ = env.reset()
    actions = [[1,1] for _ in range(len(env.edge_segments_numbers))]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == env.max_timesteps

def test_large_environment(large_environment):
    """Test if the large environment can run one episode."""
    env = large_environment

    obs = env.reset()
    actions = [[1,1] for _ in range(len(env.edge_segments_numbers))]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == env.max_timesteps