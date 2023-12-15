import pytest

from environment import RoadEnvironment
from environment_presets import *
import igraph as ig

@pytest.fixture
def small_environment():
    """Create a small environment for testing."""
    env = RoadEnvironment(**small_environment_dict)
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

    actions = [[1,1] for _ in range(4)]
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
