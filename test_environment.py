import pytest
import time
import numpy as np

from environment import RoadEnvironment
from environment_presets import *

@pytest.fixture
def small_environment():
    env = RoadEnvironment(**small_environment_dict)
    return env

def test_observation_keys(small_environment):
    """Test if the observation dictionary has the correct keys. In reset and step functions."""

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
    
    obs = env.reset()
    actions = [[1,1] for _ in range(4)]
    timestep = 0

    max_timesteps = 1000
    while timestep < max_timesteps:
        timestep += 1
        obs, cost, done, info = env.step(actions)
        if done:
            break

    assert timestep == small_environment_dict["max_timesteps"]


def test_timing(small_environment):
    "Test if the average time per trajectory is below the threshold"
    env = small_environment
    
    _ = env.reset()
    actions = [[k,k] for k in range(4)]

    MAX_TIME_PER_TRAJECTORY = 2 # seconds
    timesteps_per_traj = small_environment_dict['max_timesteps']
    repeats = 100
    store_timings = np.empty(repeats)

    # Run the environment for a number of repeats
    for k in range(repeats):
        state_time = time.time()
        done = False
        while not done:
            _, _, done, _ = env.step(actions)
        store_timings[k] = time.time() - state_time

    # print(f'Total time taken: {store_timings.sum():.2e} seconds')
    # print(f'Average time per rollout: {store_timings.mean():.2e} seconds')
    # print(f'Average time taken per timestep: {store_timings.mean()/timesteps_per_traj:.2e} seconds')

    assert store_timings.mean() < MAX_TIME_PER_TRAJECTORY