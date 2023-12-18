import pytest
import time

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
    env = small_environment
    
    obs = env.reset()
    actions = [[k,k] for k in range(4)]
    timestep = 0

    max_time_per_trajectory = 2
    timesteps_per_traj = small_environment_dict['max_timesteps']
    time_vec = list()
    repeats = 100

    # check if the average time over 100 trajectories is below the treshold
    for _ in range(repeats):
        t = time.time()
        while timestep < timesteps_per_traj:
            timestep += 1
            obs, cost, done, info = env.step(actions)
        time_vec.append(time.time() - t)

    #print(f'Average time taken per timestep: {sum(time_vec)/timesteps_per_traj/len(time_vec):.10f}')
    assert sum(time_vec)/len(time_vec) < max_time_per_trajectory