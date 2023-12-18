import pytest
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
    obs, reward, done, info = env.step(actions)

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
        obs, reward, done, info = env.step(actions)
        if done:
            break

    assert timestep == small_environment_dict["max_timesteps"]

def test_seed(small_environment):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    actions = [[1,1] for _ in range(4)]
    n_episodes = 2

    # Collect episodes
    env = small_environment
    reward_all = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    small_environment_dict_same = small_environment_dict.copy()
    small_environment_dict_same["seed"] = 42
    env_same = RoadEnvironment(**small_environment_dict_same)
    reward_same_all = []
    for episode in range(n_episodes):
        obs_same = env_same.reset()
        done = False
        reward_same_episode = []
        while not done:
            obs_same, reward_same, done, info = env_same.step(actions)
            reward_same_episode.append(reward_same)
        reward_same_all.append(reward_same_episode)

    # Create env with different random seed and collect episodes
    small_environment_dict_different = small_environment_dict.copy()
    small_environment_dict_different["seed"] = 43
    env_different = RoadEnvironment(**small_environment_dict_different)
    reward_different_all = []
    for episode in range(n_episodes):
        obs_different = env_different.reset()
        done = False
        reward_different_episode = []
        while not done:
            obs_different, reward_different, done, info = env_different.step(actions)
            reward_different_episode.append(reward_different)
        reward_different_all.append(reward_different_episode)

    # Assert episodes are different after reset
    if n_episodes > 1:
        assert not np.array_equal(reward_all[0], reward_all[1])

    # Assert env is reproducible and seeding works effectively
    assert np.array_equal(reward_all, reward_same_all)
    assert not np.array_equal(reward_all, reward_different_all)