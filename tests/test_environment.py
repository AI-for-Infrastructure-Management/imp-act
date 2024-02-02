import time

import numpy as np
import pytest
from environments.config.environment_presets import small_environment_dict

from environments.road_env import RoadEnvironment


@pytest.fixture
def small_environment():
    """Create a small environment for testing."""
    small_environment_dict["seed"] = 42
    env = RoadEnvironment(**small_environment_dict)
    return env


@pytest.fixture
def large_environment_path():
    """Path to large environment file"""
    return "environments/config/environment_presets/large_environment.yaml"


@pytest.fixture
def large_environment(large_environment_path):
    loader = EnvironmentLoader(large_environment_path)
    return loader.to_numpy()


def test_observation_keys(small_environment):
    """Test if the observation dictionary has the correct keys in reset and step functions."""

    env = small_environment
    obs = env.reset()

    keys = [
        "adjacency_matrix",
        "edge_observations",
        "edge_beliefs",
        "edge_nodes",
    ]

    for key in keys:
        assert key in obs.keys()

    actions = [[1, 1] for _ in range(len(env.edge_segments_numbers))]
    obs, cost, done, info = env.step(actions)

    for key in keys:
        assert key in obs.keys()


def test_one_episode(small_environment):
    """Test if the environment can run one episode."""
    env = small_environment

    _ = env.reset()
    actions = [[1, 1] for _ in range(len(env.edge_segments_numbers))]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == env.max_timesteps


# @pytest.mark.skip(reason="Takes too long")
def test_large_environment(large_environment):
    """Test if the large environment can run one episode."""
    env = large_environment

    start_time = time.time()

    obs = env.reset()
    actions = [[1] * len(e) for e in obs["edge_observations"]]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == small_environment_dict["max_timesteps"]

    print(f"\nLarge ENV one episode time taken: {time.time() - start_time:.2} seconds")
    print(
        f"Nodes: {len(env.graph.vs)}, Edges: {len(env.graph.es)}, Timesteps: {timestep}, Trips: {len(env.trips)}"
    )
    print("Test Result: ", end="")


def test_timing(small_environment):
    "Test if the average time per trajectory is below the threshold"
    env = small_environment

    _ = env.reset()
    actions = [[k, k] for k in range(len(env.edge_segments_numbers))]

    MAX_TIME_PER_TRAJECTORY = 2  # seconds
    # timesteps_per_traj = small_environment_dict["max_timesteps"]
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


def test_seed(small_environment):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    actions = [[1, 1] for _ in range(4)]
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


def test_seed_interfering_sampler(small_environment):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    actions = [[1, 1] for _ in range(4)]
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
    counter = 1
    for episode in range(n_episodes):
        obs_same = env_same.reset()
        done = False
        reward_same_episode = []
        while not done:
            obs_same, reward_same, done, info = env_same.step(actions)
            if counter % 5 == 0:
                np.random.randint(0, 100)
            counter += 1
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


def test_seed_np_random_seed(small_environment):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    actions = [[1, 1] for _ in range(4)]
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
    counter = 1
    for episode in range(n_episodes):
        obs_same = env_same.reset()
        done = False
        reward_same_episode = []
        while not done:
            obs_same, reward_same, done, info = env_same.step(actions)
            if counter % 5 == 0:
                np.random.seed(1337)
            counter += 1
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


def test_seeding_function(small_environment):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    actions = [[1, 1] for _ in range(4)]
    n_episodes = 2

    same_seed = small_environment_dict["seed"]
    different_seed = 1337

    # Collect episodes
    env = small_environment
    reward_all = []
    for episode in range(n_episodes):
        env.seed(same_seed + episode)
        obs = env.reset()
        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    small_environment_dict_same = small_environment_dict.copy()
    small_environment_dict_same["seed"] = different_seed  # different seed during init
    env_same = RoadEnvironment(**small_environment_dict_same)
    reward_same_all = []
    for episode in range(n_episodes):
        env_same.seed(same_seed + episode)  # same seed before episode
        obs_same = env_same.reset()
        done = False
        reward_same_episode = []
        while not done:
            obs_same, reward_same, done, info = env_same.step(actions)
            reward_same_episode.append(reward_same)
        reward_same_all.append(reward_same_episode)

    # Create env with different random seed and collect episodes
    small_environment_dict_different = small_environment_dict.copy()
    small_environment_dict_different["seed"] = same_seed  # same seed during init
    env_different = RoadEnvironment(**small_environment_dict_different)
    reward_different_all = []
    for episode in range(n_episodes):
        env_different.seed(different_seed + episode)  # different seed before episode
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


from environments.config.environment_loader import EnvironmentLoader


def test_large_environment_load(large_environment_path):
    """Test if the large environment can be loaded from congig files"""
    loader = EnvironmentLoader(large_environment_path)
    loader.to_numpy()
