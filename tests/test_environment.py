import time

import numpy as np

import pytest


def test_observation_keys(toy_environment):
    """Test if the observation dictionary has the correct keys in reset and step functions."""

    env = toy_environment
    obs = env.reset()

    keys = [
        "adjacency_matrix",
        "edge_observations",
        "edge_beliefs",
        "edge_nodes",
    ]

    for key in keys:
        assert key in obs.keys()

    actions = [[1] * len(e) for e in obs["edge_observations"]]
    obs, cost, done, info = env.step(actions)

    for key in keys:
        assert key in obs.keys()


def test_one_episode(toy_environment):
    """Test if the environment can run one episode."""
    env = toy_environment

    obs = env.reset()
    actions = [[1] * len(e) for e in obs["edge_observations"]]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == env.max_timesteps


@pytest.mark.parametrize(
    "parameter_fixture",
    ["small_environment", "medium_environment", "large_environment"],
    indirect=True,
)
def test_environment(parameter_fixture):
    """Test if the environment can run one episode."""
    env = parameter_fixture

    start_time = time.time()

    obs = env.reset()
    actions = [[1] * len(e) for e in obs["edge_observations"]]
    timestep = 0
    done = False

    while not done:
        timestep += 1
        obs, cost, done, info = env.step(actions)

    assert timestep == env.max_timesteps

    print(
        f"\nNodes: {len(env.graph.vs)}, Edges: {len(env.graph.es)}, Timesteps: {timestep}, Trips: {len(env.trips)}"
    )
    print(f"One episode time taken: {time.time() - start_time:.2} seconds")
    print("Test Result: ", end="")


def test_timing(toy_environment):
    "Test if the average time per trajectory is below the threshold"
    env = toy_environment

    obs = env.reset()
    actions = [[1] * len(e) for e in obs["edge_observations"]]

    MAX_TIME_PER_TRAJECTORY = 2  # seconds
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


@pytest.fixture
def test_seed_1():
    return 42


@pytest.fixture
def test_seed_2():
    return 1337


def test_seed(toy_environment_loader, test_seed_1, test_seed_2):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    n_episodes = 2

    # Collect episodes
    env = toy_environment_loader.to_numpy()
    env.seed(test_seed_1)
    reward_all = []
    for episode in range(n_episodes):
        obs = env.reset()
        actions = [[1] * len(e) for e in obs["edge_observations"]]
        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    env_same = toy_environment_loader.to_numpy()
    env_same.seed(test_seed_1)
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
    env_different = toy_environment_loader.to_numpy()
    env_different.seed(test_seed_2)

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


def test_seed_interfering_sampler(toy_environment_loader, test_seed_1, test_seed_2):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    n_episodes = 2

    # Collect episodes
    env = toy_environment_loader.to_numpy()
    env.seed(test_seed_1)

    reward_all = []
    for episode in range(n_episodes):
        obs = env.reset()
        actions = [[1] * len(e) for e in obs["edge_observations"]]

        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    env_same = toy_environment_loader.to_numpy()
    env_same.seed(test_seed_1)

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
    env_different = toy_environment_loader.to_numpy()
    env_different.seed(test_seed_2)

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


def test_seed_np_random_seed(toy_environment_loader, test_seed_1, test_seed_2):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    n_episodes = 2

    # Collect episodes
    env = toy_environment_loader.to_numpy()
    env.seed(test_seed_1)

    reward_all = []
    for episode in range(n_episodes):
        obs = env.reset()
        actions = [[1] * len(e) for e in obs["edge_observations"]]
        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    env_same = toy_environment_loader.to_numpy()
    env_same.seed(test_seed_1)

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
    env_different = toy_environment_loader.to_numpy()
    env_different.seed(test_seed_2)

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


def test_seeding_function(toy_environment_loader, test_seed_1, test_seed_2):
    """Test if the environment is reproducible"""
    # Fix actions and number of episodes
    n_episodes = 2

    # Collect episodes
    env = toy_environment_loader.to_numpy()
    env.seed(test_seed_1)
    reward_all = []
    for episode in range(n_episodes):
        env.seed(test_seed_1 + episode)
        obs = env.reset()
        actions = [[1] * len(e) for e in obs["edge_observations"]]
        done = False
        reward_episode = []
        while not done:
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
        reward_all.append(reward_episode)

    # Create env with same random seed and collect episodes
    env_same = toy_environment_loader.to_numpy()
    env_same.seed(test_seed_1)

    reward_same_all = []
    for episode in range(n_episodes):
        env_same.seed(test_seed_1 + episode)  # same seed before episode
        obs_same = env_same.reset()
        done = False
        reward_same_episode = []
        while not done:
            obs_same, reward_same, done, info = env_same.step(actions)
            reward_same_episode.append(reward_same)
        reward_same_all.append(reward_same_episode)

    # Create env with different random seed and collect episodes
    env_different = toy_environment_loader.to_numpy()
    env_different.seed(test_seed_1)

    reward_different_all = []
    for episode in range(n_episodes):
        env_different.seed(test_seed_2 + episode)  # different seed before episode
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


def test_registry_unknown():
    from imp_act.environments.registry import Registry

    registry = Registry()
    with pytest.raises(ValueError):
        registry.make("unknown_environment")


def seeded_episode_rollout(environment, seed, actions):
    """Run a single episode with a given seed and actions."""
    env = environment
    env.seed(seed)
    obs = env.reset()
    done = False
    observations, rewards, dones, infos = [], [], [], []
    while not done:
        obs, reward, done, info = env.step(actions)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
    return observations, rewards, dones, infos 


@pytest.mark.parametrize(
    "parameter_fixture",
    ["toy_environment", "small_environment", "medium_environment", "large_environment"],
    indirect=True,
)
def test_only_negative_rewards(parameter_fixture, test_seed_1, random_time_seed):
    """ Test to ensure that the environment only returns negative rewards (or zero). """
    env = parameter_fixture
    obs = env.reset()
    actions = [[0] * len(e) for e in obs["edge_observations"]]
    _, rewards, _, _  = seeded_episode_rollout(env, test_seed_1, actions)
    check = [reward <= 0 for reward in rewards]
    assert all(check)

    _, rewards, _, _  = seeded_episode_rollout(env, random_time_seed, actions)
    check = [reward <= 0 for reward in rewards]
    assert all(check)


WARN_LIMIT_RATIO = 2
FAIL_LIMIT_RATIO = 3
@pytest.mark.parametrize(
    "parameter_fixture",
    ["toy_environment", "small_environment", "medium_environment", "large_environment"],
    indirect=True,
)
def test_segment_volume_to_capacity_ratio_within_resonable_limits(parameter_fixture, test_seed_1, random_time_seed):
    """ Test if the segment volume to capacity ratio is within reasonable limits. """
    env = parameter_fixture
    obs = env.reset()
    actions = [[0] * len(e) for e in obs["edge_observations"]]
    done = False
    warned_once = False
    while not done:
        _, _, done, _ = env.step(actions)
        for edge in env.graph.es:
            volume = edge["volume"]
            for segment in edge["road_segments"].segments:
                capacity = segment.capacity
                if capacity == 0:
                    continue
                assert (volume / capacity <= FAIL_LIMIT_RATIO)
                if not warned_once and volume / capacity > WARN_LIMIT_RATIO:
                    warned_once = True
                    print(f"Warning: Volume to capacity ratio is {volume / capacity} ({volume} / {capacity})")

