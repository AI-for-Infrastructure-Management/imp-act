import time

import numpy as np

import pytest

from imp_act.environments.road_env import RoadSegment


environment_fixtures = [
    "toy_environment_1",
    "toy_environment_2",
    "small_environment",
    "medium_environment",
    "large_environment",
]


def test_observation_keys(toy_environment_1):
    """Test if the observation dictionary has the correct keys in reset and step functions."""

    env = toy_environment_1
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


def test_increasing_timesteps(toy_environment_2):
    """Test if the environment can run multiple episodes with increasing timesteps."""

    TEST_EPISODES = 3

    env = toy_environment_2

    for episode in range(TEST_EPISODES):
        timestep = 0
        obs = env.reset()

        done = False
        while not done:
            actions = [
                np.random.randint(0, 2, len(e)) for e in obs["edge_observations"]
            ]
            obs, reward, done, info = env.step(actions)
            timestep += 1
            assert obs["time_step"] == env.timestep
            assert obs["time_step"] == timestep


@pytest.mark.parametrize(
    "parameter_fixture",
    environment_fixtures,
    indirect=True,
)
def test_one_episode(parameter_fixture):
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


def test_timing(toy_environment_1):
    "Test if the average time per trajectory is below the threshold"
    env = toy_environment_1

    obs = env.reset()
    actions = [[1] * len(e) for e in obs["edge_observations"]]

    MAX_TIME_PER_TRAJECTORY = 2  # seconds
    repeats = 100
    store_timings = np.empty(repeats)

    # Run the environment for a number of repeats
    for k in range(repeats):
        state_time = time.time()
        obs = env.reset()
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
    environment_fixtures,
    indirect=True,
)
def test_only_negative_rewards(parameter_fixture, test_seed_1, random_time_seed):
    """Test to ensure that the environment only returns negative rewards (or zero)."""
    env = parameter_fixture
    obs = env.reset()
    actions = [[0] * len(e) for e in obs["edge_observations"]]
    _, rewards, _, _ = seeded_episode_rollout(env, test_seed_1, actions)
    check = [reward <= 0 for reward in rewards]
    assert all(check)

    _, rewards, _, _ = seeded_episode_rollout(env, random_time_seed, actions)
    check = [reward <= 0 for reward in rewards]
    assert all(check)


WARN_LIMIT_RATIO = 2
FAIL_LIMIT_RATIO = 3


@pytest.mark.skip(reason="Waiting for final calibration of the environment")
@pytest.mark.parametrize(
    "parameter_fixture",
    environment_fixtures,
    indirect=True,
)
def test_segment_volume_to_capacity_ratio_within_resonable_limits(
    parameter_fixture, test_seed_1, random_time_seed
):
    """Test if the segment volume to capacity ratio is within reasonable limits."""
    env = parameter_fixture

    def test_capacity_ratio(env, seed):
        obs = env.reset()
        env.seed(seed)
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
                    assert volume / capacity <= FAIL_LIMIT_RATIO
                    if not warned_once and volume / capacity > WARN_LIMIT_RATIO:
                        warned_once = True
                        print(
                            f"Warning: Volume to capacity ratio is {volume / capacity} ({volume} / {capacity})"
                        )

    test_capacity_ratio(env, test_seed_1)
    test_capacity_ratio(env, random_time_seed)


def test_stationary_deterioration_environment(stationary_deterioration_environment):
    """Test if the stationary deterioration environment can run one episode."""
    assert (
        not stationary_deterioration_environment.graph.es[0]["road_segments"]
        .segments[0]
        .deterioration_rate_enabled
    )
    test_one_episode(stationary_deterioration_environment)


def test_actions_unchanged(toy_environment_2, test_seed_1):
    """Test if the actions are not changed by the environment."""
    env = toy_environment_2
    obs = env.reset()
    actions = [np.random.randint(0, 2, len(e)) for e in obs["edge_observations"]]
    actions_copy = actions.copy()
    _, _, _, _ = seeded_episode_rollout(env, test_seed_1, actions)
    assert actions == actions_copy


def test_budget(toy_environment_2):
    """
    Test if the budget is always positive.
    Test if the budget is reset after the budget renewal time.
    Test if the budget is decreased by the cost of the actions.
    """

    def check_budget_positive(env, obs, reward, done, info):
        assert obs["budget_remaining"] >= 0, "Budget is negative"

    def check_budget_renewed(env, obs, reward, done, info):
        if obs["budget_time_until_renewal"] % env.budget_renewal_interval == 0:
            assert (
                obs["budget_remaining"] == env.budget_amount
            ), "Budget is not reset after the budget renewal time"

    def get_forced_maintenance_actions(env, actions):
        # Corrective replace action if the worst condition is observed
        for i, edge in enumerate(env.graph.es):
            for j, segment in enumerate(edge["road_segments"].segments):
                if (
                    segment.observation == segment.number_of_states - 1
                    or segment.forced_repair_interest_counter > 0
                ):
                    actions[i][j] = RoadSegment.ACTION_REPLACE
        return actions

    TEST_EPISODES = 3

    env = toy_environment_2

    for episode in range(TEST_EPISODES):
        obs = env.reset()
        check_budget_positive(env, obs, None, None, None)
        check_budget_renewed(env, obs, None, None, None)
        last_budget = obs["budget_remaining"]

        done = False
        while not done:
            actions = [
                np.random.randint(0, 2, len(e)) for e in obs["edge_observations"]
            ]
            constrained_actions = get_forced_maintenance_actions(
                env, [action.copy() for action in actions]
            )
            action_cost = env.get_action_cost(constrained_actions)

            returns = env.step(actions)
            check_budget_renewed(env, *returns)
            check_budget_positive(env, *returns)

            obs, reward, done, info = returns
            if (
                not info["budget_constraints_applied"]
                and obs["budget_time_until_renewal"] % env.budget_renewal_interval != 0
            ):
                assert (
                    last_budget - action_cost == obs["budget_remaining"]
                ), "Budget is not decreased by the cost of the actions"

            last_budget = obs["budget_remaining"]
