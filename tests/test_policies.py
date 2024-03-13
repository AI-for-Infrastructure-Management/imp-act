import pytest

from imp_act.policies.heuristics import SimpleHeuristicAgent, TCBMHeuristicAgent


@pytest.fixture
def simple_heuristic_agent():
    return SimpleHeuristicAgent(threshold=0.15)


@pytest.fixture
def tcbm_heuristic_agent():
    return TCBMHeuristicAgent(threshold=0.15, inspection_interval=10)


def test_simple_heuristic_agents_shape(
    simple_heuristic_agent, tcbm_heuristic_agent, small_environment  # noqa: F811;
):
    """Test if the simple heuristic agent returns the correct shape of actions."""
    agent = simple_heuristic_agent
    env = small_environment

    obs = env.reset()
    actions = agent.get_action(obs)

    edge_beliefs = obs["edge_beliefs"]
    for edge_belief, edge_action in zip(edge_beliefs, actions):
        assert len(edge_belief) == len(edge_action)
        for segment_belief, segment_action in zip(edge_belief, edge_action):
            assert type(segment_action) == int

    agent = tcbm_heuristic_agent
    obs = env.reset()
    actions = agent.get_action(obs)

    edge_beliefs = obs["edge_beliefs"]
    for edge_belief, edge_action in zip(edge_beliefs, actions):
        assert len(edge_belief) == len(edge_action)
        for segment_belief, segment_action in zip(edge_belief, edge_action):
            assert type(segment_action) == int


def test_simple_heuristic_agent_one_episode(
    simple_heuristic_agent, small_environment  # noqa: F811;
):
    """Test if the simple heuristic agent can run one episode."""
    agent = simple_heuristic_agent
    env = small_environment

    obs = env.reset()
    actions = agent.get_action(obs)
    timestep = 0

    max_timesteps = 1000
    while timestep < max_timesteps:
        timestep += 1
        obs, cost, done, info = env.step(actions)
        if done:
            break


@pytest.fixture
def no_action_observation_testcase():
    edge_beliefs = {
        "edge_beliefs": [
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        "time_step": 0,
    }

    actions = [
        [0, 0],
        [0, 0],
    ]

    return edge_beliefs, actions


def test_simple_heuristic_agent_no_action(
    no_action_observation_testcase, simple_heuristic_agent
):
    """Test if the simple heuristic agent returns no action when the belief is 1.0."""
    edge_beliefs, actions = no_action_observation_testcase
    agent = simple_heuristic_agent

    agent_actions = agent.get_action(edge_beliefs)
    assert actions == agent_actions


@pytest.fixture
def all_action_observation_testcase():
    edge_beliefs = {
        "edge_beliefs": [
            [
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0],
            ],
        ],
        "time_step": 0,
    }

    actions = [
        [2, 2],
        [2, 2],
    ]

    return edge_beliefs, actions


def test_simple_heuristic_agent_all_action(
    all_action_observation_testcase, simple_heuristic_agent
):
    """Test if the simple heuristic agent returns all action when the belief is 0.0."""
    edge_beliefs, actions = all_action_observation_testcase
    agent = simple_heuristic_agent

    agent_actions = agent.get_action(edge_beliefs)
    assert actions == agent_actions


def test_tcbm_heuristic_agent_one_episode(
    tcbm_heuristic_agent, small_environment  # noqa: F811;
):
    """Test if the simple heuristic agent can run one episode."""
    agent = tcbm_heuristic_agent
    env = small_environment

    obs = env.reset()
    actions = agent.get_action(obs)
    timestep = 0

    max_timesteps = 1000
    while timestep < max_timesteps:
        timestep += 1
        obs, cost, done, info = env.step(actions)
        if done:
            break


def test_tcbm_simple_heuristic_agent_no_action(
    no_action_observation_testcase, tcbm_heuristic_agent
):
    """Test if the tcbm heuristic agent returns no action when the belief is 1.0."""
    edge_beliefs, actions = no_action_observation_testcase
    agent = tcbm_heuristic_agent

    agent_actions = agent.get_action(edge_beliefs)
    assert actions == agent_actions


def test_tcbm_simple_heuristic_agent_all_action(
    all_action_observation_testcase, tcbm_heuristic_agent
):
    """Test if the tcbm heuristic agent returns all action when the belief is 0.0."""
    edge_beliefs, actions = all_action_observation_testcase
    agent = tcbm_heuristic_agent

    agent_actions = agent.get_action(edge_beliefs)
    assert actions == agent_actions


@pytest.fixture
def tcbm_heuristic_agent_time_step_inspection_testcase():
    edge_beliefs = {
        "edge_beliefs": [
            [
                [0.1, 0.9],
                [0.5, 0.5],
            ],
            [
                [0.5, 0.5],
                [0.1, 0.9],
            ],
        ],
        "time_step": 10,
    }

    actions = [
        [2, 1],
        [1, 2],
    ]

    return edge_beliefs, actions


def test_tcbm_heuristic_agent_time_step_inspection(
    tcbm_heuristic_agent_time_step_inspection_testcase, tcbm_heuristic_agent
):
    """Test if the tcbm heuristic agent returns inspection action when time step is 10 and repair action when belief is below threshold."""
    edge_beliefs, actions = tcbm_heuristic_agent_time_step_inspection_testcase
    agent = tcbm_heuristic_agent

    agent_actions = agent.get_action(edge_beliefs)
    assert actions == agent_actions


@pytest.fixture
def tcbm_heuristic_agent_no_timestep_key():
    edge_beliefs = {
        "edge_beliefs": [
            [
                [0.1, 0.9],
                [0.5, 0.5],
            ],
            [
                [0.5, 0.5],
                [0.1, 0.9],
            ],
        ],
    }

    return edge_beliefs


# def test_tcbm_heuristic_agent_no_timestep(
#     tcbm_heuristic_agent_no_timestep_key, tcbm_heuristic_agent
# ):
#     """Test if the tcbm heuristic agent can handle time_step key not in observation."""
#     edge_beliefs = tcbm_heuristic_agent_no_timestep_key
#     agent = tcbm_heuristic_agent
#
#     agent_actions = agent.get_action(edge_beliefs)
