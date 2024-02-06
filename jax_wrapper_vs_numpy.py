import time

import jax

from environment import RoadEnvironment as NumPyRoadEnvironment
from environment_presets import small_environment_dict

from jax import numpy as jnp

from jax_env_wrapper import JaxRoadEnvironmentWrapper
from params import EnvParams


def do_nothing_policy_numpy(env):
    return [[0] * segments for segments in env.edge_segments_numbers]


def do_nothing_policy_jax(env):
    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)
    return action


def rollout(env, actions):
    obs = env.reset()

    total_reward = 0

    for _ in range(env.max_timesteps):
        obs, reward, done, _ = env.step(actions)
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    experiments = [1, 10, 100, 1_000, 10_000]

    store_returns_for = experiments[-1]

    # NUMPY
    np_env = NumPyRoadEnvironment(**small_environment_dict)
    numpy_actions = do_nothing_policy_numpy(np_env)

    numpy_timings = []
    numpy_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = rollout(np_env, numpy_actions)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    # JAX
    params = EnvParams()
    jax_env = JaxRoadEnvironmentWrapper(params)
    jax_actions = do_nothing_policy_jax(jax_env)

    jax_timings = []
    jax_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = rollout(jax_env, jax_actions)

            if NUM_EPISODES == store_returns_for:
                jax_returns.append(total_reward)

        end = time.time()

        jax_timings.append(end - start)

    # Print results
    print(f"NumPy timings: {numpy_timings}")
    print(f"Jax timings: {jax_timings}")

    # Plot results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    # Timing
    ax[0].plot(experiments, numpy_timings, ".-", label="NumPy")
    ax[0].plot(experiments, jax_timings, ".-", label="Jax")

    ax[0].set_xlabel("Number of episodes")
    ax[0].set_ylabel("Time (s)")
    ax[0].legend()

    # Returns
    ax[1].hist(numpy_returns, label="NumPy")
    ax[1].hist(jax_returns, label="Jax")

    ax[1].set_xlabel("Return")
    ax[1].set_title(f"Returns for {store_returns_for} episodes")
    ax[1].legend()

    plt.show()
