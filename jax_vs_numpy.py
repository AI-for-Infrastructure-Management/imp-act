import time

import jax
import jax.numpy as jnp

from environment import RoadEnvironment as NumPyRoadEnvironment
from environment_presets import small_environment_dict

from jax_environment import JaxRoadEnvironment
from params import EnvParams


def do_nothing_policy(env, obs):
    return [[0] * segments for segments in env.edge_segments_numbers]


def numpy_rollout(env, policy):
    obs = env.reset()
    done = False

    total_reward = 0

    while not done:
        actions = policy(env, obs["edge_observations"])
        next_obs, reward, done, _ = env.step(actions)

        obs = next_obs
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    experiments = [1, 10, 100, 1_000, 10_000]

    store_returns_for = experiments[-1]

    # NUMPY
    env = NumPyRoadEnvironment(**small_environment_dict)

    numpy_timings = []
    numpy_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = numpy_rollout(env, do_nothing_policy)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    # JAX
    params = EnvParams()
    jax_env = JaxRoadEnvironment(params)

    _action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)

    jax_timings = []
    jax_returns = []

    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(442)
        step_keys, key = jax_env.split_key(key)

        # rollout
        for _ in range(NUM_EPISODES):
            # reset
            obs, state = jax_env.reset_env()

            total_reward = 0

            # rollout
            for _ in range(jax_env.max_timesteps):
                # step
                obs, reward, done, info, state = jax_env.step_env(
                    step_keys, state, action
                )

                step_keys, key = jax_env.split_key(key)

                total_reward += reward

            if NUM_EPISODES == store_returns_for:
                jax_returns.append(total_reward)

        end_jax = time.time()

        jax_timings.append(end_jax - start_jax)

    # Print results
    print(f"NumPy timings: {numpy_timings}")
    print(f"Jax static timings: {jax_timings}")

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
