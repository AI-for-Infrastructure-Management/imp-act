import time

import jax
import jax.numpy as jnp

from environments.config.environment_loader import EnvironmentLoader


def do_nothing_policy_np(edge_segments_numbers):
    return [[0] * segments for segments in edge_segments_numbers]


def do_nothing_policy_jax(edge_segments_numbers):
    action_dict = {}
    for i, n_segments in enumerate(edge_segments_numbers):
        action_dict[f"{i}"] = [0] * n_segments
    _action = [action_dict]
    __action = jax.tree_util.tree_leaves(_action)
    action = jnp.array(__action, dtype=jnp.uint8)
    return action


def numpy_rollout(env, edge_segments_numbers, actions):
    _ = env.reset()
    done = False

    total_reward = 0

    while not done:
        _, reward, done, _ = env.step(actions)
        total_reward += reward

    return total_reward


def compute_edge_segments_numbers_np(env):
    edge_segments_numbers_np = []
    for edge in env.graph.es:
        edge_segments_numbers_np.append(edge["road_segments"].number_of_segments)
    return edge_segments_numbers_np


def compute_edge_segments_numbers_jax(env):
    edge_segments_numbers_jax = []
    for edge in env.segments_list:
        edge_segments_numbers_jax.append(len(edge))
    return edge_segments_numbers_jax


def main(filename):
    experiments = [1, 10, 100, 1000]

    store_returns_for = experiments[-1]

    env_loader = EnvironmentLoader(filename)

    # NUMPY
    env = env_loader.to_numpy()
    edge_segments_numbers_np = compute_edge_segments_numbers_np(env)

    actions_np = do_nothing_policy_np(edge_segments_numbers_np)

    numpy_timings = []
    numpy_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = numpy_rollout(env, edge_segments_numbers_np, actions_np)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    # JAX
    jax_env = env_loader.to_jax()
    edge_segments_numbers_jax = compute_edge_segments_numbers_jax(jax_env)

    action_jax = do_nothing_policy_jax(edge_segments_numbers_jax)
    jax_timings = []
    jax_returns = []

    key = jax.random.PRNGKey(442)

    for NUM_EPISODES in experiments:

        start_jax = time.time()

        step_keys, key = jax_env.split_key(key)

        # rollout
        for _ in range(NUM_EPISODES):
            # reset
            _, state = jax_env.reset_env()

            total_reward = 0

            # rollout
            for _ in range(jax_env.max_timesteps):
                # step
                _, reward, _, _, state = jax_env.step_env(step_keys, state, action_jax)

                step_keys, key = jax_env.split_key(key)

                total_reward += reward

            if NUM_EPISODES == store_returns_for:
                jax_returns.append(total_reward)

        end_jax = time.time()

        jax_timings.append(end_jax - start_jax)

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
    plt.savefig(f"profiling/{filename.split('/')[-1].split('.')[0]}_jax_vs_numpy.png")


if __name__ == "__main__":
    paths = [
        "environments/config/environment_presets/toy_environment.yaml",
        "environments/config/environment_presets/small_environment.yaml",
        "environments/config/environment_presets/large_environment.yaml",
    ]
    for filename in paths:
        print(filename)
        main(filename)
