"""
Based on https://github.com/omniscientoctopus/jax-imprl/blob/main/profile/jax_vs_numpy.py
"""

import itertools
import multiprocessing as mp
import time

import jax
import jax.numpy as jnp
import numpy as np

from imp_act import make


def do_nothing_policy_np(numpy_env):
    actions = []
    for edge in numpy_env.graph.es:
        acts = [0 for _ in edge["road_edge"].segments]
        actions.append(acts)
    return actions


def do_nothing_policy_jax(jax_env):
    return jnp.zeros(jax_env.total_num_segments, dtype=jnp.int32)


def numpy_rollout(env, actions):
    _ = env.reset()
    done = False

    total_reward = 0

    while not done:
        _, reward, done, _ = env.step(actions)
        total_reward += reward

    return total_reward


def parallel_numpy_rollout(env, actions, rollout_method, num_episodes, verbose=False):

    # use all cores
    cpu_count = mp.cpu_count()

    # create an iterable for the starmap
    iterable = zip(
        itertools.repeat(env, num_episodes), itertools.repeat(actions, num_episodes)
    )

    with mp.Pool(cpu_count) as pool:
        list_func_evaluations = pool.starmap(rollout_method, iterable)

        results = np.hstack(list_func_evaluations)

    return results


def round_list(lst, decimals=2):
    return [round(x, decimals) for x in lst]


if __name__ == "__main__":

    main_start = time.time()

    # experiments = [1, 10]
    experiments = [1, 10, 100, 1_000]
    ENV_NAME = "Cologne-v1"

    store_returns_for = experiments[-1]

    ############################# NUMPY ################################
    numpy_env = make(ENV_NAME)
    actions_np = do_nothing_policy_np(numpy_env)

    numpy_timings = []
    numpy_returns = []

    print('Running Numpy (for loop) ...')
    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = numpy_rollout(numpy_env, actions_np)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    #################### NUMPY (multiprocessing) #######################
    numpy_mp_timings = []
    numpy_mp_returns = []

    print('Running Numpy (multiprocessing) ...')
    for NUM_EPISODES in experiments:
        start = time.time()

        results = parallel_numpy_rollout(
            numpy_env, actions_np, numpy_rollout, NUM_EPISODES
        )

        if NUM_EPISODES == store_returns_for:
            numpy_mp_returns = results

        end = time.time()

        numpy_mp_timings.append(end - start)

    ########################## JAX (for loop) ##########################
    jax_env = make(f"{ENV_NAME}-jax")

    action_jax = do_nothing_policy_jax(jax_env)
    jax_for_loop_timings = []
    jax_for_loop_returns = []

    print('Running Jax (for loop) ...')
    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        # reset
        key, subkey = jax.random.split(key)
        obs, state = jax_env.reset(subkey)

        for _ in range(NUM_EPISODES):

            done = False
            total_reward = 0

            while not done:

                # generate keys for next timestep
                key, step_key = jax.random.split(key)
                obs, state, reward, done, info = jax_env.step(
                    step_key, state, action_jax
                )

                total_reward += reward

            if NUM_EPISODES == store_returns_for:
                jax_for_loop_returns.append(total_reward)

        end_jax = time.time()

        jax_for_loop_timings.append(end_jax - start_jax)

    ############################ JAX (scan) ############################

    import chex

    @chex.dataclass(frozen=True)
    class Runner:
        key: chex.PRNGKey
        env_state: chex.Array
        obs: chex.Array
        ep: int = 0

    def init_runner(key, env):

        # Initialize the environment
        key, env_rng = jax.random.split(key, 2)
        init_obs, env_state = env.reset(env_rng)

        return Runner(key=key, env_state=env_state, obs=init_obs)

    def update_runner(runner, metrics):

        # 1. Select action
        # currently only supports a fixed action for all episodes
        action = action_jax

        # 2. Environment step
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, done, info = jax_env.step(
            step_key, runner.env_state, action
        )

        # 3. Update metrics
        # get from info because env_state.return is reset to 0 when done
        metrics = {
            "returns": info["returns"],
            "dones": done,
        }

        # 4. Update runner state
        runner = Runner(key=key, env_state=env_state, obs=next_obs)

        return runner, metrics

    def scanned_rollout(key, episodes):

        runner = init_runner(key, jax_env)

        timesteps = jax_env.max_timesteps * episodes

        runner, metrics = jax.block_until_ready(
            jax.lax.scan(update_runner, runner, length=timesteps)
        )
        return runner, metrics

    jax_scan_timings = []
    jax_scan_returns = []

    print('Running Jax (scan) ...')
    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        runner, metrics = scanned_rollout(subkey, NUM_EPISODES)

        end_jax = time.time()

        jax_scan_timings.append(end_jax - start_jax)

        if NUM_EPISODES == store_returns_for:
            evals = metrics["returns"] * metrics["dones"]
            jax_scan_returns = evals[jnp.nonzero(evals)]

    main_end = time.time()
    print(f"Total time: {main_end - main_start}")

    ########################## Print results ###########################
    print(f"NumPy (for loop): {round_list(numpy_timings)}")
    print(f"NumPy (multiprocessing): {round_list(numpy_mp_timings)}")
    print(f"Jax (for loop): {round_list(jax_for_loop_timings)}")
    print(f"Jax (scan): {round_list(jax_scan_timings)}")

    def compare(list1, list2):
        return [l1 / l2 for l1, l2 in zip(list1, list2)]

    speedup_for_loop = compare(numpy_timings, jax_for_loop_timings)
    speedup_scan = compare(numpy_timings, jax_scan_timings)
    print("")
    print("Speedups wrt NumPy (for loop):")
    print(f"Speedup (Jax): {round_list(speedup_for_loop)}")
    print(f"Speedup (Jax scan): {round_list(speedup_scan)}")

    speedup_for_loop = compare(numpy_mp_timings, jax_for_loop_timings)
    speedup_scan = compare(numpy_mp_timings, jax_scan_timings)
    print("")
    print("Speedups wrt NumPy (multiprocessing):")
    print(f"Speedup (Jax): {round_list(speedup_for_loop)}")
    print(f"Speedup (Jax scan): {round_list(speedup_scan)}")

    # Mean returns
    print("")
    mean_numpy_returns = np.mean(numpy_returns)
    mean_numpy_mp_returns = np.mean(numpy_mp_returns)
    mean_jax_for_loop_returns = np.mean(jax_for_loop_returns)
    mean_jax_scan_returns = np.mean(jax_scan_returns).item()
    mean_list = [
        mean_numpy_returns,
        mean_numpy_mp_returns,
        mean_jax_for_loop_returns,
        mean_jax_scan_returns,
    ]
    print(f"Mean returns: {round_list(mean_list)}")

    def relative_error(x, y):
        return abs(x - y) * 100 / x

    rel_error_numpy_mp = relative_error(mean_numpy_returns, mean_numpy_mp_returns)
    rel_error_jax_for_loop = relative_error(
        mean_numpy_returns, mean_jax_for_loop_returns
    )
    rel_error_jax_scan = relative_error(mean_numpy_returns, mean_jax_scan_returns)
    _list = [rel_error_numpy_mp, rel_error_jax_for_loop, rel_error_jax_scan]
    print(f"Relative error mean returns wrt NumPy: {round_list(_list)}")

    ########################## Plot results ############################
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    # Timing
    ax[0].plot(experiments, numpy_timings, ".--", label="NumPy")
    ax[0].plot(
        experiments,
        numpy_mp_timings,
        ".--",
        label=f"NumPy (multiprocessing, {mp.cpu_count()} cores)",
    )
    ax[0].plot(experiments, jax_for_loop_timings, ".-", label="Jax (for loop)")
    ax[0].plot(experiments, jax_scan_timings, ".-", label="Jax (scan)")

    ax[0].set_xlabel("Number of episodes")
    ax[0].set_ylabel("Time (s)")
    ax[0].legend()

    # Returns
    ax[1].hist(numpy_returns, label="NumPy")
    ax[1].hist(
        numpy_mp_returns,
        label=f"NumPy (multiprocessing, {mp.cpu_count()} cores)",
        fill=False,
    )
    ax[1].hist(
        jax_for_loop_returns,
        label="Jax (for loop)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:orange",
    )
    ax[1].hist(
        jax_scan_returns,
        label="Jax (scan)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:green",
    )

    ax[1].set_xlabel("Return")
    ax[1].set_title(f"Returns for {store_returns_for} episodes")
    ax[1].legend()

    plt.show()
    fig.savefig("profiling/jax_vs_numpy.png")
