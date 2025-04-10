"""
Based on https://github.com/omniscientoctopus/jax-imprl/blob/main/profile/jax_vs_numpy.py
"""

import itertools
import multiprocessing as mp
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from imp_act import make
from tabulate import tabulate


def do_nothing_policy_np(numpy_env):
    actions = []
    for edge in numpy_env.graph.es:
        acts = [0 for _ in edge["road_edge"].segments]
        actions.append(acts)
    return actions


def do_nothing_policy_jax(jax_env):
    return jnp.zeros(jax_env.total_num_segments, dtype=jnp.int32)


def numpy_rollout(env, actions):
    env.seed(None)
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


def print_formatted_timings(timings, baseline=None):

    baseline = timings[baseline]
    rel_timings = {key: baseline / timings[key] for key in timings}

    df = pd.DataFrame(index=timings.keys(), columns=experiments)

    # formatting
    GREEN, RESET = "\033[1;32m", "\033[0m"

    for i, experiment in enumerate(experiments):
        _best_key = min(timings, key=lambda x: timings[x][i])
        for key in timings.keys():
            value = f"{timings[key][i]:.1f} ({rel_timings[key][i]:.1f}x)"

            # Color best key
            if key == _best_key:
                value = f"{GREEN}{value}{RESET}"

            df.loc[key, experiment] = value

    # Rename the first column header to "Episodes"
    df.index.name = "Episodes"
    print(tabulate(df, headers="keys", tablefmt="pretty", stralign="right"))


def print_formatted_returns(store_returns_for, mean_returns):

    baseline = mean_returns["numpy"]
    rel_error = {
        key: (mean_returns[key] - baseline) * 100 / baseline for key in mean_returns
    }

    df = pd.DataFrame(
        index=mean_returns.keys(), columns=["Mean return (M)", "Relative error"]
    )

    print("")
    print(f"Mean returns (in millions) for {store_returns_for} episodes")
    for key in mean_returns.keys():
        df.loc[key, "Mean return (M)"] = f"{mean_returns[key]:.2f}"
        df.loc[key, "Relative error"] = f"{rel_error[key]:.2f} %"

    print(tabulate(df, headers="keys", tablefmt="pretty", stralign="right"))


if __name__ == "__main__":

    main_start = time.time()

    # experiments = [1, 10]
    experiments = [1, 10, 100, 1000]
    #ENV_NAME = "ToyExample-v2"
    #ENV_NAME = "Cologne-v1"
    ENV_NAME = "Cologne-v1-unconstrained"

    print(f"Environment: {ENV_NAME}")

    store_returns_for = experiments[-1]

    ############################# NUMPY ################################
    numpy_env = make(ENV_NAME)
    actions_np = do_nothing_policy_np(numpy_env)

    numpy_timings = []
    numpy_returns = []

    print("Running: Numpy (for loop),", end=" ")
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

    cpu_count = mp.cpu_count()

    print("Numpy (multiprocessing),", end=" ")
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

    print("Jax (for loop),", end=" ")
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

    print("Jax (scan)")
    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        key, key_ = jax.random.split(key)

        runner, metrics = scanned_rollout(key_, NUM_EPISODES)

        end_jax = time.time()

        jax_scan_timings.append(end_jax - start_jax)

        if NUM_EPISODES == store_returns_for:
            evals = metrics["returns"] * metrics["dones"]
            jax_scan_returns = evals[jnp.nonzero(evals)]

    ########################## JAX VMAP + SCAN ################################
    jax_vmap_timings = []
    jax_vmap_returns = []
    print("Jax (vmap + scan)")
    for NUM_EPISODES in experiments:
        key, key_ = jax.random.split(key)
        keys = jax.random.split(key_, NUM_EPISODES)
        start_jax_= time.time()
        runners, metrics = jax.block_until_ready(
            jax.vmap(jax.jit(scanned_rollout, static_argnums=(1)), in_axes=(0, None))(keys, 1)
        )

        end_jax_ = time.time()
        jax_vmap_timings.append(end_jax_ - start_jax_)

        if NUM_EPISODES == store_returns_for:
            evals = metrics["returns"] * metrics["dones"]
            jax_vmap_returns = evals[jnp.nonzero(evals)]


    main_end = time.time()
    print(f"Total time: {main_end - main_start:.1f} s")


    ########################## Print results ###########################

    ####### Timings #######
    timings = {
        "numpy": np.array(numpy_timings),
        "numpy_mp": np.array(numpy_mp_timings),
        "jax_for_loop": np.array(jax_for_loop_timings),
        "jax_scan": np.array(jax_scan_timings),
        "jax_vmap": np.array(jax_vmap_timings),
    }

    print("")
    print("Timings (s) for different number of episodes")
    print("(best time in green)")
    print("")
    print("(wrt NumPy)")
    print_formatted_timings(timings, baseline="numpy")

    print("")
    print(f"(wrt NumPy multiprocessing with {cpu_count} cores)")
    print_formatted_timings(timings, baseline="numpy_mp")

    ######## Mean returns ########
    normalizing_constant = 1_000_000
    mean_returns = {
        "numpy": np.mean(numpy_returns) / normalizing_constant,
        "numpy_mp": np.mean(numpy_mp_returns) / normalizing_constant,
        "jax_for_loop": np.mean(jax_for_loop_returns) / normalizing_constant,
        "jax_scan": np.mean(jax_scan_returns).item() / normalizing_constant,
        "jax_vmap": np.mean(jax_vmap_returns).item() / normalizing_constant,
    }

    print_formatted_returns(store_returns_for, mean_returns)

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
    ax[0].plot(experiments, jax_vmap_timings, ".-", label="Jax (vmap + scan)")

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

    ax[1].hist(
        jax_vmap_returns,
        label="Jax (vmap + scan)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:red",
    )

    ax[1].set_xlabel("Return")
    ax[1].set_title(f"Returns for {store_returns_for} episodes")
    ax[1].legend()

    plt.show()
    # fig.savefig("./jax_vs_numpy.png")
