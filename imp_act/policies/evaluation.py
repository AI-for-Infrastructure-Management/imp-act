import itertools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_episode(agent, environment, collect_output=False, seed=None):
    total_reward = 0
    observations, actions, rewards, infos = [], [], [], []
    done = False
    if seed is not None:
        environment.seed(seed)
    observation = environment.reset()
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, done, info = environment.step(action)
        if collect_output:
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
        total_reward += reward
        observation = next_observation

    if collect_output:
        output = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
        }
    else:
        output = None

    return total_reward, output


def evaluate_agent(agent, environment, number_of_episodes):
    rewards = []
    for _ in range(number_of_episodes):
        reward, _ = collect_episode(agent, environment)
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards)


def evaluate_heuristic(
    heuristic_class, environment, parameter_dict, number_of_episodes
):
    heuristic_agent = heuristic_class(**parameter_dict)
    reward_mean, reward_std = evaluate_agent(
        heuristic_agent, environment, number_of_episodes
    )
    return reward_mean, reward_std, parameter_dict


def evaluate_heursitic_grid(
    heuristic_class,
    environment,
    parameter_dict,
    number_of_episodes,
    number_of_processes=32,
    result_path=None,
    overwrite=False,
):
    pool = mp.Pool(processes=number_of_processes)

    parameter_keys = list(parameter_dict.keys())
    parameter_list = [
        dict(zip(parameter_dict, v))
        for v in itertools.product(*parameter_dict.values())
    ]

    if result_path is not None and overwrite is False and os.path.exists(result_path):
        results = pd.read_csv(result_path)  # load previous results if possible
        if not results.loc[0]["episodes"] != number_of_episodes:
            parameter_list = parameter_list[
                len(results) :
            ]  # only evaluate new parameters
        else:
            raise ValueError(
                "The result file already exists and has a different number of episodes. Set overwrite=True to overwrite the file."
            )
    else:
        columns = parameter_keys + ["reward_mean", "reward_std", "episodes"]
        results = pd.DataFrame(columns=columns)

    jobs = [
        pool.apply_async(
            evaluate_heuristic,
            args=(heuristic_class, environment, parameters, number_of_episodes),
        )
        for parameters in parameter_list
    ]
    with tqdm(total=len(jobs)) as pbar:
        for job in jobs:
            pbar.update(1)
            reward_mean, reward_std, parameters = job.get()
            results.loc[len(results)] = [parameters[k] for k in parameter_keys] + [
                reward_mean,
                reward_std,
                number_of_episodes,
            ]
            if result_path is not None:
                results.to_csv(result_path, index=False)
            pbar.set_description(
                f"{' '.join([f'{k}: {v}' for k,v in parameters.items()])} - Avg: {reward_mean:.2f}, Std: {reward_std:.2f}"
            )
    pool.close()

    return results
