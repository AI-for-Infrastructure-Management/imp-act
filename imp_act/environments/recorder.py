"""
The Recorder class is used to record the interactions of the agent with
the environment. It records the observations, actions, rewards, done
flags etc. in "rollout_data". It is a wrapper around the environment.

The rollout_data is a dict for 3 reasons:
    - Each episode can have varying keys,
        for example, if there is an evaluation at that episode etc.
    - Easy manipulation using pandas,
    - Easy conversion to csv for storage where it is human readable

"""

import pandas as pd


class Recorder:
    def __init__(self, env):
        self.env = env
        self.episode = -1  # because we increment it at the beginning of reset
        self.rollout_data = {}
        self.record_exclude_keys = ["adjacency_matrix"]

    def reset(self):
        self.episode += 1

        obs = self.env.reset()

        # Record
        self.rollout_data[self.episode] = {}
        self.record_dict(obs)
        self.record("edge_states", self.env._get_states())

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Record
        self.record("action", action)
        self.record_dict(obs)
        self.record("reward", reward)
        self.record("done", done)
        self.record_dict(info)

        return obs, reward, done, info

    def record(self, key, value):
        if key not in self.rollout_data[self.episode]:
            self.rollout_data[self.episode][key] = [value]
        else:
            self.rollout_data[self.episode][key].append(value)

    def record_dict(self, value):
        for k, v in value.items():
            if k not in self.record_exclude_keys:
                self.record(k, v)

    def rollout_data_to_df(self):
        return pd.DataFrame(self.rollout_data).T

    def save_rollout_data(self, path):
        self.rollout_data_to_df().to_csv(path)

    def load_rollout_data(self, path):
        self.rollout_data = pd.read_csv(path).to_dict()
