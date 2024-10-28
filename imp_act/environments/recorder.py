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
        self.init_recorder()

    def init_recorder(self):
        self.time_step = 0
        self.episode = -1  # because we increment it at the beginning of reset

        self.rollout_data = {}

    def reset(self):
        self.time_step = 0
        self.episode += 1

        obs = self.env.reset()

        # Record
        self.rollout_data[self.episode] = {
            # lists with 'max_timesteps+1' elements
            "time_step": [self.time_step],
            "edge_states": [self.env._get_states()],
            "edge_observations": [obs["edge_observations"]],
            "edge_deterioration_rates": [obs["edge_deterioration_rates"]],
            "edge_beliefs": [obs["edge_beliefs"]],
            # lists with 'max_timesteps' elements
            "action": [],
            "reward": [],
            "done": [],
            "total_travel_time": [],
            "travel_times": [],
            "volumes": [],
            "reward_elements": [],
            "actions_taken": [],
        }

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Record
        self.record("time_step", self.time_step)
        self.record("edge_states", info["states"])
        self.record("edge_observations", obs["edge_observations"])
        self.record("edge_deterioration_rates", obs["edge_deterioration_rates"])
        self.record("edge_beliefs", obs["edge_beliefs"])
        self.record("action", action)
        self.record("reward", reward)
        self.record("done", done)
        self.record("total_travel_time", info["total_travel_time"])
        self.record("travel_times", info["travel_times"])
        self.record("volumes", info["volumes"])
        self.record("reward_elements", info["reward_elements"])
        self.record("actions_taken", info["actions_taken"])

        self.time_step += 1

        return obs, reward, done, info

    def record(self, key, value):
        self.rollout_data[self.episode][key].append(value)

    def rollout_data_to_df(self):
        return pd.DataFrame(self.rollout_data).T

    def save_rollout_data(self, path):
        self.rollout_data_to_df().to_csv(path)

    def load_rollout_data(self, path):
        self.rollout_data = pd.read_csv(path).to_dict()
