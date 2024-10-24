"""
The Recorder class is used to record the interactions of the agent with 
the environment. It records the observations, actions, rewards, done flags,
in a "tape". It is a wrapper around the environment.

The tape is a dict for 3 reasons:
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

        self.tape = {}

    def reset(self):
        self.time_step = 0
        self.episode += 1

        obs = self.env.reset()

        # Record
        self.tape[self.episode] = {
            "obs": [obs],
            "action": [],
            "reward": [],
            "done": [],
            "info": [],
        }

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Record
        self.record("obs", obs)
        self.record("action", action)
        self.record("reward", reward)
        self.record("done", done)
        self.record("info", info)

        self.time_step += 1

        return obs, reward, done, info

    def record(self, key, value):
        self.tape[self.episode][key].append(value)

    def tape_to_df(self):
        return pd.DataFrame(self.tape).T

    def save_tape(self, path):
        self.tape_to_df().to_csv(path)

    def load_tape(self, path):
        self.tape = pd.read_csv(path).to_dict()
