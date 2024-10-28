## Overview

This directory is organized as follows:

![software architecture](https://github.com/user-attachments/assets/25639cfe-42d4-4fc3-8e89-b86482307364)

### Wrappers

<img src="wrappers.png" alt="wrappers" width="50%">

#### Recorder Wrapper

An optional wrapper to record the agent's interactions with the environment. This is useful for analyzing the agent's behavior. The Recoder wrapper is available in the `recorder.py` file, and stores the data in `rollout_data` as dict.

Usage:
```python
from imp_act import make
from imp_act.environments.recorder import Recorder

env = make("ToyExample-v2")
recorded_env = Recorder(env)

for ep in range(10):

    obs = recorded_env.reset()
    done = False

    while not done:
        actions = [[1] * len(e["road_segments"].segments) for e in env.graph.es]

        obs, reward, done, info = recorded_env.step(actions)

# convert the rollout_data to a pandas dataframe
df = recorded_env.rollout_data_to_df()
```
