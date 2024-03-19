# Quickstart


**Example rollout with a simple policy**

```python
from imp_act import make

# Environment: Montenegro
# #nodes: 8 #edges: 7 #segments: 55
env = make("Montenegro-v1")

# do-nothing policy 
# (environment forces segment reconstruction if failure is observed)
def do_nothing_policy(env, obs):
    edge_obs = obs['edge_observations']
    return [[0 for obs in e] for e in edge_obs]

# reset
obs = env.reset()
done = False
total_reward = 0

while not done:

    # select action
    actions = do_nothing_policy(env, obs)

    # environment step
    next_obs, reward, done, info = env.step(actions)

    total_reward += reward

    obs = next_obs

```