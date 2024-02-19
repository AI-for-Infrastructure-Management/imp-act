import jax
import jax.numpy as jnp

from environments.config.environment_loader import EnvironmentLoader

path = "environments/config/environment_presets/toy_environment.yaml"

# numpy_env = EnvironmentLoader(path).to_numpy()

env = EnvironmentLoader(path).to_jax()

_action = [{"0": [0, 0]}, {"1": [0, 0]}, {"2": [0, 0]}, {"3": [0, 0]}]
__action = jax.tree_util.tree_leaves(_action)
action = jnp.array(__action, dtype=jnp.uint8)

key = jax.random.PRNGKey(442)
step_keys, key = env.split_key(key)

# reset
obs, state = env.reset_env()

done = False
total_rewards = 0.0

# rollout
while not done:
    # step
    obs, reward, done, info, state = env.step_env(step_keys, state, action)

    # generate keys for next timestep
    step_keys, key = env.split_key(key)

    # update total rewards
    total_rewards += reward

print(f"Total rewards: {total_rewards}")
print(f"Total travel time: {info['total_travel_time']}")
