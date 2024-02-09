import jax

from jax_environment import JaxRoadEnvironment


class JaxRoadEnvironmentWrapper:
    """Wrapper of Jax RoadEnvironment for RL libraries compatibility"""

    def __init__(self, params):
        self.jax_env = JaxRoadEnvironment(params)
        self.max_timesteps = params.max_timesteps

        key = jax.random.PRNGKey(442)
        self.step_keys, self.key = self.jax_env.split_key(key)

    def reset(self):
        obs, self.state = self.jax_env.reset_env()
        return obs

    def step(self, action):
        # Before this we might want to preprocess action to standardize to the numpy env
        obs, reward, done, info, self.state = self.jax_env.step_env(
            self.step_keys, self.state, action
        )
        self.step_keys, self.key = self.jax_env.split_key(self.key)
        return obs, reward, done, info
