from functools import partial

import jax

from rookie_jax_static import RoadEnvironment


class JaxRoadEnvironmentWrapper:
    """Wrapper of Jax RoadEnvironment for RL libraries compatibility"""

    def __init__(self, params):
        self.jax_env = RoadEnvironment(params)
        self.num_segments = params.total_num_segments
        self.max_timesteps = params.max_timesteps

        key = jax.random.PRNGKey(442)
        self.step_keys, self.key = self.split_key(key, self.num_segments)

    @partial(jax.jit, static_argnums=(0, 2))
    def split_key(self, key, num_segments):
        """Split key into keys for each segment"""

        keys = jax.random.split(key, num_segments + 1)  # keys for all segments
        subkeys = keys[:num_segments, :]  # subkeys for each segment
        keys = keys[num_segments, :]  # key for next timestep

        return subkeys, keys

    def reset(self):
        obs, self.state = self.jax_env.reset_env()
        return obs

    def step(self, action):
        # Before this we might want to preprocess action to standardize to the numpy env
        obs, reward, done, info, self.state = self.jax_env.step_env(
            self.step_keys, self.state, action
        )
        self.step_keys, self.key = self.split_key(self.key, self.num_segments)
        return obs, reward, done, info
