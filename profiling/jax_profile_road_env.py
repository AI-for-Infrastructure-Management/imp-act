import os
import time
import jax
import jax.numpy as jnp

from imp_act import make

ENV_NAME = "ToyExample-v2"
# ENV_NAME = "Cologne-v1-unconstrained"

jax_env = make(f"{ENV_NAME}-jax")

key = jax.random.PRNGKey(15)
key, subkey = jax.random.split(key)
obs, state = jax_env.reset(subkey)

start_time = time.time()

with jax.profiler.trace(f"{os.getcwd()}/jax-profile"):

    jit_step = jax.jit(jax_env.step)
    
    done = False
    total_reward = jnp.zeros(1)

    while not done:

        # action selection
        # take new actions for timestep
        key, rng_key, step_key = jax.random.split(key, 3)
        actions = jax.random.randint(rng_key, (jax_env.num_actions,), 0, 5)

        _, state, reward, done, info = jit_step(step_key, state, actions)

        total_reward += reward

end_time = time.time()

print(f"Time taken: {end_time - start_time}")
