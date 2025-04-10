import os
import time

import jax

start_time = time.time()

with jax.profiler.trace(f"{os.getcwd()}/jax-profile"):

    # Run the operations to be profiled
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5_000, 5_000))
    y = x @ x
    y.block_until_ready()

end_time = time.time()

print(f"Time taken: {end_time - start_time}")
