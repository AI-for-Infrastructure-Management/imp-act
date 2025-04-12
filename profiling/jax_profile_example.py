"""
This example demonstrates how to use JAX's profiler to profile the execution time of specific operations. In Tensorboard, 
A. sampling: 185ms 
B. mat_mul: 472ms
C. sampling + mat_mul: 657ms

Time taken: 673ms (C + 16ms)

(MacBookPro 14, 2021)

"""

import os
import time

import jax

with jax.profiler.trace(f"{os.getcwd()}/jax-profile"):

    start_time = time.time()    
    
    # Run the operations to be profiled
    key = jax.random.PRNGKey(0)
    with jax.profiler.TraceAnnotation("sampling"):
        x = jax.random.normal(key, (5_000, 5_000))
    with jax.profiler.TraceAnnotation("mat_mul"):
        y = x @ x
    y.block_until_ready()
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")


