# Speedups and Profiling

### Installing the dependencies

Profiling the JAX code requires `tensorboard` and `tensorboard_plugin_profile`.
The dependencies for can be found under `jax_profiling` in the pyproject.toml file. You can install them via Poetry:

```bash
poetry install --with dev,vis,jax,jax_profiling
```

The lock file is also provided for reproducibility.

### Speedups

Returns ✅ : Jax and Numpy environment are equivalent

Speedups* ⚡: Upto **24x** for ToyExample-v2, **23x** for Cologne-v1

*tested on a MacBookPro 14, 2021 with an Apple M1 Pro-chip (8 CPU cores).

You can run the code in `jax_vs_numpy.py` to see the speedups for different environments. The code will run the JAX and Numpy versions of the environment for the do-nothing policy and plot the results.

```bash
python profiling/jax_vs_numpy.py
```
  
### Environment: ToyExample-v2 | # Agents = 12

![jax_vs_numpy](https://github.com/user-attachments/assets/4d6b2469-bd54-4e7b-abcd-bceeb92684eb)

<details>
<summary>More Details</summary>

Run times (in seconds) and returns for different number of episodes (best time in green)

<img src="https://github.com/user-attachments/assets/847e9aee-1c30-4496-9eda-b7da270749a7" width="75%">
</details>

### Environment: Cologne-v1 | # Agents = 60
![jax_vs_numpy_cologne](https://github.com/user-attachments/assets/49bef040-cde6-43cc-bc73-dca9544771c3)

<details>
<summary>More Details</summary>

Run times (in seconds) and returns for different number of episodes (best time in green)

<img src="https://github.com/user-attachments/assets/587962e7-e961-4af4-8096-dea084fb83b7" width="75%">
</details>


### Profiling JAX code

Main documentation for profiling can be found in the [JAX documentation](https://docs.jax.dev/en/latest/profiling.html#profiling-computation).

We will follow the [Programmatic capture](https://docs.jax.dev/en/latest/profiling.html#programmatic-capture) method for profiling.

The `jax_profile_example.py` profiles the following code:

```python
import jax

# Run the operations to be profiled
key = jax.random.key(0)
x = jax.random.normal(key, (5000, 5000))
y = x @ x
y.block_until_ready()
```

You can generate the profile by running:

```bash
python jax_profile_example.py
```

This creates a directory `./jax-profile` with the profiling data. You can visualize the profiling data using TensorBoard:

```bash
tensorboard --logdir=./jax-profile
```

(If you cannot see any data, try using a different browser.)

Select the run under "Runs" and trace_viewer in the "Tool" option to visualize the profiling data. It should look like this:

![alt text](image.png)
