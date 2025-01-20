from .environment_loader import EnvironmentLoader
from .registry import Registry


def numpy_environment_loader(filename):
    """Create a numpy environment loader for testing."""
    return EnvironmentLoader(filename).to_numpy()


def jax_environment_loader(filename):
    """Create a jax environment loader for testing."""
    return EnvironmentLoader(filename).to_jax()


environment_path = __path__[0]

presets = [
    "ToyExample-v1",
    "ToyExample-v2",
    "Montenegro-v1",
    "Denmark-v1",
    "Belgium-v1",
    "Cologne-v1",
]

for name in presets:

    # Numpy environments
    Registry().register(
        name=name,
        cls=numpy_environment_loader,
        parameters={"filename": f"{environment_path}/presets/{name}/{name}.yaml"},
    )

    # JAX environments
    Registry().register(
        name=f"{name}-jax",
        cls=jax_environment_loader,
        parameters={"filename": f"{environment_path}/presets/{name}/{name}.yaml"},
    )
