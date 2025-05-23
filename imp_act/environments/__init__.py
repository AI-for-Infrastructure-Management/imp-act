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
    "ToyExample-v2",
    "ToyExample-v2-unconstrained",
    "ToyExample-v2-only-maintenance",
    "Cologne-v1",
    "Cologne-v1-unconstrained",
    "Cologne-v1-only-maintenance",
    "CologneBonnDusseldorf-v1",
    "CologneBonnDusseldorf-v1-unconstrained",
    "CologneBonnDusseldorf-v1-only-maintenance",
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
