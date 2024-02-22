from .config.environment_loader import EnvironmentLoader
from .registry import Registry


def numpy_environment_loader(filename):
    """Create a numpy environment loader for testing."""
    return EnvironmentLoader(filename).to_numpy()


def jax_environment_loader(filename):
    """Create a jax environment loader for testing."""
    return EnvironmentLoader(filename).to_jax()


Registry().register(
    name="toy_environment_numpy",
    cls=numpy_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/toy_environment.yaml"
    },
)

Registry().register(
    name="small_environment_numpy",
    cls=numpy_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/small_environment.yaml"
    },
)

Registry().register(
    name="large_environment_numpy",
    cls=numpy_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/large_environment.yaml"
    },
)


Registry().register(
    name="toy_environment_jax",
    cls=jax_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/toy_environment.yaml"
    },
)

Registry().register(
    name="small_environment_jax",
    cls=jax_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/small_environment.yaml"
    },
)

Registry().register(
    name="large_environment_jax",
    cls=jax_environment_loader,
    parameters={
        "filename": "environments/config/environment_presets/large_environment.yaml"
    },
)
