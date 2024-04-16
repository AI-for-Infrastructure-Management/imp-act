from .config.environment_loader import EnvironmentLoader
from .registry import Registry


def numpy_environment_loader(filename):
    """Create a numpy environment loader for testing."""
    return EnvironmentLoader(filename).to_numpy()


def jax_environment_loader(filename):
    """Create a jax environment loader for testing."""
    return EnvironmentLoader(filename).to_jax()


environment_path = __path__[0]

Registry().register(
    name="ToyExample-v1",
    cls=numpy_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/ToyExample-v1.yaml"
    },
)

Registry().register(
    name="ToyNetwork-v1",
    cls=numpy_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/ToyNetwork-v1.yaml"
    },
)

Registry().register(
    name="Montenegro-v1",
    cls=numpy_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Montenegro-v1.yaml"
    },
)

Registry().register(
    name="Denmark-v1",
    cls=numpy_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Denmark-v1.yaml"
    },
)

Registry().register(
    name="Belgium-v1",
    cls=numpy_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Belgium-v1.yaml"
    },
)


Registry().register(
    name="ToyExample-v1-jax",
    cls=jax_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/ToyExample-v1.yaml"
    },
)

Registry().register(
    name="Montenegro-v1-jax",
    cls=jax_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Montenegro-v1.yaml"
    },
)

Registry().register(
    name="Denmark-v1-jax",
    cls=jax_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Denmark-v1.yaml"
    },
)

Registry().register(
    name="Belgium-v1-jax",
    cls=jax_environment_loader,
    parameters={
        "filename": f"{environment_path}/config/environment_presets/Belgium-v1.yaml"
    },
)
