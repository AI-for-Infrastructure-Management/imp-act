from pathlib import Path

from .environment_loader import EnvironmentLoader
from .registry import Registry


def numpy_environment_loader(filename):
    """Create a numpy environment loader for testing."""
    return EnvironmentLoader(filename).to_numpy()


def jax_environment_loader(filename):
    """Create a jax environment loader for testing."""
    return EnvironmentLoader(filename).to_jax()


environment_path = Path(__file__).parent
presets_root = environment_path / "presets"

# get all preset YAMLs
for preset_dir in sorted(presets_root.iterdir()):
    if not preset_dir.is_dir() or preset_dir.name == "common":
        continue

    # Loop through all YAML files in each map directory
    for yaml_file in sorted(preset_dir.glob("*.yaml")):
        name = yaml_file.stem  # e.g. "Cologne-v1-unconstrained"
        yaml_path = str(yaml_file.resolve())

        # Numpy environments
        Registry().register(
            name=name,
            cls=numpy_environment_loader,
            parameters={"filename": yaml_path},
        )

        # JAX environments
        Registry().register(
            name=f"{name}-jax",
            cls=jax_environment_loader,
            parameters={"filename": yaml_path},
        )
