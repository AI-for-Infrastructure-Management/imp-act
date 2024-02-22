import pytest
from environments.config.environment_loader import EnvironmentLoader
from environments.registry import Registry


@pytest.fixture
def toy_environment_loader():
    """Create a toy environment loader for testing."""
    toy_environment_path = Registry().get("toy_environment_numpy")[1]["filename"]
    return EnvironmentLoader(toy_environment_path)


@pytest.fixture
def toy_environment():
    """Create a toy environment loader for testing."""
    return Registry().make("toy_environment_numpy")


@pytest.fixture
def small_environment():
    return Registry().make("small_environment_numpy")


@pytest.fixture
def large_environment():
    return Registry().make("large_environment_numpy")


@pytest.fixture
def parameter_fixture(request):
    return request.getfixturevalue(request.param)
