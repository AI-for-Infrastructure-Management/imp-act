import pytest
from environments.config.environment_loader import EnvironmentLoader


@pytest.fixture
def toy_environment_path():
    """Path to toy environment file"""
    return "environments/config/environment_presets/toy_environment.yaml"


@pytest.fixture
def toy_environment_loader(toy_environment_path):
    """Create a toy environment loader for testing."""
    return EnvironmentLoader(toy_environment_path)


@pytest.fixture
def toy_environment(toy_environment_loader):
    """Create a toy environment loader for testing."""
    env = toy_environment_loader.to_numpy()
    return env


@pytest.fixture
def small_environment_path():
    """Path to small environment file"""
    return "environments/config/environment_presets/small_environment.yaml"


@pytest.fixture
def small_environment(small_environment_path):
    loader = EnvironmentLoader(small_environment_path)
    return loader.to_numpy()


@pytest.fixture
def large_environment_path():
    """Path to large environment file"""
    return "environments/config/environment_presets/large_environment.yaml"


@pytest.fixture
def large_environment(large_environment_path):
    loader = EnvironmentLoader(large_environment_path)
    return loader.to_numpy()


@pytest.fixture
def parameter_fixture(request):
    return request.getfixturevalue(request.param)
