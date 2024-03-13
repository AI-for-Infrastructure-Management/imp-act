import pytest
from imp_act import make
from imp_act.environments.config.environment_loader import EnvironmentLoader
from imp_act.environments.registry import Registry


@pytest.fixture
def toy_environment_loader():
    """Create a toy environment loader for testing."""
    toy_example_path = Registry().get("ToyExample-v1")[1]["filename"]
    return EnvironmentLoader(toy_example_path)


@pytest.fixture
def toy_environment():
    """Create a toy environment loader for testing."""
    return make("ToyExample-v1")


@pytest.fixture
def small_environment():
    return make("Montenegro-v1")


@pytest.fixture
def medium_environment():
    return make("Denmark-v1")


@pytest.fixture
def large_environment():
    return make("Belgium-v1")


@pytest.fixture
def parameter_fixture(request):
    return request.getfixturevalue(request.param)
