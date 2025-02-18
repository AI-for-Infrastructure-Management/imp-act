from time import time

import pytest

from imp_act import make
from imp_act.environments.environment_loader import EnvironmentLoader
from imp_act.environments.registry import Registry


@pytest.fixture
def toy_environment_loader():
    """Create a toy environment loader for testing."""
    toy_example_path = Registry().get("ToyExample-v1")[1]["filename"]
    return EnvironmentLoader(toy_example_path)


@pytest.fixture
def toy_environment_1():
    """Create a toy environment loader for testing."""
    return make("ToyExample-v1")


@pytest.fixture
def toy_environment_2():
    """Create a toy environment loader for testing."""
    return make("ToyExample-v2")


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
def cologne_environment():
    return make("Cologne-v1")

@pytest.fixture
def cologne_bonn_dusseldorf_environment():
    return make("CologneBonnDusseldorf-v1")

@pytest.fixture
def random_time_seed():
    return int(time())


@pytest.fixture
def parameter_fixture(request):
    return request.getfixturevalue(request.param)


def load_test_env(name):
    return EnvironmentLoader(
        f"tests/test_environment_configs/{name}/{name}.yaml"
    ).to_numpy()


@pytest.fixture
def stationary_deterioration_environment():
    return load_test_env("stationary_deterioration")
