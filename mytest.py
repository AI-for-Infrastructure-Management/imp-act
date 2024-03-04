from tests.test_environment import *

env = EnvironmentLoader(
    "environments/config/environment_presets/small_environment.yaml"
).to_numpy()
