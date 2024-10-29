from .environments.recorder import Recorder
from .environments.registry import Registry
from .environments.rollout_plotter import RolloutPlotter


def make(environment_name):
    """Given a registered environment name return a fully initialized environment instance."""
    registry = Registry()
    return registry.make(environment_name)


def list_environments():
    """List all registered environments."""
    registry = Registry()
    return list(registry)
