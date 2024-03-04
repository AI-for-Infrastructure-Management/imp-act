from .environments.registry import Registry


def get_environment(environment_name):
    """Given a registered environment name return a fully initialized environment instance."""
    registry = Registry()
    return registry.make(environment_name)


def list_environments():
    """List all registered environments."""
    registry = Registry()
    return list(registry)
