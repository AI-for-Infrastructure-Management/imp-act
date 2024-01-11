class Agent():
    """Base class for all agents."""
    def __init__(self):
        raise NotImplementedError

    def get_action(self, observation):
        """Returns an action given the state."""
        raise NotImplementedError
    
    def reset(self, observation):
        """Resets the agent."""
        raise NotImplementedError