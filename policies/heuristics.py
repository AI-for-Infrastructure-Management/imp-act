from .agent import Agent

class SimpleHeuristicAgent(Agent):
    """Agent that uses a simple heuristic to choose actions."""
    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def get_action(self, observation):
        edge_beliefs = observation["edge_beliefs"]
        actions = []
        for edge_belief in edge_beliefs:
            edge_actions = []
            for segment_belief in edge_belief:
                if segment_belief[0] < self.threshold:
                    edge_actions.append(1)
                else:
                    edge_actions.append(0)
            actions.append(edge_actions)
        return actions

    def reset(self, observation):
        pass