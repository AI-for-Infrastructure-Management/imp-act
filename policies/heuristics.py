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
                    edge_actions.append(2)  # minor repair
                else:
                    edge_actions.append(0)
            actions.append(edge_actions)
        return actions

    def reset(self, observation):
        pass


class TCBMHeuristicAgent(Agent):
    """Time condition based maintenance heuristic agent."""

    def __init__(self, threshold=0.15, inspection_interval=10):
        self.threshold = threshold
        self.inspection_interval = inspection_interval
        self.time_step = 0

    def get_action(self, observation):
        if observation.get("time_step") is not None:
            self.time_step = observation["time_step"]
        else:
            self.time_step += 1

        edge_beliefs = observation["edge_beliefs"]
        actions = []
        for edge_belief in edge_beliefs:
            edge_actions = []
            for segment_belief in edge_belief:
                if segment_belief[0] < self.threshold:
                    edge_actions.append(2)  # minor repair
                else:
                    if (
                        self.time_step > 0
                        and self.time_step % self.inspection_interval == 0
                    ):
                        edge_actions.append(1)
                    else:
                        edge_actions.append(0)
            actions.append(edge_actions)

        return actions

    def reset(self, observation):
        self.time_step = 0
