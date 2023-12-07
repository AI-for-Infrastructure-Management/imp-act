import igraph as ig
from igraph import Graph
import numpy as np

class RoadSegment():
    def __init__(self):
        # state [0-3]
        self.initial_observation = 0
        self.number_of_states = 4
        self.transition_tables = np.array([
             [# Action 0: do-nothing
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
                [0.0, 0.0, 0.8, 0.2],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [# Action 1: inspect
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
                [0.0, 0.0, 0.8, 0.2],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [# Action 2: minor repair
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
                [0.0, 0.0, 0.8, 0.2]
            ],
            [# Action 3: major repair (replacement)
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ]
        ])

        self.observation_tables = np.array([
            [# Action 0: do-nothing
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 1: inspect
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.8, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 2: minor repair
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 3: major repair (replacement)
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ])
        
        # Costs (negative rewards)
        self.state_action_cost = np.array([
            [0, -1, -20, -150],
            [0, -1, -25, -150],
            [0, -1, -30, -150],
            [0, -1, -40, -150],
        ])

        self.reset()

    def reset(self):
        self.state = 0
        self.observation = self.initial_observation

    def step(self, action):
        # actions: [do_nothing, inspect, repair] = [0, 1, 2]
        next_deterioration_state = np.random.choice(
            np.arange(self.number_of_states), p=self.transition_tables[action][self.state]
        )

        cost = self.state_action_cost[self.state][action]
        self.state = next_deterioration_state

        self.observation = np.random.choice(
            np.arange(self.number_of_states), p=self.observation_tables[action][self.state]
        )

        #TODO: Believe state computation

        return cost
    
    def compute_travel_time(self, action):
        return 0 # travel_time

class RoadEdge():
    def __init__(self, number_of_segments):
        self.number_of_segments = number_of_segments
        self.inspection_campain_cost = 1
        self.edge_travel_time = 200
        self.segments = [RoadSegment() for _ in range(number_of_segments)]

    def get_edge_travel_time(self):
        return self.edge_travel_time

    def update_edge_travel_time(self, actions):
        pass # TODO

    def step(self, actions):
        # states:
        cost = 0
        for segment, action in zip(self.segments, actions):
            segment_cost = segment.step(action)
            cost += segment_cost

        if 1 in actions:
            cost += self.inspection_campain_cost

        self.update_edge_travel_time(actions)

        return cost

    def reset(self):
        for segment in self.segments:
            segment.reset()

    def get_observation(self):
        return [segment.observation for segment in self.segments]

    def get_states(self):
        return [segment.state for segment in self.segments]

class RoadEnvironment():
    def __init__(self):
        self.max_timesteps = 50
        self.travel_time_factor = 1
        self.graph = Graph()
        self.graph.add_vertices(4)
        self.graph.add_edges([(0,1), (1,2), (2,3), (3,0)])
        for edge in self.graph.es:
            edge["road_segments"] = RoadEdge(number_of_segments=2)

    def reset(self):
        self.timestep = 0
        for edge in self.graph.es:
            edge["road_segments"].reset()

    def _get_observation(self):
        adjacency_matrix = np.array(self.graph.get_adjacency().data)
        edge_observations = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_segments"].get_observation()) # add edge from and to

        observations = {"adjacency_matrix": adjacency_matrix, "edge_observations": edge_observations}

        return observations
    
    def _get_states(self):
        edge_states = []
        for edge in self.graph.es:
            edge_states.append(edge["road_segments"].get_states())

        return edge_states
        
    def _get_travel_time_cost(self):
        # compute travel time
        # compute cost of travel time 
        return 0 # TODO

    def step(self, actions):
        total_cost = 0
        for i, edge in enumerate(self.graph.es):
            total_cost += edge["road_segments"].step(actions[i])

        travel_time_cost = self._get_travel_time_cost()

        cost = total_cost + self.travel_time_factor * travel_time_cost

        observation = self._get_observation()

        self.timestep += 1

        info = {"states": self._get_states()}

        return observation, cost, self.timestep >= self.max_timesteps, info
        
    
