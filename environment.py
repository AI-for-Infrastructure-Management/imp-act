import igraph as ig
from igraph import Graph
import numpy as np

class RoadSegment():
    def __init__(self):
        # state [0-3]
        self.capacity = 500 # maybe cars per minute
        self.base_travel_time = 50 # maybe minutes it takes to travel trough a segment
        self.initial_observation = 0 # 
        self.number_of_states = 4
        self.transition_tables = np.array([
             [# Action 0: do-nothing
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [# Action 1: inspect
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0]
            ],
            [# Action 2: minor repair
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.2, 0.1, 0.0]
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
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 1: inspect
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 2: minor repair
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [# Action 3: major repair (replacement)
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [1/3, 1/3, 1/3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ])
        
        # Costs (negative rewards)
        self.state_action_reward = np.array([
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
        # actions: [do_nothing, inspect, minor repair, replacement] = [0, 1, 2, 3]
        
        if self.observation == 3:
            action = 3 # force replacement
        
        next_deterioration_state = np.random.choice(
            np.arange(self.number_of_states), p=self.transition_tables[action][self.state]
        )

        reward = self.state_action_reward[self.state][action]
        self.state = next_deterioration_state

        self.observation = np.random.choice(
            np.arange(self.number_of_states), p=self.observation_tables[action][self.state]
        )

        #TODO: Belief state computation

        return reward
    
    def compute_travel_time(self, action):
        return 0 # travel_time

class RoadEdge():
    def __init__(self, number_of_segments):
        self.number_of_segments = number_of_segments
        self.inspection_campain_cost = -5
        self.edge_travel_time = 200
        self.segments = [RoadSegment() for _ in range(number_of_segments)]
        self.reset()
    
    # Define a function for calculating BPR travel times based on volume and capacity
    def calculate_bpr_travel_time(volume, capacity, base_time, alpha=0.15, beta=4):
        return base_time * (1 + alpha * (volume / capacity)**beta)
    
    def calculate_bpr_capacity_factor(self, base_time_vec: np.array, capacity_vec: np.array, alpha: float=0.15, beta: float=4) -> np.array:
        return base_time_vec*alpha / (capacity_vec**beta)

    def update_edge_travel_time_factors(self) -> None:
        # extracts the vector of base travel times and capacities from each edge and precomputes the 
        btt_vec, cap_vec = np.hsplit(np.array([[seg.base_travel_time, seg.capacity] for seg in self.segments]), 2)
        self.base_time_factor = np.sum(btt_vec)
        self.capacity_factor = np.sum(self.calculate_bpr_capacity_factor(base_time_vec=btt_vec, capacity_vec=cap_vec, alpha=0.15, beta=4))
        return
    
    def compute_edge_travel_time(self, volume: float) -> float:
        return self.base_time_factor + self.capacity_factor*(volume**4)

    def step(self, actions):
        # states:
        cost = 0
        for segment, action in zip(self.segments, actions):
            segment_cost = segment.step(action)
            cost += segment_cost

        if 1 in actions:
            cost += self.inspection_campain_cost

        self.update_edge_travel_time_factors()

        return cost

    def reset(self):
        for segment in self.segments:
            segment.reset()
        self.update_edge_travel_time_factors()

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
        
    
