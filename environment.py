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
        self.belief = np.array([1, 0, 0, 0])

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

        #Belief state computation
        self.belief = self.transition_tables[action].T @ self.belief

        state_probs = self.observation_tables[action][:, self.observation] # likelihood of observation

        # Bayes' rule
        self.belief = state_probs * self.belief # likelihood * prior
        self.belief /= np.sum(self.belief) # normalize

        return reward, self.belief

    def compute_travel_time(self, action):
        return 0 # travel_time

class RoadEdge():
    def __init__(self, number_of_segments):
        self.number_of_segments = number_of_segments
        self.inspection_campaign_cost = -5
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
        segment_beliefs = []
        for segment, action in zip(self.segments, actions):
            segment_cost, belief = segment.step(action)
            cost += segment_cost
            segment_beliefs.append(belief)

        if 1 in actions:
            cost += self.inspection_campaign_cost

        self.update_edge_travel_time_factors()

        return cost, segment_beliefs

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

        self.trips = [(0, 1, 200), (1, 3, 200), (2, 0, 200), (3, 2, 200)]
        self.traffic_assignment_max_iterations = 15
        self.traffic_assignment_convergence_threshold = 0.01
        self.traffic_assignment_update_weight = 0.5

        self.travel_time_cost = 0.01

        self.reset()
        
        self.base_total_travel_time = self._get_total_travel_time()

    def reset(self):
        self.timestep = 0
        for edge in self.graph.es:
            edge["road_segments"].reset()

    def _get_observation(self):
        adjacency_matrix = np.array(self.graph.get_adjacency().data)
        edge_observations = []
        edge_nodes = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_segments"].get_observation())
            edge_nodes.append([edge.source, edge.target])

        observations = {
            "adjacency_matrix": adjacency_matrix, 
            "edge_observations": edge_observations,
            "edge_beliefs": self.beliefs,
            "edge_nodes": edge_nodes
            }

        return observations

    def _get_states(self):
        edge_states = []
        for edge in self.graph.es:
            edge_states.append(edge["road_segments"].get_states())

        return edge_states
        
    def _get_total_travel_time(self):        
        # Initialize volumes
        self.graph.es['volume'] = 0
        
        # Initialize with all-or-nothing assignment
        self.graph.es['travel_time'] = [
                edge['road_segments'].compute_edge_travel_time(edge['volume'])
                for edge in self.graph.es
            ]

        for source, target, num_cars in self.trips:
            path = self.graph.get_shortest_paths(source, target, weights='travel_time', output='epath')[0]
            for edge_id in path:
                self.graph.es[edge_id]['volume'] += num_cars

        for iteration in range(self.traffic_assignment_max_iterations):
            # Recalculate travel times with current volumes
            self.graph.es['travel_time'] = [
                edge['road_segments'].compute_edge_travel_time(edge['volume'])
                for edge in self.graph.es
            ]

            # Find the shortest paths using updated travel times
            new_volumes = np.zeros(len(self.graph.es))
            for source, target, num_cars in self.trips:
                path = self.graph.get_shortest_paths(source, target, weights='travel_time', output='epath')[0]
                for edge_id in path:
                    new_volumes[edge_id] += num_cars

            # Check for convergence by comparing volume changes
            volume_changes = np.abs(self.graph.es['volume'] - new_volumes)
            max_change = np.max(volume_changes)

            if max_change <= self.traffic_assignment_convergence_threshold:
                break

            # Update volumes by averaging
            self.graph.es['volume'] = (
                np.array(self.graph.es['volume']) * (1 - self.traffic_assignment_update_weight) +
                new_volumes * self.traffic_assignment_update_weight
                )

        return np.sum([edge["travel_time"] * edge["volume"] for edge in self.graph.es])

    def step(self, actions):
        total_cost = 0
        all_beliefs = []
        for i, edge in enumerate(self.graph.es):
            cost, beliefs = edge["road_segments"].step(actions[i])
            total_cost += cost
            all_beliefs.append(beliefs)

        self.beliefs = all_beliefs

        total_travel_time = self._get_total_travel_time()

        cost = total_cost + self.travel_time_cost * (total_travel_time - self.base_total_travel_time)

        observation = self._get_observation()

        self.timestep += 1

        info = {
            "states": self._get_states(), 
            "total_travel_time": total_travel_time, 
            "travel_times": self.graph.es['travel_time'],
            "volumes": self.graph.es['volume']
        }

        return observation, cost, self.timestep >= self.max_timesteps, info
        
    
