import igraph as ig
from igraph import Graph
import numpy as np

class RoadSegment():
    def __init__(self, random_generator):
        # state [0-3]
        self.random_generator = random_generator
        self.initial_observation = 0 
        self.number_of_states = 4

        self.reset()

        # base travel time table
        # shape: S x A
        self.base_travel_time_table = np.array([
            [1.00, 1.00, 1.00, 1.50],
            [1.10, 1.10, 1.05, 1.50],
            [1.40, 1.40, 1.15, 1.50],
            [1.60, 1.60, 1.45, 1.50]]) * self.base_travel_time
        
        # capacity table
        # shape: S x A
        self.capacity_table = np.array([
            [1.00, 1.00, 0.80, 0.50],
            [1.00, 1.00, 0.80, 0.50],
            [1.00, 1.00, 0.80, 0.50],
            [1.00, 1.00, 0.80, 0.50]]) * self.capacity

        # deterioration tables
        # shape: A x S x S
        self.deterioration_table = np.array([
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

    def reset(self):
        self.state = 0
        self.observation = self.initial_observation
        self.belief = np.array([1, 0, 0, 0])
        self.capacity = 500.0 # maybe cars per minute
        self.base_travel_time = 50.0 # maybe minutes it takes to travel trough a segment

    def step(self, action):
        # actions: [do_nothing, inspect, minor repair, replacement] = [0, 1, 2, 3]
        
        if self.observation == 3:
            action = 3 # force replacement
        
        next_deterioration_state = self.random_generator.choice(
            np.arange(self.number_of_states), p=self.deterioration_table[action][self.state]
        )

        self.base_travel_time = self.base_travel_time_table[self.state][action]
        self.capacity = self.capacity_table[self.state][action]

        reward = self.state_action_reward[self.state][action]
        self.state = next_deterioration_state

        self.observation = self.random_generator.choice(
            np.arange(self.number_of_states), p=self.observation_tables[action][self.state]
        )

        # Belief state computation
        self.belief = self.deterioration_table[action].T @ self.belief

        state_probs = self.observation_tables[action][:, self.observation] # likelihood of observation

        # Bayes' rule
        self.belief = state_probs * self.belief # likelihood * prior
        self.belief /= np.sum(self.belief) # normalize

        return reward

    def compute_travel_time(self, action):
        return 0 # travel_time

class RoadEdge():
    def __init__(self, number_of_segments, random_generator, bpr_alpha=0.15, bpr_beta=4):
        self.number_of_segments = number_of_segments
        self.inspection_campaign_reward = -5
        self.random_generator = random_generator
        self.segments = [RoadSegment(random_generator = random_generator) for _ in range(number_of_segments)]
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.reset()
    
    # Define a function for calculating BPR travel times based on volume and capacity
    def calculate_bpr_travel_time(volume, capacity, base_time, alpha, beta):
        return base_time * (1 + alpha * (volume / capacity)**beta)
    
    def calculate_bpr_capacity_factor(self, base_time_vec: np.array, capacity_vec: np.array) -> np.array:
        return base_time_vec*self.bpr_alpha / (capacity_vec**self.bpr_beta)

    def update_edge_travel_time_factors(self) -> None:
        # extracts the vector of base travel times and capacities from each edge and precomputes the 
        btt_vec, cap_vec = np.hsplit(np.array([[seg.base_travel_time, seg.capacity] for seg in self.segments]), 2)
        self.base_time_factor = np.sum(btt_vec)
        self.capacity_factor = np.sum(self.calculate_bpr_capacity_factor(base_time_vec=btt_vec, capacity_vec=cap_vec))
        return
    
    def compute_edge_travel_time(self, volume: float) -> float:
        return self.base_time_factor + self.capacity_factor*(volume**self.bpr_beta)

    def step(self, actions):
        
        if len(self.segments) != len(actions):
            raise ValueError("self.segments and actions must have the same length")
    
        reward = 0
        for segment, action in zip(self.segments, actions):
            segment_reward = segment.step(action)
            reward += segment_reward

        if 1 in actions:
            reward += self.inspection_campaign_reward

        self.update_edge_travel_time_factors()

        return reward

    def reset(self):
        for segment in self.segments:
            segment.reset()
        self.update_edge_travel_time_factors()

    def get_observation(self):
        return [segment.observation for segment in self.segments]
    
    def get_beliefs(self):
        return [segment.belief for segment in self.segments]

    def get_states(self):
        return [segment.state for segment in self.segments]

class RoadEnvironment():
    def __init__(self, num_vertices, edges, edge_segments_numbers, trips, max_timesteps=50, graph=None, seed=42):
        self.random_generator = np.random.default_rng(seed)
        self.max_timesteps = max_timesteps
        self.travel_time_factor = 1
        self.edge_segments_numbers = edge_segments_numbers
    
        if graph is None:
            self.create_graph(num_vertices, edges)
        else:
            self.graph = graph

        for edge, number_of_segments in zip(self.graph.es, edge_segments_numbers):
            edge["road_segments"] = RoadEdge(number_of_segments=number_of_segments, random_generator=self.random_generator)

        self.trips = trips
        self.traffic_assignment_max_iterations = 15
        self.traffic_assignment_convergence_threshold = 0.01
        self.traffic_assignment_update_weight = 0.5

        self.travel_time_reward_factor = -0.01

        self.reset()
        
        self.base_total_travel_time = self._get_total_travel_time()

    def reset(self):
        self.timestep = 0
        for edge in self.graph.es:
            edge["road_segments"].reset()
        return self._get_observation()

    def create_graph(self, num_vertices, edges):
        self.graph = Graph()
        self.num_vertices = num_vertices
        self.edges = edges
        self.graph.add_vertices(num_vertices)
        self.graph.add_edges(edges)

    def _get_observation(self):
        adjacency_matrix = np.array(self.graph.get_adjacency().data)
        edge_observations = []
        edge_nodes = []
        edge_beliefs = []
        for edge in self.graph.es:
            edge_observations.append(edge["road_segments"].get_observation())
            edge_beliefs.append(edge["road_segments"].get_beliefs())
            edge_nodes.append([edge.source, edge.target])

        observations = {
            "adjacency_matrix": adjacency_matrix, 
            "edge_observations": edge_observations,
            "edge_beliefs": edge_beliefs,
            "edge_nodes": edge_nodes,
            "time_step": self.timestep,
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
        maintenance_reward = 0
        for i, edge in enumerate(self.graph.es):
            maintenance_reward += edge["road_segments"].step(actions[i])

        total_travel_time = self._get_total_travel_time()

        travel_time_reward = self.travel_time_reward_factor * (total_travel_time - self.base_total_travel_time)

        reward = maintenance_reward + travel_time_reward

        observation = self._get_observation()

        self.timestep += 1

        info = {
            "states": self._get_states(), 
            "total_travel_time": total_travel_time, 
            "travel_times": self.graph.es['travel_time'],
            "volumes": self.graph.es['volume'],
            "reward_elements": [travel_time_reward, maintenance_reward]
        }

        return observation, reward, self.timestep >= self.max_timesteps, info
    
    def seed(self, seed):
        self.random_generator = np.random.default_rng(seed)
        for edge in self.graph.es:
            edge["road_segments"].random_generator = self.random_generator
            for segment in edge["road_segments"].segments:
                segment.random_generator = self.random_generator
