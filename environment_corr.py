import numpy as np
from igraph import Graph

class RoadSegment:
    def __init__(self, random_generator, coordinates):
        self.random_generator = random_generator
        self.coordinates = coordinates
        self.initial_observation = 0
        self.number_of_states = 4
        self.reset()
        self.initialize_deterioration_table()

    def reset(self):
        self.state = 0
        self.observation = self.initial_observation
        self.belief = np.array([1, 0, 0, 0])
        self.capacity = 500.0
        self.base_travel_time = 50.0
        self.base_travel_time_table = np.array(
            [
                [1.00, 1.10, 1.40, 1.60],
                [1.00, 1.10, 1.40, 1.60],
                [1.00, 1.05, 1.15, 1.45],
                [1.50, 1.50, 1.50, 1.50],
            ]
        ) * self.base_travel_time
        self.capacity_table = np.array(
            [
                [1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00],
                [0.80, 0.80, 0.80, 0.80],
                [0.50, 0.50, 0.50, 0.50],
            ]
        ) * self.capacity
        self.state_action_reward = np.array(
            [
                [0, -1, -20, -150],
                [0, -1, -25, -150],
                [0, -1, -30, -150],
                [0, -1, -40, -150],
            ]
        )

    def initialize_deterioration_table(self):
        self.deterioration_table = np.array(
            [
                [[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 1.0]],
                [[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.7, 0.2, 0.1, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        )

    def update_deterioration_table(self, hotspots):
        for action in [0, 1]:
            for hotspot in hotspots:
                distance = self.euclidean_distance(self.coordinates, hotspot['location'])
                influence = self.calculate_influence(distance, hotspot['severity_transition_probability'])
                self.deterioration_table[action] = np.average(
                    [self.deterioration_table[action], influence], axis=0, weights=[0.9, 0.1]
                )

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def calculate_influence(distance, severity_transition_probability):
        influence = 1 / (distance + 1e-5)
        return influence * severity_transition_probability

    # Other methods (step, compute_travel_time, etc.) go here

class RoadEdge:
    def __init__(self, number_of_segments, random_generator, bpr_alpha=0.15, bpr_beta=4):
        self.number_of_segments = number_of_segments
        self.random_generator = random_generator
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.reset()
        # Initialize RoadSegments with dummy coordinates (replace with actual data)
        self.segments = [
            RoadSegment(random_generator=random_generator, coordinates=(0, 0)) 
            for _ in range(number_of_segments)
        ]

    def reset(self):
        for segment in self.segments:
            segment.reset()

    # Other methods (step, compute_edge_travel_time, etc.) go here

class RoadEnvironment:
    def __init__(self, num_vertices, edges, edge_segments_numbers, trips, max_timesteps=50, graph=None, seed=42, hotspots=None):
        self.random_generator = np.random.default_rng(seed)
        self.max_timesteps = max_timesteps
        self.hotspots = hotspots if hotspots is not None else []
        if graph is None:
            self.create_graph(num_vertices, edges)
        else:
            self.graph = graph
        for edge, number_of_segments in zip(self.graph.es, edge_segments_numbers):
            edge["road_segments"] = RoadEdge(
                number_of_segments=number_of_segments,
                random_generator=self.random_generator,
            )
        self.trips = trips
        # Other initialization (travel_time_reward_factor, etc.) goes here
        self.update_hotspots(self.hotspots)

    def create_graph(self, num_vertices, edges):
        self.graph = Graph()
        self.graph.add_vertices(num_vertices)
        self.graph.add_edges(edges)

    def update_hotspots(self, new_hotspots):
        self.hotspots = new_hotspots
        for edge in self.graph.es:
            for segment in edge["road_segments"].segments:
                segment.update_deterioration_table(self.hotspots)

    # Other methods (step, reset, etc.) go here

# Example usage
small_environment_dict = {
    "num_vertices": 4,
    "edges": [(0, 1), (1, 3), (2, 0), (3, 2)],
    "edge_segments_numbers": [2, 2, 2, 2],
    "trips": [(0, 1, 200), (1, 3, 200), (2, 0, 200), (3, 2, 200)],
    "max_timesteps": 50,
}

hotspots = [
    {'location': (0.5, 0.5), 'severity_transition_probability': [0.1, 0.9, 0.0, 0.0]},
    # ... Add other hotspots as needed ...
]

environment = RoadEnvironment(
    num_vertices=small_environment_dict["num_vertices"],
    edges=small_environment_dict["edges"],
    edge_segments_numbers=small_environment_dict["edge_segments_numbers"],
    trips=small_environment_dict["trips"],
    max_timesteps=small_environment_dict["max_timesteps"],
    hotspots=hotspots
)

