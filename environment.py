import igraph as ig
from igraph import Graph
import numpy as np
from shock import Shock

class RoadSegment():
    def __init__(self, random_generator, shocks):
        # state [0-3]
        self.random_generator = random_generator
        self.shocks = shocks
        self.initial_observation = 0 
        self.number_of_states = 4

        self.reset()

        # base travel time table
        # shape: A x S
        self.base_travel_time_table = np.array([
                    [1.00, 1.10, 1.40, 1.60],
                    [1.00, 1.10, 1.40, 1.60],
                    [1.00, 1.05, 1.15, 1.45],
                    [1.50, 1.50, 1.50, 1.50]]) * self.base_travel_time
        
        # capacity table
        # shape: A x S
        self.capacity_table = np.array([
            [1.00, 1.00, 1.00, 1.00],
            [1.00, 1.00, 1.00, 1.00],
            [0.80, 0.80, 0.80, 0.80],
            [0.50, 0.50, 0.50, 0.50]]) * self.capacity

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
    

    def calc_distance(self, loc_a: np.array, loc_b: np.array) -> float:
        return np.linalg.norm(loc_a - loc_b)


    def reset(self):
        self.state = 0
        self.observation = self.initial_observation
        self.belief = np.array([1, 0, 0, 0])
        self.capacity = 500.0 # maybe cars per minute
        self.base_travel_time = 50.0 # maybe minutes it takes to travel trough a segment
        times_len = len(len(self.shocks.times))
        if times_len > 0:
            self.distances = np.zeros(times_len)
            self.shock_tables = self.shocks.loc_based_det_table_transform(magn=self.shocks.magni, det_table_list=[self.deterioration_table[0]]*times_len, 
                                                                          dist=self.distances, shift_list=[np.diag(np.diag(self.deterioration_table[0],1))]*times_len, 
                                                                          pga_dict=self.shocks.pga_dict, fragility_dict=self.shocks.fragility_dict)
            self.pgas = list()
            self.fragilities = list()
            for k in range(len(self.shocks.times)):
                self.distances.append(self.calc_distance(self.loc, self.shocks.locations[k]))
                self.pgas.append(self.shocks.get_pga_from_distance(self, magn=self.shocks.magni[k], dist=self.distances[-1], **self.shocks.pga_dict))

        # calculate here the effects of a potential shock, has to get passed somehow the shock instance

    def step(self, action, timestep):
        # actions: [do_nothing, inspect, minor repair, replacement] = [0, 1, 2, 3]

        # add here shock stuff, have to check for timestep in shocks.times; has to get passed
        # somehow the shock instance and the current timestep

        if timestep in self.shocks.times:
            # get modified deterioration matrix
            shock_ind = np.where(timestep == self.shocks.times)[0]
            magn = np.where(timestep == self.shocks.times)[0]
            shock_det_mat = self.shocks.loc_based_det_table_transform(magn=self.shocks.magni[shock_ind], 
                                                                      det_table=self.deterioration_table[0], 
                                                                      dist=self.dist[shock_ind], 
                                                                      pga_dict=self.shocks.pga_dict, 
                                                                      fragility_dict=self.shocks.fragility_dict)
            next_deterioration_state = self.random_generator.choice(
                np.arange(self.number_of_states), p=shock_det_mat[shock_ind][self.state]
            )
            
        
        # 1. could already superpose shock with this, but: no effect of action
        next_deterioration_state = self.random_generator.choice(
            np.arange(self.number_of_states), p=self.deterioration_table[action][self.state]
        )

        # 2. could add shock as another deterioration step after potential repair
        # -> shock happens directly after repair affects whole year
        # -> agent gets notified of shock due to negative reward + bad observations 
        self.base_travel_time = self.base_travel_time_table[action][self.state]
        self.capacity = self.capacity_table[action][self.state]

        reward = self.state_action_reward[self.state][action]

        # 3. could add shock after reward computation
        # -> shocks happens at end of year (does not affect reward of whole year)
        # -> agent first gets notified of shock due to bad observations and has opportunity to act
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
    def __init__(self, num_vertices, edges, edge_segments_numbers, trips, max_timesteps=50, 
                 graph=None, lambda_t=1/10, lambda_m=np.log(5)/5, seed=42):
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

        self.shocks = Shock(lambda_t=lambda_t, lambda_m=lambda_m, max_timesteps=self.max_timesteps, random_state=self.random_generator)

        self.reset()
        
        self.base_total_travel_time = self._get_total_travel_time()

    def reset(self):
        self.timestep = 0
        self.shocks.reset()
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
        """newly added shock"""
        if self.timestep in self.shocks.times:
            #print('In shock', self.shocks.times, self.timestep, self.shocks.magnits)
            when = np.where(self.timestep==self.shocks.times)[0][0]
            """the shock changes the deterioration tables fo the segments to increase the 
            probability of transitioning to the next state. This is currently horribly solved
            by saving and then restoring the original deterioration tables. This will be adjusted
            when we are doing vectorization"""
            self.shocks.save_det_tables(graph=self.graph)
            for i, edge in enumerate(self.graph.es["road_segments"]):
                for j, s in enumerate(edge.segments):
                    s.deterioration_table[actions[i][j]] = self.shocks.add_equal_shock_to_deterioration_table(magn=self.shocks.magnits[when], 
                                                                                                              det_table=s.deterioration_table[actions[i][j]])
            #print(self.shocks.copied_graph)
                    
        maintenance_reward = 0
        for i, edge in enumerate(self.graph.es):
            maintenance_reward += edge["road_segments"].step(actions[i])

        total_travel_time = self._get_total_travel_time()

        travel_time_reward = self.travel_time_reward_factor * (total_travel_time - self.base_total_travel_time)

        reward = maintenance_reward + travel_time_reward

        observation = self._get_observation()

        if self.timestep in self.shocks.times:
            """ restoring of deterioration tables """
            self.graph = self.shocks.restore_det_tables(graph=self.graph)

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
        self.shocks.random_state = self.random_generator
        for edge in self.graph.es:
            edge["road_segments"].random_generator = self.random_generator
            for segment in edge["road_segments"].segments:
                segment.random_generator = self.random_generator
