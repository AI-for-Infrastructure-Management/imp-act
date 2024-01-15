import numpy as np
import scipy as sp 
from scipy import stats

import igraph as ig
from igraph import Graph
import copy

class Shock(object):

    def __init__(self, lambda_t: float, lambda_m: float, max_timesteps: int, random_state=None) -> None:
        self.lambda_t = lambda_t
        self.lambda_m = lambda_m
        self.random_state = random_state
        self.max_timesteps = max_timesteps

    # shock modeled as exponential distribution with PDF: lambda_t*exp(-lambda_t*t)
    def get_earthquake_occurrence_time(self, lambda_t: float, size: int=1, random_state: int=None) -> float:
        return stats.expon.rvs(loc=0, scale=1/lambda_t, size=size, random_state=random_state)

    # shock magnitude modeled as another exponential distribution with parameter lambda_m
    def get_shock_magnitude(self, lambda_m: float, size: int=1, random_state: int=None) -> float:
        return stats.expon.rvs(loc=0, scale=1/lambda_m, size=size, random_state=random_state)
    
    def get_shocks(self, lambda_t: float, lambda_m: float, max_timesteps: int, random_state=None) -> [int, float]:
        times = list()
        magni = list()
        t = 0
        #print('Beginning: ', t, max_timesteps)
        while t < max_timesteps:
            # effect of two shocks within the same timestep not accounted for
            t += self.get_earthquake_occurrence_time(lambda_t=lambda_t, random_state=random_state)[0]
            #print('\nDrawn time:  ', t, max_timesteps)
            if t < max_timesteps:
                #print('Inside second if')
                m = self.get_shock_magnitude(lambda_m=lambda_m, random_state=random_state)[0]
                #print('Magnitude', m, 5)
                if m > 5:
                    times.append(np.ceil(t).astype(int))
                    magni.append(m)

        times = np.array(times)
        magni = np.array(magni)

        # check if two shocks at same timestep
        if len(np.unique(times)) != len(times):
            times, t_indices = np.unique(times, return_index=True)
            magni = magni[t_indices]

        return times, magni

    def add_equal_shock_to_deterioration_table(self, magn: float, det_table: np.array) -> np.array:
        addition = 10**(magn-6)
        # trans_table is for now sxs, where s is the number of states
        # in future, this should be nxsxs where n is the number of segments to perform the update for all segments at the same time
        if len(det_table.shape) == 2:
            # increase the probability of the state next to the current state upper (off-diagonal)
            x = np.arange(det_table.shape[0])
            det_table[x, np.clip(x+1, a_min=None, a_max=det_table.shape[0]-1)] += addition
            det_table /= np.sum(det_table, axis=1, keepdims=True)
        elif len(det_table.shape) == 3:
            x = np.arange(det_table.shape[0])
            y = np.arange(det_table.shape[1])
            det_table[x, y, np.clip(y+1, a_max=det_table.shape[1])] += addition
            det_table /= np.sum(det_table, axis=2, keepdims=True)
        return det_table
    
    def reset(self) -> None:
        self.times, self.magnits = self.get_shocks(lambda_t=self.lambda_t, lambda_m=self.lambda_m, 
                                                   max_timesteps=self.max_timesteps, random_state=self.random_state)
        return
    
    def save_det_tables(self, graph: Graph) -> None:
        self.copied_graph = copy.deepcopy(graph.es["road_segments"])
        return

    def restore_det_tables(self, graph: Graph) -> Graph:
        #for i, edge in enumerate(graph.es["road_segments"]):
        #    pass

        #print(graph.es['road_segments'][0].segments[0].deterioration_table)
        for o, n in zip(graph.es["road_segments"], self.copied_graph):
            for so, sn in zip(o.segments, n.segments):
                so.deterioration_table = copy.deepcopy(sn.deterioration_table)
        #print(graph.es['road_segments'][0].segments[0].deterioration_table)
        return graph