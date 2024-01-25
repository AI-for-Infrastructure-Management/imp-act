import numpy as np
import scipy as sp 
from scipy import stats

import igraph as ig
from igraph import Graph
import copy

class Shock():

    def __init__(self, max_timesteps: int, lambda_t: float=None, magnitude_dict: dict={}, location_dict: dict={}, 
                 pga_dict: dict={}, fragility_dict: dict={}, random_state=None) -> None:
        self.max_timesteps = max_timesteps
        self.lambda_t = lambda_t

        self.magnitude_dict = magnitude_dict
        """
        magnitude_dict = {
        'beta_m': _,    # decay parameter of double-exponential model
        'm_min':  _,    # minimum magnitude occurrence
        'm_max':  _,    # maximum magnitude occurrence
        }
        """

        self.location_dict = location_dict
        """
        location_dict = {
        'x_min': _,     # left x-boundary of earthquake occurrence rectangle
        'x_max': _,     # right x-boundary of earthquake occurrence rectangle
        'y_min': _,     # lower y-boundary of earthquake occurrence rectangle
        'y_max': _,     # upper y-boundary of earthquake occurrence rectangle
        }
        """

        self.pga_dict = pga_dict
        """
        pga_dict = {
        'b1_hat': _,    # b1 + bV*ln(VS/VS) of BFJ97 model
        'b2':     _,    # b2 of BFJ97 model
        'b3':     _,    # b3 of BFJ97 model
        'b_5':    _,    # b5 of BFJ97 model
        'h':      _,    # h of BFJ97 model
        }
        """

        self.fragility_dict = fragility_dict
        """
        fragility_dict = {
        'theta_vec': _,     # vector or matrix describing the median of the lognormal fragility curve 
        'sigma':  _,        # standard deviation of lognormal fragility curve
        }
        """

        self.random_state = random_state
        self.max_timesteps = max_timesteps
        return
    
    def get_theta_mat_from_vec(self, theta_vec: np.array) -> np.array:
        # theta_vec: fragility parameters (median of lognormal dist) of perfect state
        # assumption: probability of staying in current state in case of an earthquake 
        # stays the same in every deterioration state
        # -> implementation: delete the second fragility curve from the top
        theta_mat = np.full(shape=(len(theta_vec)+1,len(theta_vec)), fill_value=np.nan)
        for k in range(0, len(theta_vec)):
            ind = np.zeros(len(theta_vec), dtype=bool)
            ind[0] = True
            ind[k+1:] = True
            theta_mat[k, k:] = theta_vec[np.arange(len(theta_vec))[ind]]
        return theta_mat

    # shock modeled as exponential distribution with PDF: lambda_t*exp(-lambda_t*t)
    def get_earthquake_occurrence_time(self, lambda_t: float, size: int=1, random_state: int=None) -> float:
        return stats.expon.rvs(loc=0, scale=1/lambda_t, size=size, random_state=random_state)

    """
    # shock magnitude modeled as another exponential distribution with parameter beta_m
    def get_shock_magnitude_old(self, lambda_m: float, size: int=1, random_state: int=None) -> float:
        return stats.expon.rvs(loc=0, scale=1/lambda_m, size=size, random_state=random_state)
    """
    
    # shock magnitude modeled as a doubly exponential distribution beta_m, where m_min <= m <= m_max
    def get_shock_magnitude(self, beta_m: float, m_min: float=5, m_max: float=10, size: int=1, random_state: int=None) -> float:
        return stats.truncexpon.rvs(b=m_max, loc=m_min, scale=1/beta_m, size=size, random_state=random_state)
    
    # function that draws N 2D-uniformly distributed random locations from a specified grid [xmin, xmax] x [ymin, ymax]
    def get_shock_location(self, xmin: float, xmax: float, ymin: float, ymax: float, 
                           N_samples: int, random_state: int=None) -> np.array:
        return stats.uniform.rvs(loc=[xmin,ymin], scale=[xmax,ymax], size=(N_samples,2), random_state=random_state)

    # function that returns the peak-ground-acceleration at the site dependent on the earthquake magnitude and the site-source-distance
    def get_pga_from_distance(self, magn: float, dist: float, b1_hat: float, b2: float, b3: float, b5: float, h: float) -> np.array:
        return b1_hat + b2*(magn-6) + b3*(magn-6)**2 + b5*np.log(np.sqrt(dist**2 + h**2))
    
    # function that returns a fragility values parametrized by a vector of local pgas, a vector of medians (theta) and a single std parameters
    # -> the standard deviation has to be same for all states, otherwise the curves cross 
    def get_fragilities(self, shift: np.array, pga:float, theta_mat: np.array, sigma: float) -> np.array:
        return stats.norm.cdf(x=np.log((pga-shift)/theta_mat)/sigma)
    
    def get_det_probs_from_fragility_matrix(self, frag_mat: np.array) -> np.array:
        f2 = np.tril(np.ones_like(frag_mat), k=-1) + np.triu(frag_mat)
        return np.concatenate([np.ones((f2.shape[0],1)), f2], axis=1) - np.concatenate([f2, np.zeros((f2.shape[0],1))], axis=1)

    def single_loc_based_det_table_transform(self, magn: float, det_table: np.array, dist: float, shift: np.array,
                                      pga_dict: dict, fragility_dict: dict) -> np.array:
        assert det_table.dtype == float # otherwise np.nan is just a very large negative number and yields an error in np.log

        # get local pga from distance
        pga = self.get_pga_from_distance(magn=magn, dist=dist, **pga_dict)
        print('pga', pga)
        frag_mat = self.get_fragilities(pga=pga, shift=shift, **fragility_dict)
        print('frag_mat', frag_mat)
        shock_det_table = self.get_det_probs_from_fragility_matrix(frag_mat=frag_mat)
        return shock_det_table
    
    def loc_based_det_table_transform(self, magn: np.array, det_table_list: list, dist: np.array, shift_list: list, 
                                      pga_dict: dict, fragility_dict: dict) -> list:
        
        assert len(magn) == len(det_table_list) == len(dist) == len(shift_list)
        shock_table_list = list()
        for k in range(len(magn)):
            shock_table_list.append(self.single_loc_based_det_table_transform(magn=magn[k], det_table=det_table_list[k],
                                                                              dist=dist[k], shift=shift_list[k]),
                                                                              pga_dict=pga_dict, fragility_dict=fragility_dict)
        return shock_table_list
    
    def get_shock_t(self, max_timesteps: int, lambda_t: float, random_state=None) -> [int, float]:
        times = list()
        t = 0
        # loop until end of trajectory is reached
        while t < max_timesteps:
            # effect of two shocks within the same timestep not accounted for
            t += self.get_earthquake_occurrence_time(lambda_t=lambda_t, random_state=random_state)[0]
            if t < max_timesteps:
                times.append(np.ceil(t).astype(int))
        times = np.array(times)

        # check if two shocks at same timestep
        if len(np.unique(times)) != len(times):
            times, _ = np.unique(times, return_index=False)

        return times
    
    def get_shocks(self, max_timesteps: int, lambda_t: float, magnitude_dict: dict, 
                   location_dict: dict, random_state: int) -> list:
        times = self.get_shock_t(max_timesteps=max_timesteps, lambda_t=lambda_t, random_state=random_state)
        magni = np.array([])
        locs = np.array([])
        if len(times) > 0:
            magni = self.get_shock_magnitude(**magnitude_dict, size=len(times), random_state=random_state)
            locs = self.get_shock_location(**location_dict, N_samples=len(times), random_state=random_state)
        return times, magni, locs


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
        if not 'self.theta_mat' in locals():
            self.get_theta_mat_from_vec(theta_vec=self.fragility_dict.get('theta_vec')) 
        self.times, self.magnits, self.locs = self.get_shocks(max_timesteps=self.max_timesteps, lambda_t=self.lambda_t, 
                                                              magnitude_dict=self.magnitude_dict, location_dict=self.location_dict,
                                                              random_state=self.random_state)
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