import copy

import igraph as ig
import numpy as np
import scipy as sp
from igraph import Graph
from scipy import stats


class Shock:
    def __init__(
        self,
        max_timesteps: int,
        lambda_t: float = None,
        magnitudes: dict = {},
        locations: dict = {},
        pgas: dict = {},
        fragilities: dict = {},
        eps: float = 1e-10,
        random_generator=None,
    ) -> None:
        self.max_timesteps = max_timesteps
        self.lambda_t = lambda_t

        self.magnitude_dict = magnitudes
        """
        magnitude_dict = {
        'beta_m': _,    # decay parameter of double-exponential model
        'm_min':  _,    # minimum magnitude occurrence
        'm_max':  _,    # maximum magnitude occurrence
        }
        """

        self.location_dict = locations
        """
        location_dict = {
        'x_min': _,     # left x-boundary of earthquake occurrence rectangle
        'x_max': _,     # right x-boundary of earthquake occurrence rectangle
        'y_min': _,     # lower y-boundary of earthquake occurrence rectangle
        'y_max': _,     # upper y-boundary of earthquake occurrence rectangle
        }
        """

        self.pga_dict = pgas
        """
        pga_dict = {
        'b1_hat': _,    # b1 + bV*ln(VS/VS) of BFJ97 model
        'b2':     _,    # b2 of BFJ97 model
        'b3':     _,    # b3 of BFJ97 model
        'b5':     _,    # b5 of BFJ97 model
        'h':      _,    # h of BFJ97 model
        }
        """

        self.fragility_dict = fragilities
        """
        fragility_dict = {
        'theta_mat': _,     # vector or matrix describing the median of the lognormal fragility curve 
        'sigma':  _,        # standard deviation of lognormal fragility curve
        }
        """

        self.random_generator = random_generator
        self.max_timesteps = max_timesteps
        self.eps = eps

        self.reset()
        return

    def get_theta_mat_from_vec(self, theta_mat: np.array) -> np.array:
        # theta_mat: fragility parameters (median of lognormal dist) of perfect state
        # assumption: probability of staying in current state in case of an earthquake
        # stays the same in every deterioration state
        # -> implementation: delete the second fragility curve from the top
        if type(theta_mat) == list:
            if type(theta_mat[0]) == list:
                theta_mat = np.array([[np.nan if (x=='None') else x for x in row] for row in theta_mat])
            else:
                theta_mat = np.array(theta_mat)
        if len(theta_mat.shape) > 1:
            theta_mat = theta_mat.copy()
        else:
            theta_mat = np.full(
                shape=(len(theta_mat) + 1, len(theta_mat)), fill_value=np.nan
            )
            for k in range(0, len(theta_mat)):
                ind = np.zeros(len(theta_mat), dtype=bool)
                ind[0] = True
                ind[k + 1 :] = True
                theta_mat[k, k:] = theta_mat[np.arange(len(theta_mat))[ind]]
        return theta_mat

    def get_shift_table_from_det_table(self, det_table: np.array) -> np.array:
        return np.triu(det_table[:-1, 1:])

    # shock modeled as exponential distribution with PDF: lambda_t*exp(-lambda_t*t)
    def get_earthquake_occurrence_time(
        self, lambda_t: float, size: int = 1, random_generator: int = None
    ) -> float:
        return stats.expon.rvs(
            loc=0, scale=1 / lambda_t, size=size, random_state=random_generator
        )

    # shock magnitude modeled as a doubly exponential distribution beta_m, where m_min <= m <= m_max
    def get_shock_magnitude(
        self,
        beta_m: float,
        m_min: float = 5,
        m_max: float = 10,
        size: int = 1,
        random_generator: int = None,
    ) -> float:
        return stats.truncexpon.rvs(
            b=m_max, loc=m_min, scale=1 / beta_m, size=size, random_state=random_generator
        )

    # function that draws N 2D-uniformly distributed random locations from a specified grid [xmin, xmax] x [ymin, ymax]
    def get_shock_location(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        N_samples: int,
        random_generator: int = None,
    ) -> np.array:
        return stats.uniform.rvs(
            loc=[x_min, y_min],
            scale=[x_max, y_max],
            size=(N_samples, 2),
            random_state=random_generator,
        )

    # function that returns the peak-ground-acceleration at the site dependent on the earthquake magnitude and the site-source-distance
    def get_pga_from_distance(
        self,
        magn: float,
        dist: float,
        b1_hat: float,
        b2: float,
        b5: float,
        h: float,
    ) -> np.array:
        return np.exp(b1_hat + b2 * (magn - 6) + b5 * np.log(np.sqrt(dist**2 + h**2)))

    # function that returns a fragility values parametrized by a vector of local pgas, a vector of medians (theta) and a single std parameters
    # -> the standard deviation has to be same for all states, otherwise the curves cross
    def get_fragilities(
        self, shift: np.array, pga: float, theta_mat: np.array, sigma: float
    ) -> np.array:
        #print(type(pga), pga)
        #print(type(theta_mat), theta_mat)
        #print(type(shift), shift)
        #print(type(sigma), sigma)
        #print(type(self.eps), self.eps)
        return stats.norm.cdf(x=np.log(pga/theta_mat + np.exp(stats.norm.ppf(shift)*sigma + self.eps)) / sigma)

    def get_det_probs_from_fragility_matrix(self, frag_mat: np.array) -> np.array:
        f2 = np.tril(np.ones_like(frag_mat), k=-1) + np.triu(frag_mat)
        return np.concatenate([np.ones((f2.shape[0], 1)), f2], axis=1) - np.concatenate(
            [f2, np.zeros((f2.shape[0], 1))], axis=1
        )

    def single_loc_based_det_table_transform(
        self,
        magn: float,
        dist: float,
        shift: np.array,
        det_table_append: np.array,
        pga_dict: dict,
        fragility_dict: dict,
    ) -> np.array:
        assert (
            shift.dtype == float
        )  # otherwise np.nan is just a very large negative number and yields an error in np.log

        # get local pga from distance
        pga = self.get_pga_from_distance(magn=magn, dist=dist, **pga_dict)
        frag_mat = self.get_fragilities(pga=pga, shift=shift, **fragility_dict)
        shock_det_table = self.get_det_probs_from_fragility_matrix(frag_mat=frag_mat)
        # add the last row at the bottom (disregarded in computation)
        shock_det_table = np.concatenate([shock_det_table, det_table_append], axis=0)
        return pga, shock_det_table

    def loc_based_det_table_transform(
        self,
        magn: np.array,
        dist: np.array,
        shift_list: list,
        det_table_append_list: list,
        pga_dict: dict,
        fragility_dict: dict,
    ) -> list:
        
        assert len(magn) == len(dist) == len(shift_list) == len(det_table_append_list)
        shock_table_list = list()
        pga_list = list()
        for k in range(len(magn)):
            pga, shock_table = self.single_loc_based_det_table_transform(
                    magn=magn[k],
                    dist=dist[k],
                    shift=shift_list[k],
                    det_table_append=det_table_append_list[k],
                    pga_dict=pga_dict,
                    fragility_dict=fragility_dict,
                )
            pga_list.append(pga)
            shock_table_list.append(shock_table)
        return np.array(pga_list).squeeze(), shock_table_list

    def get_shock_t(
        self, max_timesteps: int, lambda_t: float, random_generator=None
    ) -> [int, float]:
        times = list()
        t = 0
        # loop until end of trajectory is reached
        while t < max_timesteps:
            # effect of two shocks within the same timestep not accounted for
            t += self.get_earthquake_occurrence_time(
                lambda_t=lambda_t, random_generator=random_generator
            )[0]
            if t < max_timesteps:
                times.append(np.ceil(t).astype(int))
        times = np.array(times)

        # check if two shocks at same timestep
        if len(np.unique(times)) != len(times):
            times = np.unique(times)

        return times

    def get_shocks(
        self,
        max_timesteps: int,
        lambda_t: float,
        magnitude_dict: dict,
        location_dict: dict,
        random_generator: int,
    ) -> list:
        times = self.get_shock_t(
            max_timesteps=max_timesteps, lambda_t=lambda_t, random_generator=random_generator
        )
        magni = np.array([])
        locs = np.array([])
        if len(times) > 0:
            magni = self.get_shock_magnitude(
                **magnitude_dict, size=len(times), random_generator=random_generator
            )
            locs = self.get_shock_location(
                **location_dict, N_samples=len(times), random_generator=random_generator
            )
        return times, magni, locs

    def reset(self) -> None:
        if not hasattr(self, "theta_mat"):
            self.fragility_dict["theta_mat"] = self.get_theta_mat_from_vec(theta_mat=self.fragility_dict.get("theta_mat"))
        self.times, self.magni, self.locs = self.get_shocks(
            max_timesteps=self.max_timesteps,
            lambda_t=self.lambda_t,
            magnitude_dict=self.magnitude_dict,
            location_dict=self.location_dict,
            random_generator=self.random_generator,
        )
        return
    
    """
    def add_equal_shock_to_deterioration_table(
        self, magn: float, det_table: np.array
    ) -> np.array:
        addition = 10 ** (magn - 6)
        # trans_table is for now sxs, where s is the number of states
        # in future, this should be nxsxs where n is the number of segments to perform the update for all segments at the same time
        if len(det_table.shape) == 2:
            # increase the probability of the state next to the current state upper (off-diagonal)
            x = np.arange(det_table.shape[0])
            det_table[
                x, np.clip(x + 1, a_min=None, a_max=det_table.shape[0] - 1)
            ] += addition
            det_table /= np.sum(det_table, axis=1, keepdims=True)
        elif len(det_table.shape) == 3:
            x = np.arange(det_table.shape[0])
            y = np.arange(det_table.shape[1])
            det_table[x, y, np.clip(y + 1, a_max=det_table.shape[1])] += addition
            det_table /= np.sum(det_table, axis=2, keepdims=True)
        return det_table
    """


    """
    def save_det_tables(self, graph: Graph) -> None:
        self.copied_graph = copy.deepcopy(graph.es["road_segments"])
        return

    def restore_det_tables(self, graph: Graph) -> Graph:
        # for i, edge in enumerate(graph.es["road_segments"]):
        #    pass

        # print(graph.es['road_segments'][0].segments[0].deterioration_table)
        for o, n in zip(graph.es["road_segments"], self.copied_graph):
            for so, sn in zip(o.segments, n.segments):
                so.deterioration_table = copy.deepcopy(sn.deterioration_table)
        # print(graph.es['road_segments'][0].segments[0].deterioration_table)
        return graph
    """
