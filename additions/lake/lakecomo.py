from additions.lake.lake import Lake
import numpy as np

class Lakecomo(Lake):

    def __init__(self, evap_rates, tailwater, minEnvFlow, lsv_rel, rating_curve, initial_cond=None, surface=1459e5, EV=0, seed=None):
        '''

        initialization
        :param evap_rates: evapotranspiration rates
        :param tailwater: release - tailwater level
        :param minEnvFlow: MEF
        :param lsv_rel: level - surface - storage relationship
        :param rating_curve: level - max release - min release
        :param initial_cond: initial condition storage/level if none is uniformly sampled in [min=-0.5,max=1.3]
        :param surface: lake surface
        :param EV: -1 = no, 0 = hist. trajectory (evap), 1 = compute
        :param seed: the seed to get reproducibility in case the initial condition is not given
        '''
        if initial_cond is None:
            assert seed is not None
            np.random.seed(seed)
            initial_cond = np.random.uniform(-0.5, 1.23)
        super().__init__(evap_rates, tailwater, minEnvFlow, lsv_rel, rating_curve, initial_cond, surface, EV)

    def storageToLevel(self, s):

        h0 = -0.5
        h = s / self.surface + h0

        return h

    def levelToStorage(self, h):

        h0 = -0.5
        s = self.surface * (h - h0)

        return s

    def levelToSurface(self, h):

        S = self.surface
        return S

    def min_release(self, s, tt):

        DMV = self.minEnvFlow[tt]
        h = self.storageToLevel(s)
        if h <= -0.5:
            q = 0
        elif h <= 1.25:
            q = DMV
        else:
            q = 33.37 * ((h + 2.5) ** 2.015)

        return q
    
    def max_release(self, s):

        h = self.storageToLevel(s)
        if h <= -0.5:
            q = 0.0
        elif h <= -0.40:
            q = 1488.1*h + 744.05
        else:
            q = 33.37*((h + 2.5) ** 2.015)

        return q
    