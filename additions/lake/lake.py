#from utils import interp_lin
import numpy as np
class Lake:
    def __init__(self, evap_rates, tailwater, minEnvFlow, lsv_rel, rating_curve, initial_cond, surface, EV):
        '''
        initialization
        :param evap_rates: evapotranspiration rates
        :param tailwater: release - tailwater level
        :param minEnvFlow: MEF
        :param lsv_rel: level - surface - storage relationship
        :param rating_curve: level - max release - min release
        :param initial_cond: initial condition storage/levl
        :param surface: lake surface
        :param EV: 0 = no evaporation, 1 = from file, 2 = call specific function
        '''

        self.initial_cond = initial_cond
        self.surface = surface
        self.lsv_rel = lsv_rel
        self.rating_curve = rating_curve
        self.EV = EV
        self.evap_rates = evap_rates
        self.tailwater = tailwater  # assumo tailwater una lista di due numpy array se inserita (da capire meglio)
        self.minEnvFlow = minEnvFlow

    def integrate(self, HH, tt, s0, uu, n_sim):
        '''
        integration routine
        :param HH: number of sub-daily steps for the integration
        :param tt: the current step within the simulation horizon (day of the year -1)
        :param s0: water storage at the beginning of the day t
        :param uu: decision made by the policy
        :param n_sim: inflow
        :return:
        '''

        sim_step = 3600 * 24 / HH
        s = np.zeros(HH+1)
        r = np.zeros(HH)

        s[0] = s0
        for i in range(HH):
            r[i] = self.actual_release(uu, s[i], tt)
            if self.EV == 1:
                S = self.levelToSurface(self.storageToLevel(s[i]))
                E = self.evap_rates[tt] / 1000 * S / 86400
            elif self.EV > 1:
                raise NotImplementedError
            else:
                E = 0

            s[i + 1] = s[i] + sim_step * (n_sim - r[i] - E) #transition of the sistem
        return s[HH], np.mean(r)

    '''def relToTailwater(self, r):
        hd = 0
        if len(self.tailwater) > 0:
            hd = interp_lin(self.tailwater[0], self.tailwater[1], r)

        return hd'''

    def actual_release(self, uu, s, tt):
        qm = self.min_release(s, tt)
        qM = self.max_release(s)
        rr = min(qM, max(qm, uu))

        return rr

    def min_release(self, s, tt):
        raise NotImplementedError

    def max_release(self, s):
        raise NotImplementedError

    def levelToSurface(self, h):
        raise NotImplementedError

    def levelToStorage(self, h):
        raise NotImplementedError

    def storageToLevel(self, s):
        raise NotImplementedError