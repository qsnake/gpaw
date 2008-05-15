# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Ref. to Kresse-paper ... XXX
"""

import numpy as npy

from gpaw.utilities.blas import axpy
from gpaw.operators import Operator


class BaseMixer:
    """Pulay density mixer."""
    
    def __init__(self, beta=0.25, nmaxold=3, metric=None, weight=50.0):
        """Construct density-mixer object.

        Parameters
        ----------
        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        metrix: None, 'old' or 'new'
            Type of metric to use.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.metric_type = metric
        self.weight = weight

        self.dNt = None

        self.mix_rho = False

    def initialize_metric(self, gd):
        self.gd = gd

        if self.metric_type is None:
            self.metric = None

        elif self.metric_type == 'old':
            b = 0.25 * (self.weight - 1)
            a = 1.0 - 2.0 * b
            self.metric = Operator([a,
                                    b, b, b, b, b, b],
                                   [(0, 0, 0),
                                    (-1, 0, 0), (1, 0, 0),
                                    (0, -1, 0), (0, 1, 0),
                                    (0, 0, -1), (0, 0, 1)],
                                   gd, float).apply
            self.mR_G = gd.empty()

        elif self.metric_type == 'new':
            a = 0.125 * (self.weight + 7)
            b = 0.0625 * (1 - self.weight)
            c = 0.03125 * (self.weight - 1)
            d = 0.015625 * (self.weight - 1)
            self.metric = Operator([a,
                                    b, b, b, b, b, b,
                                    c, c, c, c, c, c, c, c, c, c, c, c,
                                    d, d, d, d, d, d, d, d],
                                   [(0, 0, 0),
                                    (-1, 0, 0), (1, 0, 0),                 #b
                                    (0, -1, 0), (0, 1, 0),                 #b
                                    (0, 0, -1), (0, 0, 1),                 #b
                                    (1, 1, 0), (1, 0, 1), (0, 1, 1),       #c
                                    (1, -1, 0), (1, 0, -1), (0, 1, -1),    #c
                                    (-1, 1, 0), (-1, 0, 1), (0, -1, 1),    #c
                                    (-1, -1, 0), (-1, 0, -1), (0, -1, -1), #c
                                    (1, 1, 1), (1, 1, -1), (1, -1, 1),     #d
                                    (-1, 1, 1), (1, -1, -1), (-1, -1, 1),  #d
                                    (-1, 1, -1), (-1, -1, -1)              #d
                                    ],
                                   gd, float).apply
            self.mR_G = gd.empty()

        else:
            raise RuntimeError('Unknown metric type: "%s".' % self.metric_type)
        
    def initialize(self, paw):
        self.initialize_metric(paw.gd)

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """
        
        # History for Pulay mixing of densities:
        self.nt_iG = [] # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = npy.zeros((0, 0))
        self.dNt = None
        
        self.D_iap = []
        self.dD_iap = []

    def get_charge_sloshing(self):
        """Return number of electrons moving around.

        Calculated as the integral of the absolute value of the change
        of the density from input to output."""
        
        return self.dNt

    def mix(self, nt_G, D_ap):
        iold = len(self.nt_iG)
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                del self.D_iap[0]
                del self.dD_iap[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_G = nt_G - self.nt_iG[-1]
            self.dNt = self.gd.integrate(npy.fabs(R_G))
            self.R_iG.append(R_G)
            self.dD_iap.append([])
            for D_p, D_ip in zip(D_ap, self.D_iap[-1]):
                self.dD_iap[-1].append(D_p - D_ip)

            # Update matrix:
            A_ii = npy.zeros((iold, iold))
            i1 = 0
            i2 = iold - 1
            
            if self.metric is None:
                mR_G = R_G
            else:
                mR_G = self.mR_G
                self.metric(R_G, mR_G)
                
            for R_1G in self.R_iG:
                a = self.gd.comm.sum(npy.vdot(R_1G, mR_G))
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
                i1 += 1
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = npy.linalg.inv(A_ii)
            except npy.linalg.LinAlgError:
                alpha_i = npy.zeros(iold)
                alpha_i[-1] = 1.0
            else:
                alpha_i = B_ii.sum(1)
                try:
                    # Normalize:
                    alpha_i /= alpha_i.sum()
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0
            
            # Calculate new input density:
            nt_G[:] = 0.0
            #for D_p, D_ip, dD_ip in self.D_a:
            for D in D_ap:
                D[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                axpy(alpha, self.nt_iG[i], nt_G)
                axpy(alpha * beta, self.R_iG[i], nt_G)
                for D_p, D_ip, dD_ip in zip(D_ap, self.D_iap[i], self.dD_iap[i]):
                    axpy(alpha, D_ip, D_p)
                    axpy(alpha * beta, dD_ip, D_p)


        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G.copy())
        self.D_iap.append([])
        for D_p in D_ap:
            self.D_iap[-1].append(D_p.copy())


class Mixer(BaseMixer):
    """Mix spin up and down densities separately"""


    def initialize(self, paw):
        self.mixers = []
        for s in range(paw.nspins):
            mixer = BaseMixer(self.beta, self.nmaxold,
                              self.metric_type, self.weight)
            mixer.initialize_metric(paw.gd)
            self.mixers.append(mixer)
    
    def mix(self, density):
        """Mix pseudo electron densities."""

        nt_sG = density.nt_sG
        D_asp = [nucleus.D_sp for nucleus in density.my_nuclei]
        D_sap = []
        for s in range(density.nspins):
            D_sap.append([D[s] for D in D_asp])
        for nt_G, D_ap, mixer in zip(nt_sG, D_sap, self.mixers):
            mixer.mix(nt_G, D_ap)

    def reset(self):
        for mixer in self.mixers:
            mixer.reset()

    def get_charge_sloshing(self):
        """Return number of electrons moving around.

        Calculated as the integral of the absolute value of the change
        of the density from input to output."""
        
        if self.mixers[0].dNt is None:
            return None
        return sum([mixer.dNt for mixer in self.mixers])


class MixerSum(BaseMixer):
    """For pseudo electron densities, mix the total charge density and for
    density matrices, mix spin up and densities separately.
    Magnetization density is not mixed, i.e new magnetization density is used
    """

    def mix(self, density):

        nt_sG = density.nt_sG
        D_asp = [nucleus.D_sp for nucleus in density.my_nuclei]

        # Mix density
        nt_G = density.nt_sG.sum(0)
        BaseMixer.mix(self, nt_G, D_asp)

        # Only new magnetization for spin density
        dnt_G = nt_sG[0] - nt_sG[1]
        dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]

        # Construct new spin up/down densities 
        nt_sG[0] = 0.5 * (nt_G + dnt_G)
        nt_sG[1] = 0.5 * (nt_G - dnt_G)

class MixerSum2(BaseMixer):
    """Mix the total pseudoelectron density and the total density matrices.
    Magnetization density is not mixed, i.e new magnetization density is used.
    """

    def mix(self, density):

        nt_sG = density.nt_sG
        D_asp = [nucleus.D_sp for nucleus in density.my_nuclei]

        # Mix density
        nt_G = density.nt_sG.sum(0)
        D_ap = [D_p[0] + D_p[1] for D_p in D_asp]
        BaseMixer.mix(self, nt_G, D_ap)

        # Only new magnetization for spin density
        dnt_G = nt_sG[0] - nt_sG[1]
        dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]

        # Construct new spin up/down densities 
        nt_sG[0] = 0.5 * (nt_G + dnt_G)
        nt_sG[1] = 0.5 * (nt_G - dnt_G)
        for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
            D_sp[0] = 0.5 * (D_p + dD_p)
            D_sp[1] = 0.5 * (D_p - dD_p)

class MixerDif(BaseMixer):
    """Mix the charge density and magnetization density separately"""
    
    def __init__(self, beta=0.25, nmaxold=3, metric=None, weight=50.0,
                 beta_m=0.7, nmaxold_m=2, metric_m=None, weight_m=10.0):
        """Construct density-mixer object.

        Parameters
        ----------
        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        metrix: None, 'old' or 'new'
            Type of metric to use.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.metric_type = metric
        self.weight = weight

        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.metric_type_m = metric_m
        self.weight_m = weight_m
        self.dNt = None

        self.mix_rho = False


    def initialize(self, paw):
        
        assert paw.nspins == 2
        self.mixer = BaseMixer(self.beta, self.nmaxold,
                                   self.metric_type, self.weight)
        self.mixer.initialize_metric(paw.gd)
        self.mixer_m = BaseMixer(self.beta_m, self.nmaxold_m,
                                   self.metric_type_m, self.weight_m)
        self.mixer_m.initialize_metric(paw.gd)

    def reset(self):
        self.mixer.reset()
        self.mixer_m.reset()

    def mix(self, density):

        nt_sG = density.nt_sG
        D_asp = [nucleus.D_sp for nucleus in density.my_nuclei]

        # Mix density
        nt_G = density.nt_sG.sum(0)
        D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
        self.mixer.mix(nt_G, D_ap)

        # Mix magnetization
        dnt_G = nt_sG[0] - nt_sG[1]
        dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
        self.mixer_m.mix(dnt_G, dD_ap)

        # Construct new spin up/down densities 
        nt_sG[0] = 0.5 * (nt_G + dnt_G)
        nt_sG[1] = 0.5 * (nt_G - dnt_G)
        for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
            D_sp[0] = 0.5 * (D_p + dD_p)
            D_sp[1] = 0.5 * (D_p - dD_p)
            

    def get_charge_sloshing(self):
        if self.mixer.dNt is None:
            return None
        return self.mixer.dNt


class MixerRho(BaseMixer):
    def initialize(self, paw):
    
        self.mix_rho = True
        self.initialize_metric(paw.finegd)
    
    def mix(self, density):
        """Mix pseudo electron densities."""

        rhot_g = density.rhot_g
        BaseMixer.mix(self, rhot_g, [])

class MixerRho2(BaseMixer):
    def initialize(self, paw):
    
        self.mix_rho = True
        self.initialize_metric(paw.finegd)
    
    def mix(self, density):
        """Mix pseudo electron densities."""

        rhot_g = density.rhot_g
        D_asp = [nucleus.D_sp for nucleus in density.my_nuclei]
        BaseMixer.mix(self, rhot_g, D_asp)

