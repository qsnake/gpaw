# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Ref. to Kresse-paper ... XXX
"""

import Numeric as num
import LinearAlgebra as linalg

from gpaw.utilities.blas import axpy
from gpaw.operators import Operator


class BaseMixer:
    """Pulay density mixer."""
    
    def __init__(self, beta=0.25, nmaxold=3, metric=None, weight=50.0):
        """Mixer(beta, nold) -> mixer object.

        beta:  Mixing parameter between zero and one (one is most
               aggressive).
               
        nold:  Maximum number of old densities."""

        self.beta = beta
        self.nmaxold = nmaxold
        self.metric_type = metric
        self.weight = weight

        self.dNt = None

    def initialize(self, gd, nspins=None):
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
                                   gd, True, num.Float).apply
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
                                   gd, False, num.Float).apply
            self.mR_G = gd.empty()

        else:
            raise RuntimeError('Unknown metric type: "%s".' % self.metric_type)
        
    def reset(self, my_nuclei, s=None):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """
        
        # History for Pulay mixing of densities:
        self.nt_iG = [] # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = num.zeros((0, 0), num.Float)
        self.dNt = None
        
        # Collect atomic density matrices:
        if s is None:
            self.D_a = [(nucleus.D_sp, [], []) for nucleus in my_nuclei]
        else:
            self.D_a = [(nucleus.D_sp[s], [], []) for nucleus in my_nuclei]

    def get_charge_sloshing(self):
        """Return number of electrons moving around.

        Calculated as the integral of the absolute value of the change
        of the density from input to output."""
        
        return self.dNt

    def mix(self, nt_G):
        iold = len(self.nt_iG)
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                for D_p, D_ip, dD_ip in self.D_a:
                    del D_ip[0]
                    del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_G = nt_G - self.nt_iG[-1]
            self.dNt = self.gd.integrate(num.fabs(R_G))
            self.R_iG.append(R_G)
            for D_p, D_ip, dD_ip in self.D_a:
                dD_ip.append(D_p - D_ip[-1])

            # Update matrix:
            A_ii = num.zeros((iold, iold), num.Float)
            i1 = 0
            i2 = iold - 1
            
            if self.metric is None:
                mR_G = R_G
            else:
                mR_G = self.mR_G
                self.metric(R_G, mR_G)
                
            for R_1G in self.R_iG:
                a = self.gd.comm.sum(num.vdot(R_1G, mR_G))
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
                i1 += 1
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = linalg.inverse(A_ii)
            except linalg.LinAlgError:
                alpha_i = num.zeros(iold, num.Float)
                alpha_i[-1] = 1.0
            else:
                alpha_i = num.sum(B_ii, 1)
                try:
                    # Normalize:
                    alpha_i /= num.sum(alpha_i)
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0
            
            # Calculate new input density:
            nt_G[:] = 0.0
            for D_p, D_ip, dD_ip in self.D_a:
                D_p[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                axpy(alpha, self.nt_iG[i], nt_G)
                axpy(alpha * beta, self.R_iG[i], nt_G)
                for D_p, D_ip, dD_ip in self.D_a:
                    axpy(alpha, D_ip[i], D_p)
                    axpy(alpha * beta, dD_ip[i], D_p)


        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G.copy())
        for D_p, D_ip, dD_ip in self.D_a:
            D_ip.append(D_p.copy())


class Mixer(BaseMixer):
    def initialize(self, gd, nspins):
        self.mixers = []
        for s in range(nspins):
            mixer = BaseMixer(self.beta, self.nmaxold,
                              self.metric_type, self.weight)
            mixer.initialize(gd)
            self.mixers.append(mixer)
    
    def mix(self, nt_sG):
        """Mix pseudo electron densities."""

        for nt_G, mixer in zip(nt_sG, self.mixers):
            mixer.mix(nt_G)

    def reset(self, my_nuclei):
        for s, mixer in enumerate(self.mixers):
            mixer.reset(my_nuclei, s)

    def get_charge_sloshing(self):
        """Return number of electrons moving around.

        Calculated as the integral of the absolute value of the change
        of the density from input to output."""
        
        if self.mixers[0].dNt is None:
            return None
        return sum([mixer.dNt for mixer in self.mixers])


class MixerSum(BaseMixer):
    def mix(self, nt_sG):
        nt_G = num.sum(nt_sG)
        BaseMixer.mix(self, nt_G)
        dnt_G = nt_sG[0] - nt_sG[1]
        nt_sG[0] = 0.5 * (nt_G + dnt_G)
        nt_sG[1] = 0.5 * (nt_G - dnt_G)
