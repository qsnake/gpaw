# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
This module contains the Spin class.  A spin-polarized calculation
has two spin objects and a spin-paired calculation only one.  Each
spin object has an effective potential, a pseudo density and methods
to updated the density.  Updating the density is done using Pulay
mixing.

Ref. to Kresse-paper ... XXX
"""

import Numeric as num
import LinearAlgebra as linalg

from gpaw.utilities.blas import axpy


class Mixer:
    """Pulay density mixer."""
    
    def __init__(self, beta, nold, nspins):
        """Mixer(beta, nold) -> mixer object.

        beta:  Mixing parameter between zero and one (one is most
               aggressive).
               
        nold:  Maximum number of old densities."""
        
        self.beta = beta
        self.nmaxold = nold

        self.mixers = [Mixer1(beta, nold) for s in range(nspins)]

    def reset(self, my_nuclei):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """
        
        for s, mixer in enumerate(self.mixers):
            mixer.reset(my_nuclei, s)

    def mix(self, nt_sG, comm):
        """Mix pseudo electron densities."""

        for nt_G, mixer in zip(nt_sG, self.mixers):
            mixer.mix(nt_G, comm)


class Mixer1:
    def __init__(self, beta, nold):
        self.nmaxold = nold
        self.beta = beta

    def reset(self, my_nuclei, s):
        # History for Pulay mixing of densities:
        self.nt_iG = [] # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = num.zeros((0, 0), num.Float)

        # Collect atomic density matrices:
        # XXX ref to nucleus!n
        self.D_a = [(nucleus.D_sp[s], [], []) for nucleus in my_nuclei]

    def mix(self, nt_G, comm):
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
            self.R_iG.append(R_G)
            for D_p, D_ip, dD_ip in self.D_a:
                dD_ip.append(D_p - D_ip[-1])

            # Update matrix:
            A_ii = num.zeros((iold, iold), num.Float)
            i1 = 0
            i2 = iold - 1
            for R_1G in self.R_iG:
                a = comm.sum(num.vdot(R_1G, R_G))
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


class MixerSum:
    """Pulay density mixer."""
    
    def __init__(self, beta, nold, nspins):
        """Mixer(beta, nold) -> mixer object.

        beta:  Mixing parameter between zero and one (one is most
               aggressive).
               
        nold:  Maximum number of old densities.
        """

        self.nmaxold = nold
        self.beta = beta

    def reset(self, my_nuclei):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """
        
        # History for Pulay mixing of densities:
        self.nt_iG = [] # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = num.zeros((0, 0), num.Float)

        # Collect atomic density matrices:
        self.D_a = [(nucleus.D_sp, [], []) for nucleus in my_nuclei]

    def mix(self, nt_sG, comm):
        """Mix pseudo electron densities."""

        nspins = len(nt_sG)
        if nspins == 1:
            nt_G = nt_sG[0].copy()
        else:
            nt_G = num.sum(nt_sG)

        iold = len(self.nt_iG)
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                for D_sp, D_isp, dD_isp in self.D_a:
                    del D_isp[0]
                    del dD_isp[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_G = nt_G - self.nt_iG[-1]
            self.R_iG.append(R_G)
            for D_sp, D_isp, dD_isp in self.D_a:
                dD_isp.append(D_sp - D_isp[-1])

            # Update matrix:
            A_ii = num.zeros((iold, iold), num.Float)
            i1 = 0
            i2 = iold - 1
            for R_1G in self.R_iG:
                a = comm.sum(num.vdot(R_1G, R_G))
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
            for D_sp, D_isp, dD_isp in self.D_a:
                D_sp[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                axpy(alpha, self.nt_iG[i], nt_G)
                axpy(alpha * beta, self.R_iG[i], nt_G)
                for D_sp, D_isp, dD_isp in self.D_a:
                    axpy(alpha, D_isp[i], D_sp)
                    axpy(alpha * beta, dD_isp[i], D_sp)

        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G)
        for D_sp, D_isp, dD_isp in self.D_a:
            D_isp.append(D_sp.copy())

        if nspins == 1:
            nt_sG[0] = nt_G
        else:
            dnt_G = nt_sG[0] - nt_sG[1]
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)
