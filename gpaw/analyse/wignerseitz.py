from math import sqrt, pi
import numpy as npy

from gpaw.utilities import pack, pack2, wignerseitz
from gpaw.mpi import MASTER

class WignerSeitz:
    def __init__(self, gd, nuclei):
        """Find the grid points nearest to the atoms"""

        self.nuclei = nuclei
        self.gd = gd

        n = len(self.nuclei)
        atom_c = npy.empty((n, 3))
        for a, nucleus in enumerate(nuclei):
            atom_c[a] = nucleus.spos_c * gd.N_c

        # define the atom index for each grid point 
        atom_index = gd.empty(dtype=int)
        wignerseitz(atom_index, atom_c, gd.beg_c, gd.end_c)
        self.atom_index = atom_index

    def expand(self, density):
        """Expand a smooth density in Wigner-Seitz cells around the atoms"""
        n = len(self.nuclei)
        weights = npy.empty((n,))
        for a in range(n):
            mask = npy.where(self.atom_index == a, density, 0.0)
            # XXX Optimize! No need to integrate in zero-region
            weights[a] = self.gd.integrate(mask)

        return weights

    def expand_density(self, nt_G, s, nspins):
        """Get the weights of spin-density in Wigner-Seitz cells
        around the atoms. The spin index and number of spins are
        needed for the augmentation sphere corrections."""
        weights_a = self.expand(nt_G)
        for a, nucleus in enumerate(self.nuclei):
            weights_a[a] += nucleus.get_density_correction(s, nspins)
        return weights_a
    
    def expand_wave_function(self, psit_G, u, n):
        """Get the weights of wave function in Wigner-Seitz cells
        around the atoms. The spin-k-point index u and band number n
        are needed for the augmentation sphere corrections."""
        # smooth part
        weigths = self.expand(psit_G**2)

        # add augmentation sphere corrections
        for a, nucleus in enumerate(self.nuclei):
            P_i = nucleus.P_uni[u, n]
            P_p = pack(npy.outer(P_i, P_i))
            Delta_p = sqrt(4 * pi) * nucleus.setup.Delta_pL[:, 0]
            weigths[a] += npy.dot(Delta_p, P_p) 

        return weigths

class LDOSbyBand:
    """Base class for a band by band LDOS"""
    
    def by_element(self):
        # get element indicees
        elemi = {}
        for i, nucleus in enumerate(self.paw.nuclei):
            symbol = nucleus.setup.symbol
            if elemi.has_key(symbol):
                elemi[symbol].append(i)
            else:
                elemi[symbol] = [i]
        for key in elemi.keys():
            elemi[key] = self.get(elemi[key])
        return elemi

class WignerSeitzLDOS(LDOSbyBand):
    """Class to get the unfolded LDOS defined by Wigner-Seitz cells"""
    def __init__(self, paw):
        self.paw = paw
        self.ws = WignerSeitz(paw.gd, paw.nuclei)
        
        nu = paw.nkpts * paw.nspins
        ldos = npy.empty((nu, paw.nbands, len(paw.nuclei)))
        for u, kpt in enumerate(paw.kpt_u):
            for n, psit_G in enumerate(kpt.psit_nG):
                ldos[u, n, :] = ws.expand_wave_function(psit_G, u, n)

    def write(self, filename, slavewrite=False):
        if self.world.rank == MASTER or slavewrite:
            paw = self.paw
            f = open(filename, 'w')

            nn = len(paw.nuclei)
            
            
            for k in range(paw.nkpts):
                for s in range(paw.nspins):
                    u = s*paw.nkpts + k
                    for n in range(paw.nbands): 
                        # avery: Added dummy loop body to make compiling work.
                        1
