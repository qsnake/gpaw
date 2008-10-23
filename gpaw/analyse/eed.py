import sys
import numpy as npy

from ase.units import Bohr, Hartree
from ase.parallel import paropen

import _gpaw

class ExteriorElectronDensity:
    """Exterior electron density to describe MIES spectra.

    Simple approach to describe MIES spectra after
    Y. Harada et al., Chem. Rev. 97 (1997) 1897
    """
    def __init__(self, gd, nuclei):
        """Find the grid points outside of the van der Waals radii 
        of the atoms"""

        self.nuclei = nuclei
        self.gd = gd

        n = len(self.nuclei)
        atom_c = npy.empty((n, 3))
        vdWradius = npy.empty((n))
        for a, nucleus in enumerate(nuclei):
            atom_c[a] = nucleus.spos_c * gd.domain.cell_c
            vdWradius[a] = self.get_vdWradius(nucleus.setup.Z)

        # define the exterior region mask
        mask = gd.empty(dtype=int)
        _gpaw.eed_region(mask, atom_c, gd.beg_c, gd.end_c, gd.h_c, vdWradius)
        self.mask = mask

    def get_weight(self, psit_G):
        """Get the weight of a wave function in the exterior region
        (outside of the van der Waals radius). The augmentation sphere
        is assumed to be smaller as the van der Waals radius and hence 
        does not contribute."""

        # smooth part
        weigth = self.gd.integrate(npy.where(self.mask == 1, 
                                             psit_G**2, 0.0))

        return weigth

    def get_vdWradius(self, Z):
        """Return van der Waals radius in Bohr"""
        r = vdW_radii[Z] / Bohr
        if npy.isnan(r):
            msg = 'van der Waals radius for Z=' + str(Z) + ' not known!'
            raise RuntimeError(msg)
        else:
            return r
        
    def write_mies_weights(self, paw, file=None):
        if paw.nkpts > 1:
            raise NotImplementedError # XXXX TODO

        out = sys.stdout
        if file is None:
            file = 'mies.dat'

        if isinstance(file, str):
            out = paropen(file, 'aw')
        else:
            out = file

        print >> out, '# exterior electron density weights after'
        print >> out, '# Y. Harada et al., Chem. Rev. 97 (1997) 1897'
        if paw.nspins == 1:
            print >> out, '# Band   energy      occ         weight'
            kpt = paw.kpt_u[0]
            for n in range(paw.nbands):
                print  >> out, '%4d  %10.5f  %10.5f  %10.5f' % \
                    (n, 
                     kpt.eps_n[n] * Hartree,
                     kpt.f_n[n], 
                     self.get_weight(kpt.psit_nG[n]) )
                if hasattr(out, 'flush'):
                    out.flush()
        else:
            print >> out, '# Band   energy      occ         weight     energy      occ         weight'
            kpta = paw.kpt_u[0]
            kptb = paw.kpt_u[1]
            for n in range(paw.nbands):
                print  >> out, '%4d  %10.5f  %10.5f  %10.5f  %10.5f    %10.5f  %10.5f' % \
                    (n, 
                     kpta.eps_n[n] * Hartree,
                     kpta.f_n[n], 
                     self.get_weight(kpta.psit_nG[n]),
                     kptb.eps_n[n] * Hartree,
                     kptb.f_n[n], 
                     self.get_weight(kptb.psit_nG[n]),
                     )
                if hasattr(out, 'flush'):
                    out.flush()
                
# van der Waals radii in [A] taken from
# http://www.webelements.com/periodicity/van_der_waals_radius/
vdW_radii = npy.array([
 npy.nan, # X
 1.20, # H
 1.40, # He
 1.82, # Li
 npy.nan, # Be
 npy.nan, # B
 1.70, # C
 1.55, # N
 1.52, # O
 1.47, # F
 1.54, # Ne
 2.27, # Na
 1.73, # Mg
 npy.nan, # Al
 2.10, # Si
 1.80, # P
 1.80, # S
 1.75, # Cl
 1.88, # Ar
 2.75, # K
 npy.nan, # Ca
 npy.nan, # Sc
 npy.nan, # Ti
 npy.nan, # V
 npy.nan, # Cr
 npy.nan, # Mn
 npy.nan, # Fe
 npy.nan, # Co
 1.63, # Ni
 1.40, # Cu
 1.39, # Zn
 1.87, # Ga
 npy.nan, # Ge
 1.85, # As
 1.90, # Se
 1.85, # Br
 2.02, # Kr
 npy.nan, # Rb
 npy.nan, # Sr
 npy.nan, # Y
 npy.nan, # Zr
 npy.nan, # Nb
 npy.nan, # Mo
 npy.nan, # Tc
 npy.nan, # Ru
 npy.nan, # Rh
 1.63, # Pd
 1.72, # Ag
 1.58, # Cd
 1.93, # In
 2.17, # Sn
 npy.nan, # Sb
 2.06, # Te
 1.98, # I
 2.16, # Xe
 npy.nan, # Cs
 npy.nan, # Ba
 npy.nan, # La
 npy.nan, # Ce
 npy.nan, # Pr
 npy.nan, # Nd
 npy.nan, # Pm
 npy.nan, # Sm
 npy.nan, # Eu
 npy.nan, # Gd
 npy.nan, # Tb
 npy.nan, # Dy
 npy.nan, # Ho
 npy.nan, # Er
 npy.nan, # Tm
 npy.nan, # Yb
 npy.nan, # Lu
 npy.nan, # Hf
 npy.nan, # Ta
 npy.nan, # W
 npy.nan, # Re
 npy.nan, # Os
 npy.nan, # Ir
 1.75, # Pt
 1.66, # Au
 1.55, # Hg
 1.96, # Tl
 2.02, # Pb
 npy.nan, # Bi
 npy.nan, # Po
 npy.nan, # At
 npy.nan, # Rn
 npy.nan, # Fr
 npy.nan, # Ra
 npy.nan, # Ac
 npy.nan, # Th
 npy.nan, # Pa
 1.86, # U
 npy.nan, # Np
 npy.nan, # Pu
 npy.nan, # Am
 npy.nan, # Cm
 npy.nan, # Bk
 npy.nan, # Cf
 npy.nan, # Es
 npy.nan, # Fm
 npy.nan, # Md
 npy.nan, # No
 npy.nan]) # Lr
