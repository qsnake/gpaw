import sys

import numpy as np
from ase.units import Bohr, Hartree
from ase.parallel import paropen

import _gpaw
from gpaw.io.fmf import FMF

class ExteriorElectronDensity:
    """Exterior electron density to describe MIES spectra.

    Simple approach to describe MIES spectra after
    Y. Harada et al., Chem. Rev. 97 (1997) 1897
    """
    def __init__(self, gd, atoms):
        """Find the grid points outside of the van der Waals radii 
        of the atoms"""

        assert gd.orthogonal
        self.gd = gd

        n = len(atoms)
        atom_c = atoms.positions / Bohr
        vdWradius = np.empty((n))
        for a, atom in enumerate(atoms):
            vdWradius[a] = self.get_vdWradius(atom.get_atomic_number())

        # define the exterior region mask
        mask = gd.empty(dtype=int)
        _gpaw.eed_region(mask, atom_c, gd.beg_c, gd.end_c,
                         gd.h_cv.diagonal().copy(), vdWradius)
        self.mask = mask

    def get_weight(self, psit_G):
        """Get the weight of a wave function in the exterior region
        (outside of the van der Waals radius). The augmentation sphere
        is assumed to be smaller than the van der Waals radius and hence 
        does not contribute."""

        # smooth part
        weigth = self.gd.integrate(np.where(self.mask == 1, 
                                             psit_G * psit_G.conj(), 0.0))

        return weigth

    def get_vdWradius(self, Z):
        """Return van der Waals radius in Bohr"""
        r = vdW_radii[Z] / Bohr
        if np.isnan(r):
            msg = 'van der Waals radius for Z=' + str(Z) + ' not known!'
            raise RuntimeError(msg)
        else:
            return r
        
    def write_mies_weights(self, wfs, file=None):
        if file is None:
            file = 'eed_mies.dat'

        if isinstance(file, str):
            out = paropen(file, 'aw')
        else:
            out = file

        fmf = FMF(['exterior electron density weights after',
                   'Y. Harada et al., Chem. Rev. 97 (1997) 1897'])
        print >> out, fmf.header(),
        print >> out, fmf.data(['band index: n',
                                'k-point index: k',
                                'spin index: s',
                                'k-point weight: weight',
                                'energy: energy [eV]',
                                'occupation number: occ',
                                'relative EED weight: eed_weight']),
        
        print >> out, '#; n   k s   weight      energy         occ  eed_weight'
        for kpt in wfs.kpt_u:
            for n in range(wfs.nbands):
                print  >> out, '%4d %3d %1d %8.5f  %10.5f  %10.5f  %10.5f' % \
                    (n, kpt.k, kpt.s, kpt.weight,
                     kpt.eps_n[n] * Hartree,
                     kpt.f_n[n], 
                     self.get_weight(kpt.psit_nG[n])
                     )
                if hasattr(out, 'flush'):
                    out.flush()
                
# van der Waals radii in [A] taken from
# http://www.webelements.com/periodicity/van_der_waals_radius/
vdW_radii = np.array([
 np.nan, # X
 1.20, # H
 1.40, # He
 1.82, # Li
 np.nan, # Be
 np.nan, # B
 1.70, # C
 1.55, # N
 1.52, # O
 1.47, # F
 1.54, # Ne
 2.27, # Na
 1.73, # Mg
 np.nan, # Al
 2.10, # Si
 1.80, # P
 1.80, # S
 1.75, # Cl
 1.88, # Ar
 2.75, # K
 np.nan, # Ca
 np.nan, # Sc
 np.nan, # Ti
 np.nan, # V
 np.nan, # Cr
 np.nan, # Mn
 np.nan, # Fe
 np.nan, # Co
 1.63, # Ni
 1.40, # Cu
 1.39, # Zn
 1.87, # Ga
 np.nan, # Ge
 1.85, # As
 1.90, # Se
 1.85, # Br
 2.02, # Kr
 np.nan, # Rb
 np.nan, # Sr
 np.nan, # Y
 np.nan, # Zr
 np.nan, # Nb
 np.nan, # Mo
 np.nan, # Tc
 np.nan, # Ru
 np.nan, # Rh
 1.63, # Pd
 1.72, # Ag
 1.58, # Cd
 1.93, # In
 2.17, # Sn
 np.nan, # Sb
 2.06, # Te
 1.98, # I
 2.16, # Xe
 np.nan, # Cs
 np.nan, # Ba
 np.nan, # La
 np.nan, # Ce
 np.nan, # Pr
 np.nan, # Nd
 np.nan, # Pm
 np.nan, # Sm
 np.nan, # Eu
 np.nan, # Gd
 np.nan, # Tb
 np.nan, # Dy
 np.nan, # Ho
 np.nan, # Er
 np.nan, # Tm
 np.nan, # Yb
 np.nan, # Lu
 np.nan, # Hf
 np.nan, # Ta
 np.nan, # W
 np.nan, # Re
 np.nan, # Os
 np.nan, # Ir
 1.75, # Pt
 1.66, # Au
 1.55, # Hg
 1.96, # Tl
 2.02, # Pb
 np.nan, # Bi
 np.nan, # Po
 np.nan, # At
 np.nan, # Rn
 np.nan, # Fr
 np.nan, # Ra
 np.nan, # Ac
 np.nan, # Th
 np.nan, # Pa
 1.86, # U
 np.nan, # Np
 np.nan, # Pu
 np.nan, # Am
 np.nan, # Cm
 np.nan, # Bk
 np.nan, # Cf
 np.nan, # Es
 np.nan, # Fm
 np.nan, # Md
 np.nan, # No
 np.nan]) # Lr
