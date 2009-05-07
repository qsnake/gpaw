import numpy as np
from ase.units import Bohr

from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.lfc import NewLocalizedFunctionsCollection as LFC
import gpaw.mpi as mpi

class LocalizedFunctions:
    def __init__(self, gd, f_iG, corner_c, vt_G=None):
        self.gd = gd
        #assert gd.is_orthogonal()
        assert gd.is_not_orthogonal()
        self.size_c = np.array(f_iG.shape[1:4])
        self.f_iG = f_iG
        self.corner_c = corner_c
        self.vt_G = vt_G

    def apply_t(self):
        """Apply kinetic energy operator and return new object."""
        p = 2  # padding
        newsize_c = self.size_c + 2 * p
        gd = GridDescriptor(N_c=newsize_c + 1,
                            cell_cv=self.gd.h_c * (newsize_c + 1),
                            pbc_c=False,
                            comm=mpi.serial_comm)
        T = Transformer(gd, nn=p)
        f_iG = np.zeros((len(self.f_iG),) + tuple(newsize_c))
        f_ig[:, p:-p, p:-p, p:-p] = self.f_iG
        Tf_iG = np.empty_like(f_iG)
        T.apply(f_iG, Tf_iG)
        return LocalizedFunctions(Tf_iG, self.corner_c - p)
        
    def overlap(self, other):
        start_c = np.maximum(self.corner_c, other.corner_c)
        stop_c = np.minimum(self.corner_c + self.size_c,
                            other.corner_c + other.size_c)
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c - self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]].reshape((len(self.f_iG), -1))
            bstart_c = start_c - other.corner_c
            bstop_c = stop_c - other.corner_c
            b_iG = other.f_iG[:,
                bstart_c[0]:bstop_c[0],
                bstart_c[1]:bstop_c[1],
                bstart_c[2]:bstop_c[2]].reshape((len(other.f_iG), -1))
            if self.vt_G is not None:
                a_iG *= self.vt_G[start_c[0]:stop_c[0],
                                  start_c[1]:stop_c[1],
                                  start_c[2]:stop_c[2]].reshape((-1,))
            return self.gd.dv * np.inner(a_iG, b_iG)
        else:
            return 0.0

    def __or__(self, other):
        if isinstance(other, LocalizedFunctions):
            return self.overlap(other)
        
        return LocalizedFunctions(self.gd, self.f_iG, self.corner_c, other)

class WannierFunction(LocalizedFunctions):
    def __init__(self, gd, wanf_G, corner_c):
        LocalizedFunctions.__init__(self, gd, wanf_G[np.newaxis, :, :, :],
                                    corner_c)

class AtomCenteredFunctions(LocalizedFunctions):
    def __init__(self, gd, spline_j, spos_c):
        rcut = max([spline.get_cutoff() for spline in spline_j])
        corner_c = np.ceil(spos_c * gd.N_c - rcut / gd.h_c).astype(int)
        size_c = np.ceil(spos_c * gd.N_c + rcut / gd.h_c).astype(int) - corner_c
        smallgd = GridDescriptor(N_c=size_c + 1,
                                 cell_cv=gd.h_c * (size_c + 1),
                                 pbc_c=False,
                                 comm=mpi.serial_comm)
        lfc = LFC(smallgd, [spline_j])
        lfc.set_positions((spos_c[np.newaxis, :] * gd.N_c - corner_c + 1) /
                          smallgd.N_c)
        ni = lfc.Mmax
        f_iG = smallgd.zeros(ni)
        lfc.add(f_iG, {0: np.eye(ni)})
        LocalizedFunctions.__init__(self, gd, f_iG, corner_c)

class STM:
    def __init__(self, tip, surface):
        self.tip = tip
        self.srf = surface

        tgd = tip.gd
        sgd = surface.gd

        #assert tgd.h_cv == sgd.h_cv
        assert not (tgd.h_c - sgd.h_c).any()

    def initialize(self, tip_atom_index, dmin=2.0):
        dmin /= Bohr
        tip_pos_av = self.tip.atoms.get_positions() / Bohr
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        tip_zmin = tip_pos_av[tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()

        offset_c = (tip_pos_av[tip_atom_index] / self.tip.gd.h_c).astype(int)

        tip_zmin_a = np.empty(len(tip_pos_av))
        
        for a, setup in enumerate(self.tip.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            tip_zmin_a[a] = tip_pos_av[a, 2] - rcutmax - tip_zmin

        srf_zmax_a = np.empty(len(srf_pos_av))
        for a, setup in enumerate(self.srf.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            srf_zmax_a[a] = srf_pos_av[a, 2] + rcutmax - srf_zmax

        tip_indices = np.where(tip_zmin_a < srf_zmax_a.max() - dmin)[0]  
        srf_indices = np.where(srf_zmax_a > tip_zmin_a.min() + dmin)[0]  
        print 'Tip atoms:', tip_indices
        print 'Surface atoms:', srf_indices
        
        self.tip_functions_kin = []
        self.tip_functions = []
        for a in tip_indices:
            setup = self.tip.wfs.setups[a]
            spos_c = tip_pos_av[a] / self.tip.gd.cell_c
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.tip.gd, [phit], spos_c)
                tip_functions.append(f)
                tip_functions_kin.append(f.apply_t())

        self.srf_functions = []
        for a in srf_indices:
            setup = self.srf.wfs.setups[a]
            spos_c = srf_pos_av[a] / self.srf.gd.cell_c
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.srf.gd, [phit], spos_c)
                srf_functions.append(f)
                
        #self.gd = GridDescriptor()
