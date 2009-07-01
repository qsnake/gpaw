from math import pi, sqrt

import numpy as np
from ase.atoms import Atoms

from gpaw.aseinterface import GPAW
from gpaw.wavefunctions import WaveFunctions
from gpaw.grid_descriptor import EquidistantRadialGridDescriptor
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize
from gpaw.occupations import OccupationNumbers
import gpaw.mpi as mpi


class MakeWaveFunctions:
    def __init__(self, gd):
        self.gd = gd

    def __call__(self, paw, gd, *args):
        paw.gd = self.gd
        return AtomWaveFunctions(self.gd, *args)
    

class AtomWaveFunctions(WaveFunctions):
    def initialize(self, density, hamiltonian, spos_ac):
        r = self.gd.r_g
        N = len(r)
        setup = self.setups[0]
        bf = AtomBasisFunctions(self.gd, setup.phit_j)
        density.initialize_from_atomic_densities(bf)
        hamiltonian.update(density)      

    def add_to_density_from_k_point(self, nt_sG, kpt):
        nt_sG[kpt.s] += np.dot(kpt.f_n / 4 / pi, kpt.psit_nG**2)
        #import pylab as p
        #p.plot(self.gd.r_g, nt_sG[0])
        #p.show();sdg

class AtomPoissonSolver:
    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.relax_method = 0
        self.nn = 1
        
    def initialize(self):
        pass

    def solve(self, vHt_g, rhot_g, charge=0):
        r = self.gd.r_g
        dp = rhot_g * r * self.gd.dr_g
        dq = dp * r
        p = np.add.accumulate(dp[::-1])[::-1]
        q = np.add.accumulate(dq[::-1])[::-1]
        vHt_g[:] = 4 * pi * (p - 0.5 * dp - (q - 0.5 * dq - q[0]) / r)
        return 1
    
class AtomEigensolver:
    def __init__(self, gd, f_sln):
        self.gd = gd
        self.f_sln = f_sln
        self.error = 0.0
        self.initialized = False
        
    def initialize(self, wfs):
        r = self.gd.r_g
        h = r[0]
        N = len(r)
        lmax = len(self.f_sln[0]) - 1
        
        self.T_l = [np.eye(N) * (1.0 / h**2)]
        self.T_l[0].flat[1::N + 1] = -0.5 / h**2
        self.T_l[0].flat[N::N + 1] = -0.5 / h**2
        for l in range(1, lmax + 1):
            self.T_l.append(self.T_l[0] + np.diag(l * (l + 1) / 2.0 / r**2))
            
        self.S_l = [np.eye(N) for l in range(lmax + 1)]
        setup = wfs.setups[0]
        self.pt_j = np.array([[pt(x) * x**l for x in r]
                              for pt, l in zip(setup.pt_j, setup.l_j)])

        dS_ii = setup.O_ii
        i1 = 0
        for pt1, l1 in zip(self.pt_j, setup.l_j):
            i2 = 0
            for pt2, l2 in zip(self.pt_j, setup.l_j):
                if l1 == l2 and l1 <= lmax:
                    self.S_l[l1] += (np.outer(pt1 * r, pt2 * r) *
                                     h * dS_ii[i1, i2]) 
                i2 += 2 * l2 + 1
            i1 += 2 * l1 + 1

        for kpt in wfs.kpt_u:
            kpt.eps_n = np.empty(wfs.nbands)
            kpt.psit_nG = self.gd.empty(wfs.nbands)
            kpt.P_ani = {0: np.zeros((wfs.nbands, len(dS_ii)))}
        
    def iterate(self, hamiltonian, wfs):
        if not self.initialized:
            self.initialize(wfs)
        
        r = self.gd.r_g
        h = r[0]
        N = len(r)
        lmax = len(self.f_sln[0]) - 1
        setup = wfs.setups[0]

        e_n = np.zeros(N)

        for s in range(wfs.nspins):
            dH_ii = unpack(hamiltonian.dH_asp[0][s])
            kpt = wfs.kpt_u[s]
            N1 = 0
            for l in range(lmax + 1):
                H = self.T_l[l] + np.diag(hamiltonian.vt_sg[s])
                i1 = 0
                for pt1, l1 in zip(self.pt_j, setup.l_j):
                    i2 = 0
                    for pt2, l2 in zip(self.pt_j, setup.l_j):
                        if l1 == l2 == l:
                            H += (h * dH_ii[i1, i2] *
                                  np.outer(pt1 * r,  pt2 * r))
                        i2 += 2 * l2 + 1
                    i1 += 2 * l1 + 1
                H0 = H.copy()#XXX
                error = diagonalize(H, e_n, self.S_l[l].copy())
                if error != 0:
                    raise RuntimeError('Diagonalization failed for l=%d.' % l)

                for n in range(len(self.f_sln[s][l])):
                    N2 = N1 + 2 * l + 1
                    kpt.eps_n[N1:N2] = e_n[n]
                    kpt.psit_nG[N1:N2] = H[n] / r / sqrt(h)
                    i1 = 0
                    for pt, ll in zip(self.pt_j, setup.l_j):
                        i2 = i1 + 2 * ll + 1
                        if ll == l:
                            P = np.dot(kpt.psit_nG[N1], pt * r**2) * h
                            kpt.P_ani[0][N1:N2, i1:i2] = P * np.eye(2 * l + 1)
                        i1 = i2
                    N1 = N2
                    
class AtomLocalizedFunctionsCollection:
    def __init__(self, gd, spline_aj):
        self.gd = gd
        spline = spline_aj[0][0]
        self.b_g = np.array([spline(x) for x in gd.r_g]) / sqrt(4 * pi)

    def set_positions(self, spos_ac):
        pass
    
    def add(self, a_xG, c_axi=1.0, q=-1):
        if isinstance(c_axi, float):
            a_xG += c_axi * self.b_g
        else:
            a_xG += c_axi[0][0] * self.b_g

    def integrate(self, a_g, c_ai):
        assert a_g.ndim == 1
        c_ai[0][0] = self.gd.integrate(a_g, self.b_g)
        c_ai[0][1:] = 0.0

class AtomBasisFunctions:
    def __init__(self, gd, phit_j):
        self.gd = gd
        self.bl_j = []
        self.Mmax = 0
        for phit in phit_j:
            l = phit.get_angular_momentum_number()
            self.bl_j.append((np.array([phit(x) * x**l for x in gd.r_g]), l))
            self.Mmax += 2 * l + 1
        self.atom_indices = [0]
        self.my_atom_indices = [0]
                      
    def set_positions(self, spos_ac):
        pass
    
    def add_to_density(self, nt_sG, f_asi):
        i = 0
        for b_g, l in self.bl_j:
            nt_sG += f_asi[0][:, i:i + 1] * (2 * l + 1) / 4 / pi * b_g**2
            i += 2 * l + 1

class AtomGridDescriptor(EquidistantRadialGridDescriptor):
    def __init__(self, h, rcut):
        ng = int(float(rcut) / h + 0.5) - 1
        EquidistantRadialGridDescriptor.__init__(self, h, ng)
        self.sdisp_cd = np.empty((3, 2))
        self.comm = mpi.serial_comm
        self.pbc_c = np.zeros(3, bool)
        self.cell_c = np.ones(3) * rcut
        self.N_c = np.ones(3, dtype=int) * 2 * ng
        self.h_c = np.ones(3) * h
    def get_ranks_from_positions(self, spos_ac):
        return np.array([0])
    def refine(self):
        return self
    def is_non_orthogonal(self):
        return True
    def get_lfc(self, gd, spline_aj):
        return AtomLocalizedFunctionsCollection(gd, spline_aj)
    def integrate(self, a_xg, b_xg=None, global_integral=True):
        """Integrate function(s) in array over domain."""
        if b_xg is None:
            return np.dot(a_xg, self.dv_g)
        else:
            return np.dot(a_xg * b_xg, self.dv_g)
    def calculate_dipole_moment(self, rhot_g):
        return np.zeros(3)

class AtomOccupations(OccupationNumbers):
    def __init__(self, f_sln):
        self.f_sln = f_sln
        OccupationNumbers.__init__(self, None, len(f_sln))
        
    def calculate(self, wfs):
        OccupationNumbers.calculate(self, wfs)
        for s in range(self.nspins):
            n1 = 0
            for l, f_n in enumerate(self.f_sln[s]):
                for f in f_n:
                    n2 = n1 + 2 * l + 1
                    wfs.kpt_u[s].f_n[n1:n2] = f / float(2 * l + 1)
                    n1 = n2
        self.calculate_band_energy(wfs.kpt_u)
        if self.nspins == 2:
            self.magmom = wfs.kpt_u[0].f_n.sum() - wfs.kpt_u[1].f_n.sum()

    def get_fermi_level(self):
        raise NotImplementedError

class AtomPAW(GPAW):
    def __init__(self, symbol, f_sln, h=0.05, rcut=10.0, **kwargs):
        assert len(f_sln) in [1, 2]
        self.symbol = symbol
        
        gd = AtomGridDescriptor(h, rcut)
        GPAW.__init__(self,
                      mode=MakeWaveFunctions(gd),
                      eigensolver=AtomEigensolver(gd, f_sln),
                      poissonsolver=AtomPoissonSolver(),
                      stencils=(1, 9),
                      nbands=sum([(2 * l + 1) * len(f_n)
                                  for l, f_n in enumerate(f_sln[0])]),
                      **kwargs)
        self.occupations = AtomOccupations(f_sln)
        self.initialize(Atoms(symbol, calculator=self))
        self.calculate(converge=True)
        
    def state_iter(self):
        """Yield the tuples (l, n, f, eps, psit_G) of states.

        Skips degenerate states."""
        f_sln = self.occupations.f_sln
        assert len(f_sln) == 1, 'Not yet implemented with more spins'
        f_ln = f_sln[0]
        kpt = self.wfs.kpt_u[0]

        band = 0
        for l, f_n in enumerate(f_ln):
            for n, f in enumerate(f_n):
                psit_G = kpt.psit_nG[band]
                eps = kpt.eps_n[band]
                yield l, n, f, eps, psit_G
                band += 2 * l + 1

    def extract_basis_functions(self, basis_name='atompaw.sz'):
        """Create BasisFunctions object with pseudo wave functions."""
        from gpaw.basis_data import Basis, BasisFunction
        assert self.wfs.nspins == 1

        basis = Basis(self.symbol, basis_name, readxml=False)
        basis.d = self.gd.h
        basis.ng = self.gd.ng
        basis.generatorattrs = {} # attrs of the setup maybe
        basis.generatordata = 'AtomPAW' # version info too?
        
        bf_j = basis.bf_j
        for l, n, f, eps, psit_G in self.state_iter():
            bf = BasisFunction(l, self.gd.rcut, psit_G * np.sign(psit_G[-1]), 
                               '%s%d e=%.3f f=%.3f' % ('spdf'[l], n, eps, f))
            bf_j.append(bf)
        return basis
