"""Perdew-Zunger SIC to DFT functionals.

Self-consistent minimization of self-interaction corrected
functionals (Perdew-Zunger).
"""

import sys
from math import pi, sqrt, cos, sin, log10, exp, atan2

import numpy as np
from ase.units import Bohr, Hartree

from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional
from gpaw.poisson import PoissonSolver
from gpaw.utilities import pack, unpack
from gpaw.lfc import LFC
import gpaw.mpi as mpi
import _gpaw


def matrix_exponential(G_nn, dlt):

    """Computes the matrix exponential of an antihermitian operator

       U = exp(dlt*G)
        
    G_nn: ndarray
        anti-hermitian (skew-symmetric) matrix.
            
    dlt: float
        scaling factor for G_nn.
    """
    
    ndim = G_nn.shape[1]
    w_n = np.zeros((ndim), dtype=float)

    V_nn = np.zeros((ndim, ndim), dtype=complex)
    O_nn = np.zeros((ndim, ndim), dtype=complex)
    if G_nn.dtype==complex:
        V_nn = 1j*G_nn.real + G_nn.imag
    else:
        V_nn = 1j*G_nn.real

    diagonalize(V_nn, w_n)
    #
    O_nn  = np.diag(np.exp(1j*dlt*w_n))
    #
    if G_nn.dtype==complex:
        U_nn = np.dot(V_nn.T.conj(),np.dot(O_nn, V_nn)).copy()
    else:
        U_nn = np.dot(V_nn.T.conj(),np.dot(O_nn, V_nn)).real.copy()
    #        
    return U_nn


def ortho(W_nn, maxerr=1E-10):

    """ Orthogonalizes the column vectors of a matrix by Symmetric
        Loewdin orthogonalization

    W_nn: ndarray
        unorthonormal matrix.
            
    maxerr: float
        maximum error for using explicit diagonalization.

    """

    ndim = np.shape(W_nn)[1]
    #
    # overlap matrix
    O_nn = np.dot(W_nn, W_nn.T.conj())
    #
    # check error in orthonormality
    err = np.sum(np.abs(O_nn - np.eye(ndim)))
    if (err < maxerr):
        #
        # perturbative Symmetric-Loewdin
        X_nn= 1.5*np.eye(ndim) - 0.5*O_nn    
    else:
        #
        # diagonalization
        n_n = np.zeros(ndim, dtype=float)
        diagonalize(O_nn, n_n)
        U_nn = O_nn.T.conj().copy()
        nsqrt_n = np.diag(1.0/np.sqrt(n_n))
        X_nn = np.dot(np.dot(U_nn, nsqrt_n), U_nn.T.conj())
    #
    # apply orthonormalizing transformation
    O_nn = np.dot(X_nn, W_nn)
    
    return O_nn


def random_unitary_matrix(delta, n):

    """ Initializaes a (random) unitary matrix

    delta: float 
        strength of deviation from unit-matrix.
        > 0 random matrix
        < 0 non-random matrix
            
    n: int
        dimensionality of matrix.
    """
    
    assert n>0
    #
    # special case n=1:
    if n==1:
        return np.identity(1)
    #
    # non-random unitary matrix
    if delta<0:
        #
        W_nn = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                W_nn[i,j] = 1.0/(i+j)
        W_nn = (W_nn - W_nn.T)
        #
        return matrix_exponential(W_nn, delta)
    #
    # random unitary matrix
    elif delta>0:
        #
        W_nn = np.random.rand(n,n)
        W_nn = (W_nn - W_nn.T)
        #
        return matrix_exponential(W_nn, delta)
        #
    else:
        #
        return np.identity(n)

    
class SIC(XCFunctional):

    orbital_dependent = True
    unitary_invariant = False
    
    def __init__(self, xc='LDA', finegrid=False, **parameters):
        
        """Self-Interaction Corrected Functionals (PZ-SIC).

        finegrid: boolean
            Use fine grid for energy functional evaluations?
        """
        
        if isinstance(xc, str):
            xc = XC(xc)
        self.xc = xc
        self.type = xc.type
        XCFunctional.__init__(self, xc.name + '-PZ-SIC')
        self.finegrid = finegrid
        self.parameters = parameters

    
    def initialize(self, density, hamiltonian, wfs, occ=None):
        
        assert wfs.gamma
        
        self.wfs = wfs
        self.dtype = float
        self.xc.initialize(density, hamiltonian, wfs, occ)
        self.kpt_comm = wfs.kpt_comm
        self.nspins = wfs.nspins
        self.nbands = wfs.nbands
        
        if self.finegrid:
            poissonsolver = hamiltonian.poisson
            self.ghat = density.ghat
        else:
            poissonsolver = PoissonSolver(eps=1e-14)
            poissonsolver.set_grid_descriptor(density.gd)
            poissonsolver.initialize()
            self.ghat = LFC(density.gd,
                            [setup.ghat_l for setup in density.setups],
                            integral=sqrt(4 * pi), forces=True)

        self.spin_s = {}
        for kpt in wfs.kpt_u:
            self.spin_s[kpt.s] = SICSpin(kpt, self.xc,
                                         density, hamiltonian, wfs,
                                         poissonsolver, self.ghat,
                                         **self.parameters)
            
    def get_setup_name(self):
        return 'PBE'

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg=None, rnablaY_Lv=None,
                         tau_sg=None, dedtau_sg=None):
        return self.xc.calculate_radial(rgd, n_sLg, Y_L, v_sg,
                                        dndr_sLg, rnablaY_Lv)
    
    def set_positions(self, spos_ac):
        if not self.finegrid:
            self.ghat.set_positions(spos_ac)
    
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        self.gd = gd
        
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        # SIC:
        self.esic = 0.0
        self.ekin = 0.0
        for spin in self.spin_s.values():
            if spin.kpt.psit_nG is not None:
                desic, dekin = spin.calculate()
                self.esic += desic
                self.ekin += dekin
        self.esic = self.kpt_comm.sum(self.esic)
        self.ekin = self.kpt_comm.sum(self.ekin)
            
        return exc + self.esic

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_nG,
                                            Htpsit_nG=None, dH_asp=None):
        spin = self.spin_s[kpt.s]
        if spin.W_mn is None:
            return
        spin.apply_orbital_dependent_hamiltonian(psit_nG)

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        spin = self.spin_s[kpt.s]
        if spin.W_mn is None:
            return
        spin.correct_hamiltonian_matrix(H_nn)
    
    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        spin = self.spin_s[kpt.s]
        if spin.W_mn is None:
            return

        if calculate_change:
            spin.calculate_residual_change(psit_xG, Htpsit_xG, c_axi, n_x)
        else:
            spin.calculate_residual(psit_xG, Htpsit_xG, c_axi)
        
    def rotate(self, kpt, U_nn):
        self.spin_s[kpt.s].rotate(U_nn)

    def summary(self, out=sys.stdout):
        for s in range(self.nspins):
            if s in self.spin_s:
                spin = self.spin_s[s]
                pos_mv = spin.get_centers()
                exc_m = spin.exc_m
                ecoulomb_m = spin.ecoulomb_m
                if self.kpt_comm.rank == 1 and self.gd.comm.rank == 0:
                    nocc = self.kpt_comm.sum(spin.nocc)
                    self.kpt_comm.send(pos_mv, 0)
                    self.kpt_comm.send(exc_m, 0)
                    self.kpt_comm.send(ecoulomb_m, 0)
            else:
                if self.kpt_comm.rank == 0 and self.gd.comm.rank == 0:
                    nocc = self.kpt_comm.sum(0)
                    pos_mv = np.zeros((nocc, 3))
                    exc_m = np.zeros(nocc)
                    ecoulomb_m = np.zeros(nocc)
                    self.kpt_comm.receive(pos_mv, 1)
                    self.kpt_comm.receive(exc_m, 1)
                    self.kpt_comm.receive(ecoulomb_m, 1)
            if self.kpt_comm.rank == 0 and self.gd.comm.rank == 0:
                out.write('\nSIC orbital centers and energies:\n')
                out.write('                                %5.2fx   %5.2fx\n' %
                          (self.spin_s[0].xc_factor,
                           self.spin_s[0].coulomb_factor))
                out.write('          x       y       z       XC    Coulomb\n')
                out.write('-----------------------------------------------\n')
                m = 0
                for pos_v, exc, ecoulomb in zip(pos_mv, exc_m, ecoulomb_m):
                    out.write('%d  (%7.3f,%7.3f,%7.3f): %8.3f %8.3f\n' %
                              ((m,) + tuple(pos_v) +
                               (exc * Hartree, ecoulomb * Hartree)))
                    m += 1
                out.write('-----------------------------------------------\n')
        out.write('\nTotal SIC energy: %12.6f \n' % (self.esic* Hartree))


    def read(self, reader):
        xc_factor = reader['SIC_xc_factor']
        coulomb_factor = reader['SIC_coulomb_factor']
        #
        for s in range(self.nspins):
            #
            try:
                npart = reader.dimension('npart'+str(s))
            except KeyError:
                npart = 0
            #
            if npart>0:
                W_mn = reader.get('UnitaryTransformation'+str(s))
            else:
                W_mn = None
            #
            if s in self.spin_s.keys():
                self.spin_s[s].initial_W_mn = W_mn
                self.spin_s[s].xc_factor = xc_factor
                self.spin_s[s].coulomb_factor = coulomb_factor
            

    def write(self, writer, natoms=None):
        #
        for s in self.spin_s.keys():
            spin = self.spin_s[s]
            if self.wfs.world.rank==0:
                writer['SIC_xc_factor'] = spin.xc_factor
                writer['SIC_coulomb_factor'] = spin.coulomb_factor
            break
        #
        for s in range(self.nspins):
            #
            W_mn = self.get_unitary_transformation(s)
            #
            if self.wfs.world.rank == 0:
                if W_mn!=None:
                    writer.dimension('npart'+str(s), W_mn.shape[0])
                    writer.add('UnitaryTransformation'+str(s),
                               ('npart'+str(s),'npart'+str(s)),
                               dtype=self.dtype)
                    writer.fill(W_mn)
    

    def get_unitary_transformation(self, s):
        #
        if s in self.spin_s.keys():
            spin = self.spin_s[s]
            #
            if spin.W_mn==None or spin.finegd.rank!=0:
                n = 0
            else:
                n = spin.W_mn.shape[0]
            #
        else:
            n=0
        #
        n = self.wfs.world.sum(n)
        #
        if n>0:
            W_mn = np.zeros((n,n), dtype=self.dtype)
        else:
            W_mn = None
            return W_mn
        #
        if s in self.spin_s.keys():
            spin = self.spin_s[s]
            #
            if spin.W_mn==None or spin.finegd.rank!=0:
                W_mn[:] = 0.0
            else:
                W_mn[:] = spin.W_mn[:]
            #
        else:
            W_mn[:] = 0.0
        #
        self.wfs.world.sum(W_mn)   
        return W_mn
    

class SICSpin:
    def __init__(self, kpt, xc,
                 density, hamiltonian, wfs,
                 poissonsolver, ghat,
                 dtype=float,
                 coulomb_factor=1.0, xc_factor=1.0,
                 uominres=1E-1, uomaxres=1E-10,
                 uorelres=1E-4, uonscres=1E-8,
                 rattle=-0.1, scissor=0.0,
                 maxuoiter=-2, logging=0):
        
        """Single spin SIC object.
        

        coulomb_factor:
            Scaling factor for Hartree-functional

        xc_factor:
            Scaling factor for xc-functional

        uominres:
            Minimum residual before unitary optimization starts

        uomaxres:
            Target accuracy for unitary optimization
            (absolute variance)

        uorelres:
            Target accuracy for unitary optimization
            (rel. to basis residual)

        maxuoiter:
            Maximum number of unitary optimization steps

        rattle:
            perturbation to the initial states
        """

        self.wfs = wfs
        self.kpt = kpt
        self.xc = xc
        self.poissonsolver = poissonsolver
        self.ghat = ghat
        
        self.gd = wfs.gd
        self.interpolator = density.interpolator
        self.restrictor = hamiltonian.restrictor
        self.nspins = wfs.nspins
        self.spin = kpt.s
        self.timer = wfs.timer
        self.setups = wfs.setups
        
        self.dtype = dtype
        self.coulomb_factor = coulomb_factor    
        self.xc_factor = xc_factor    

        self.nocc = None           # number of occupied states
        self.W_mn = None           # unit. transf. to energy optimal states
        self.initial_W_mn = None   # initial unitary transformation 
        self.vt_mG = None          # orbital dependent potentials
        self.exc_m = None          # PZ-SIC contributions (from E_xc)
        self.ecoulomb_m = None     # PZ-SIC contributions (from E_H)

        self.finegd = ghat.gd

        self.rattle = rattle       # perturb the initial unitary transformation
        self.scissor = scissor     
        self.uominres = uominres
        self.uomaxres = uomaxres
        self.uorelres = uorelres   
        self.uonscres = uonscres
        self.maxuoiter = maxuoiter 
        self.maxlsiter = 1         # maximum number of line-search steps
        self.maxcgiter = 2         # maximum number of CG-iterations
        self.lsinterp = True       # interpolate for minimum during line search
        self.basiserror = 1E+20
        self.logging = logging


    def initialize_orbitals(self, rattle=-0.1, localize=True):
        #
        if self.initial_W_mn!=None:
            self.nocc = self.initial_W_mn.shape[0]
        else:
            self.nocc, x = divmod(int(self.kpt.f_n.sum()), 3 - self.nspins)
            assert x == 0

        if self.nocc==0:
            return

        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.kpt.psit_nG,
                                                    self.kpt.psit_nG, v, 1,
                                                    self.nocc)
        self.gd.comm.sum(Z_mmv)

        if self.initial_W_mn!=None:
            self.W_mn = self.initial_W_mn

        elif localize:
            W_nm = np.identity(self.nocc)
            localization = 0.0
            for iter in range(30):
                loc = _gpaw.localize(Z_mmv, W_nm)
                if loc - localization < 1e-6:
                    break
                localization = loc

            self.W_mn = W_nm.T.copy()
        else:
            self.W_mn = np.identity(self.nocc)

        if (rattle != 0.0 and self.W_mn is not None and
            self.initial_W_mn is None):
            U_mm = random_unitary_matrix(rattle, self.nocc)
            self.W_mn = np.dot(U_mm, self.W_mn)
        
        if self.W_mn!=None:
            self.gd.comm.broadcast(self.W_mn, 0)
            
        spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)
        self.initial_pos_mv = np.dot(spos_mc % 1.0, self.gd.cell_cv)


    def localize_orbitals(self):
        #
        #assert self.gd.orthogonal
        #
        # calculate wannier matrixelements 
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.kpt.psit_nG,
                                                    self.kpt.psit_nG, v, 1,
                                                    self.nocc)
        self.gd.comm.sum(Z_mmv)
        #
        # setup the initial configuration (identity)
        W_nm = np.identity(self.nocc)
        #
        # localize the orbitals
        localization = 0.0
        for iter in range(30):
            loc = _gpaw.localize(Z_mmv, W_nm)
            if loc - localization < 1e-6:
                break
            localization = loc
        #
        # apply localizing transformation
        self.W_mn = W_nm.T.copy()

            
    def rattle_orbitals(self, rattle=-0.1):
        #
        # check for the trivial cases
        if rattle==0.0:
            return

        if self.W_mn==None:
            return
        #
        # setup a "random" unitary matrix
        nocc = self.W_mn.shape[0]
        U_mm = random_unitary_matrix(rattle, nocc)
        #
        # apply unitary transformation
        self.W_mn = np.dot(U_mm, self.W_mn)


    def get_centers(self):
        assert self.gd.orthogonal

        # calculate energy optimal states (if necessary)
        if self.phit_mG is None:
            self.update_optimal_states()
        #
        # calculate wannier matrixelements
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.phit_mG,
                                                    self.phit_mG, v, 1,
                                                    self.nocc)
        self.gd.comm.sum(Z_mmv)
        #
        # calculate positions of localized orbitals
        spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)
        
        return np.dot(spos_mc % 1.0, self.gd.cell_cv) * Bohr


    def calculate_sic_matrixelements(self):
        # overlap of pseudo wavefunctions
        Htphit_mG = self.vt_mG * self.phit_mG
        V_mm = np.zeros((self.nocc, self.nocc), dtype=self.dtype)
        gemm(self.gd.dv, self.phit_mG, Htphit_mG, 0.0, V_mm, 't')

        #for a, P_mi in self.P_ami.items():
        #    for m, dH_p in enumerate(self.dH_amp[a]):
        #        dH_ii = unpack(dH_p)
        #        V_mm[m] += np.dot(P_mi, np.dot(P_mi[m], dH_ii))
        
        # PAW corrections

        # accumulate over grid-domains
        self.gd.comm.sum(V_mm)
        self.V_mm = V_mm

        # Symmetrization of V and kappa-matrix:
        K_mm = 0.5 * (V_mm - V_mm.T.conj())
        V_mm = 0.5 * (V_mm + V_mm.T.conj())

        # evaluate the kinetic correction
        self.ekin = -np.trace(V_mm) * (3 - self.nspins) 
        
        return V_mm, K_mm, np.vdot(K_mm, K_mm).real


    def update_optimal_states(self):
        #
        # pseudo wavefunctions
        self.phit_mG = self.gd.zeros(self.nocc)
        if self.nocc>0:
            gemm(1.0, self.kpt.psit_nG[:self.nocc],
                 self.W_mn, 0.0, self.phit_mG)
        #
        # PAW 
        self.P_ami = {}
        for a, P_ni in self.kpt.P_ani.items():
            self.P_ami[a] = np.dot(self.W_mn, P_ni[:self.nocc])

    def calculate_density(self, m):
        #
        # pseudo density
        nt_G = self.phit_mG[m]**2

        if self.finegd is self.gd:
            nt_g = nt_G
        else:
            nt_g = self.finegd.empty()
            self.interpolator.apply(nt_G, nt_g)
            # normalize single-particle density
            nt_g *= self.gd.integrate(nt_G) / self.finegd.integrate(nt_g)

        # PAW corrections
        Q_aL = {}
        D_ap = {}
        for a, P_mi in self.P_ami.items():
            P_i = P_mi[m]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        rhot_g = nt_g.copy()
        self.ghat.add(rhot_g, Q_aL)

        return nt_g, rhot_g, D_ap
        
    def update_potentials(self, save=False, restore=False):
        
        if restore:
            self.exc_m = self.exc_save_m 
            self.ecoulomb_m = self.eco_save_m
            self.esic = self.esic_save
            self.vt_mG = self.vt_save_mG.copy()
            self.dH_amp = self.dH_save_amp.copy()
            return

        self.timer.start('ODD-potentials')
        #
        nt_sg = self.finegd.empty(2)
        nt_sg[1] = 0.0
        vt_sg = self.finegd.empty(2)
        #
        # PAW
        W_aL = self.ghat.dict()
        zero_initial_phi = False
        
        #
        # initialize some bigger fields
        if self.vt_mG is None:
            self.vt_mG = self.gd.empty(self.nocc)
            self.exc_m = np.zeros(self.nocc)
            self.ecoulomb_m = np.zeros(self.nocc)
            self.vHt_mg = self.finegd.zeros(self.nocc)
            zero_initial_phi = True
            #
            # PAW
            self.dH_amp = {}
            for a, P_mi in self.P_ami.items():
                ni = P_mi.shape[1]
                self.dH_amp[a] = np.empty((self.nocc, ni * (ni + 1) // 2))
        #
        # loop all occupied orbitals
        for m, phit_G in enumerate(self.phit_mG):
            #
            # setup the single-particle density and PAW density-matrix
            nt_sg[0], rhot_g, D_ap = self.calculate_density(m)
            vt_sg[:] = 0.0
            #
            # xc-SIC
            self.timer.start('XC')
            if self.xc_factor != 0.0:
                exc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
                exc /= self.gd.comm.size
                vt_sg[0] *= -self.xc_factor
                #
                # PAW
                for a, D_p in D_ap.items():
                    setup = self.setups[a]
                    dH_p = self.dH_amp[a][m]
                    dH_sp = np.zeros((2, len(dH_p)))
                    #
                    # TODO: check this
                    
                    D_sp = np.array([D_p, np.zeros_like(D_p)])
                    exc += setup.xc_correction.calculate(self.xc, D_sp, dH_sp,
                                                         addcoredensity=False)
                    dH_p[:] = -dH_sp[0] * self.xc_factor
                
                self.exc_m[m] = -self.xc_factor * exc
            self.timer.stop('XC')
            #
            # Hartree-SIC
            self.timer.start('Hartree')
            if self.coulomb_factor != 0.0:
                self.poissonsolver.solve(self.vHt_mg[m], rhot_g,
                                         zero_initial_phi=zero_initial_phi)
                ecoulomb = 0.5 * self.finegd.integrate(rhot_g *
                                                       self.vHt_mg[m])
                ecoulomb /= self.gd.comm.size
                vt_sg[0] -= self.coulomb_factor * self.vHt_mg[m]
                #
                # PAW
                #
                # TODO: check if core is excluded
                self.ghat.integrate(self.vHt_mg[m], W_aL)
                for a, D_p in D_ap.items():
                    setup = self.setups[a]
                    dH_p = self.dH_amp[a][m]
                    M_p = np.dot(setup.M_pp, D_p)
                    ecoulomb += np.dot(D_p, M_p)
                    #
                    # TODO: check if core is excluded
                    dH_p -= self.coulomb_factor * (
                        2 * M_p + np.dot(setup.Delta_pL, W_aL[a]))
                #
                self.ecoulomb_m[m] = -self.coulomb_factor * ecoulomb
            self.timer.stop('Hartree')
            #
            # restrict to course grid
            if self.finegd is self.gd:
                self.vt_mG[m] = vt_sg[0]
            else:
                self.timer.start('Restrictor')
                self.restrictor.apply(vt_sg[0], self.vt_mG[m])
                self.timer.stop('Restrictor')
                
        self.timer.stop('ODD-potentials')
        #
        # accumulate total xc-SIC and coulomb-SIC
        self.gd.comm.sum(self.exc_m)
        self.gd.comm.sum(self.ecoulomb_m)
        #
        # total sic (including spin-degeneracy)
        self.esic = self.exc_m.sum()
        self.esic += self.ecoulomb_m.sum()
        self.esic *= (3 - self.nspins)

        if save:
            self.exc_save_m = self.exc_m
            self.eco_save_m = self.ecoulomb_m
            self.esic_save = self.esic
            self.vt_save_mG = self.vt_mG.copy()
            self.dH_save_amp = self.dH_amp.copy()
            

    def apply_orbital_dependent_hamiltonian(self, psit_nG):
        """...
        
        Setup ``|V phi_m>`` and ``<l|Vphi_m>`` for occupied states m
        and unoccupied states l.
        """

        nocc = self.nocc
        nvirt = psit_nG.shape[0] - nocc

        R_lm = np.zeros((nvirt, nocc), dtype=self.dtype)
        self.Htphit_mG = self.vt_mG * self.phit_mG

        if nvirt > 0:
            gemm(self.gd.dv, self.Htphit_mG, psit_nG[nocc:], 0.0, R_lm, 't')
            self.gd.comm.sum(R_lm)

        # TODO: PAW corrections to the matrixelements

        self.R_lm = R_lm
        
    def correct_hamiltonian_matrix(self, H_nn):
        
        """ Add contributions of the non-local part of the
            interaction potential to the Hamiltonian matrix.

            on entry:
                          H_nn[n,m] = <n|H_{KS}|m>
            on exit:
                          H_nn[n,m] = <n|H_{KS} + V_{u}|m>

            where V_u is the unified Hamiltonian

                V_u = ...
                
        """
        nocc = self.nocc
        nvirt = H_nn.shape[0] - nocc
        #
        W_mn = self.W_mn
        R_lm = self.R_lm
        V_mm = 0.5*(self.V_mm + self.V_mm.T)
        #
        H_nn[:nocc, :nocc] += np.dot(W_mn.T, np.dot(V_mm, W_mn))
        #
        if nvirt != 0:
            R_ln = np.dot(R_lm, W_mn) 
            H_nn[:nocc, nocc:] += R_ln.T
            H_nn[nocc:, :nocc] += R_ln

        if self.scissor!=0.0:
            H_nn[self.nocc:, self.nocc:] += np.eye(nbands -
                                                   self.nocc)*self.scissor

        
    def calculate_residual(self, psit_nG, Htpsit_nG, c_ani):
        
        """ Calculate the action of the unified Hamiltonian on an
            arbitrary state:

                H_u|Psi> = 
        """
        #
        nocc = self.nocc
        nvirt = psit_nG.shape[0] - nocc
        R_lm  = self.R_lm
        W_mn  = self.W_mn
        Htphit_mG = self.Htphit_mG
        phit_mG = self.phit_mG
        K_mm = 0.5*(self.V_mm - self.V_mm.T)
        V_mn = np.dot(K_mm, W_mn)
        #
        # Action of unified Hamiltonian on occupied states:
        if nocc>0:
            gemm(1.0, Htphit_mG, W_mn.T.copy(), 1.0, Htpsit_nG[:nocc])
            gemm(1.0,   phit_mG, V_mn.T.copy(), 1.0, Htpsit_nG[:nocc])
        if nvirt>0:
            gemm(1.0,   phit_mG, R_lm, 1.0, Htpsit_nG[nocc:])
        #
        # TODO: PAW corrections
        
        #Htpsit_nG[:nocc] +=  np.dot(Htphit_mG.T, W_mn).T
        #Htpsit_nG[:nocc] +=  np.dot(phit_mG.T, np.dot(K_mm, W_mn)).T
        #Htpsit_nG[nocc:] +=  np.dot(phit_mG.T, R_lm.T).T
        #
        
    def calculate_residual_change(self, psit_xG, Htpsit_xG, c_axi, n_x):

        """ 

        """
        #
        assert len(n_x) == 1
        #
        Htphit_mG = self.Htphit_mG
        phit_mG = self.phit_mG
        
        c_mx = np.zeros((self.nocc, 1), dtype=self.dtype)
        v_mx = np.zeros((self.nocc, 1), dtype=self.dtype)
        gemm(self.gd.dv, psit_xG, phit_mG  , 0.0, c_mx, 't')
        gemm(self.gd.dv, psit_xG, Htphit_mG, 0.0, v_mx, 't')
        self.gd.comm.sum(c_mx)
        self.gd.comm.sum(v_mx)
        #
        # TODO: PAW-corrections
        
        V_mm = 0.5*(self.V_mm + self.V_mm.T)
        v_mx -= np.dot(V_mm, c_mx)
        
        gemm(1.0, Htphit_mG, c_mx.T.copy(), 1.0, Htpsit_xG)
        gemm(1.0,   phit_mG, v_mx.T.copy(), 1.0, Htpsit_xG)
        #
        # TODO: PAW-corrections
        
        #Htpsit_xG += (np.dot(Htphit_mG.T, c_mx).T
        #             +np.dot(self.phit_mG.T, v_mx).T)


    def rotate(self, U_nn):
        
        """ Unitary transformations amongst the canonic states
            require to apply a counter-acting transformation to
            the energy optimal states. This subroutine takes
            care of it.

            Reorthogonalization is required whenever unoccupied
            states are mixed in.
        """
        #
        # skip if no transformation to optimal states is set-up
        if self.W_mn is None:
            return
        #
        # compensate the transformation amongst the occupied states
        self.W_mn = np.dot(self.W_mn, U_nn[:self.nocc, :self.nocc].T)
        #
        # reorthogonalize if unoccupied states may have been mixed in
        if self.nocc != U_nn.shape[0]:
            self.W_mn = ortho(self.W_mn)

                
    def calculate(self):

        """ Evaluate the non-unitary invariant part of the
            energy functional and returns

            esic: float
                self-interaction energy
            
            ekin: float
                correction to the kinetic energy
        """
        #
        # initialize transformation from canonic to energy
        # optimal states (if necessary)
        if self.W_mn is None:
            self.initialize_orbitals(rattle=self.rattle)
        #
        # optimize the non-unitary invariant part of the
        # functional
        self.unitary_optimization()
        
        return self.esic, self.ekin


    def unitary_optimization(self):

        """ Optimization of the non-unitary invariant part of the
            energy functional.
        """
        
        ESI_init = 0.0
        ESI      = 0.0
        dE       = 1e-16  

        optstep  = 0.0
        Gold     = 0.0
        cgiter   = 0
        #
        epsstep  = 0.005  # 0.005
        dltstep  = 0.1    # 0.1
        prec     = 1E-7
        oldstep  = 0.0
        #
        #
        # get the initial ODD potentials/energy/matrixelements
        self.update_optimal_states()
        self.update_potentials(save=True)
        ESI = self.esic
        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
        ESI_init = ESI

        if norm < self.uonscres and self.maxuoiter>0:
            return
        
        if self.nocc <= 1:
            return
        #
        # optimize the unitary transformation
        # --------------------------------------------------------------
        #
        # allocate arrays for the search direction,
        # i.e., the (conjugate) gradient
        D_old_mm = np.zeros_like(self.W_mn)
        #
        for iter in range(abs(self.maxuoiter)):
            #
            # copy the initial unitary transformation and orbital
            # dependent energies
            W_old_mn    = self.W_mn.copy()
            ESI_old  = ESI
            #
            # setup the steepest-descent/conjugate gradient
            # D_nn:  search direction
            # K_nn:  inverse gradient
            # G0  :  <K,D> (projected length of D along K)
            if (Gold!=0.0):
                #
                # conjugate gradient
                G0        = np.sum(K_mm*K_mm.conj()).real
                beta      = G0/Gold
                Gold      = G0
                D_mm      = K_mm + beta*D_old_mm
                #
                G0        = np.sum(K_mm*D_mm.conj()).real
            else:
                #
                # steepest-descent
                G0        = np.sum(K_mm*K_mm.conj()).real
                Gold      = G0
                D_mm      = K_mm
            #
            updated  = False
            minimum  = False
            failed   = True
            noise    = False
            E0       = ESI
            #
            # try to estimate optimal step-length from change in length
            # of the gradient (force-only)
            # ----------------------------------------------------------
            if (epsstep!=0.0 and norm > self.uomaxres):
                step = epsstep
                while (True):
                    U_mm = matrix_exponential(D_mm, step)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E1 = self.esic
                    K0_mm = K_mm.copy()
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                    #
                    # projected length of the gradient at the new position
                    G1 = np.sum(((K_mm-K0_mm)/step)*D_mm.conj()).real
                    #
                    if (abs(E1-E0)<prec and E1>=E0):
                        #
                        eps_works = True
                        Eeps      = E1
                        noise     = True
                    elif (E1<E0):
                        #
                        # trial step reduced energy
                        eps_works = True
                        Eeps      = E1
                    else:
                        #
                        # epsilon step did not work
                        eps_works = False
                        optstep   = 0.0
                        break
                    #
                    # compute the optimal step size
                    #optstep = step*G0/(G0-G1)
                    #print -G0/G1
                    optstep = -G0/G1
                    #
                    if (eps_works):
                        break
                    #
                #print eps_works, optstep, G0/((G0-G1)/step)
                #
                # decide on the method for stepping
                oldstep = 0.0
                if (optstep > 0.0):
                    #
                    # convex region -> force only estimate for minimum
                    U_mm = matrix_exponential(D_mm, optstep)
                    self.W_mn = np.dot(U_mm,W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E1 = self.esic
                    if (abs(E1-E0)<prec and E1>=E0):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI       = E1
                        optstep   = optstep
                        lsiter    = 0
                        maxlsiter = -1
                        updated   = True
                        minimum   = True
                        failed    = False
                        lsmethod  = 'CV-N'
                        noise     = True
                    elif (E1<E0):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI       = E1
                        optstep   = optstep
                        lsiter    = 0
                        maxlsiter = -1
                        updated   = True
                        minimum   = True
                        failed    = False
                        lsmethod  = 'CV-S'
                        oldstep   = optstep
                    else:
                        #self.K_unn[q] = K_nn
                        ESI       = E0
                        step      = optstep
                        optstep   = 0.0
                        lsiter    = 0
                        maxlsiter = self.maxlsiter
                        updated   = False
                        minimum   = False
                        failed    = True
                        lsmethod  = 'CV-F-CC'
                else:
                    #self.K_unn[q] = K_nn
                    ESI       = E0
                    step      = optstep
                    optstep   = 0.0
                    lsiter    = 0
                    maxlsiter = self.maxlsiter
                    updated   = False
                    minimum   = False
                    failed    = True
                    lsmethod  = 'CC'
                #
            else:
                maxlsiter = 0
                lsiter = -1
                optstep = epsstep
                updated = False
                minimum = True
                failed = False
                lsmethod = 'CC-EPS'
                
            if (optstep==0.0):
                #
                # we are in the concave region or force-only estimate failed,
                # just follow the (conjugate) gradient
                step = dltstep * abs(step)
                U_mm = matrix_exponential(D_mm, step)
                self.W_mn = np.dot(U_mm,W_old_mn)
                self.update_optimal_states()
                self.update_potentials()
                E1 = self.esic
                #
                #
                if (abs(E1-E0)<prec and E1>=E0):
                    ESI       = E1
                    optstep   = 0.0
                    updated   = False
                    minimum   = True
                    failed    = True
                    lsmethod  = lsmethod+'-DLT-N'
                    noise     = True
                    maxlsiter = -1
                elif (E1<E0):
                    ESI       = E1
                    optstep   = step
                    updated   = True
                    minimum   = False
                    failed    = False
                    lsmethod  = lsmethod+'-DLT'
                    maxlsiter = self.maxlsiter
                elif (eps_works):
                    ESI       = Eeps
                    E1        = Eeps
                    step      = epsstep
                    updated   = False
                    minimum   = False
                    failed    = False
                    lsmethod  = lsmethod+'-EPS'
                    maxlsiter = self.maxlsiter
                else:
                    optstep   = 0.0
                    updated   = False
                    minimum   = False
                    failed    = True
                    lsmethod  = lsmethod+'-EPS-F'
                    maxlsiter = -1
                #
                G       = (E1-E0)/step
                step0   = 0.0
                step1   = step
                step2   = 2*step
                #
                for lsiter in range(maxlsiter):
                    #
                    # energy at the new position
                    U_mm = matrix_exponential(D_mm, step2)
                    self.W_mn = np.dot(U_mm,W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E2 = self.esic
                    G  = (E2-E1)/(step2-step1)
                    #
                    #
                    if (G>0.0):
                        if self.lsinterp:
                            a= E0/((step2-step0)*(step1-step0)) \
                             + E2/((step2-step1)*(step2-step0)) \
                             - E1/((step2-step1)*(step1-step0))
                            b=(E2-E0)/(step2-step0)-a*(step2+step0)
                            if (a<step**2):
                                optstep = 0.5*(step0+step2)
                            else:
                                optstep =-0.5*b/a
                            updated  = False
                            minimum  = True
                            break
                        else:
                            optstep  = step1
                            updated  = False
                            minimum  = True
                            break
                    #
                    elif (G<0.0):
                        optstep = step2
                        step0   = step1
                        step1   = step2
                        step2   = step2 + step
                        E0      = E1
                        E1      = E2
                        ESI     = E2
                        updated = True
                        minimum = False
            #
            if (cgiter!=0):
                lsmethod = lsmethod + ' CG'
            #
            if (cgiter>=self.maxcgiter or not minimum):
                Gold        = 0.0
                cgiter      = 0
            else:
                cgiter      = cgiter + 1
                D_old_mm[:,:]  = D_mm[:,:]
            #
            # update the energy and matrixelements of V and Kappa
            # and accumulate total residual of unitary optimization
            if (not updated):
                if optstep>0.0:
                    U_mm = matrix_exponential(D_mm, optstep)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    ESI = self.esic
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                else:
                    self.W_mn = W_old_mn
                    self.update_optimal_states()
                    self.update_potentials(restore=True)
                    ESI = self.esic
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                

            if (lsiter==maxlsiter-1):
                V_mm, K_mm, norm = self.calculate_sic_matrixelements()
            #
            E0=ESI
            #
            # orthonormalize the energy optimal orbitals
            self.W_mn = ortho(self.W_mn)
            K = max(norm, 1.0e-16)
           
            if self.gd.comm.rank == 0:
                if self.logging==1:
                    print "           UO-%d: %2d %5.1f  %20.5f  " % (
                        self.spin, iter, np.log10(K), ESI*Hartree)
                elif self.logging==2:
                    print "           UO-%d: %2d %5.1f  %20.5f  " % (
                    self.spin, iter, np.log10(K), ESI*Hartree) + lsmethod
                
            if ((K<self.uomaxres or
                 K<self.wfs.eigensolver.error*self.uorelres)
                 or noise or failed) and not self.maxuoiter<0:
                break


        
        
    
