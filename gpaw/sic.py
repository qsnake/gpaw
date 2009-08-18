"""SIC stuff - work in progress!"""

import numpy as np
import ase.units as units

from gpaw.utilities import pack
from gpaw.xc_functional import XCFunctional

from gpaw.atom.generator import Generator, parameters
from gpaw.utilities import hartree
from math import pi

class SIC:
    def __init__(self, nspins=1, xcname='LDA',
                 coufac=1.0, excfac=1.0):
        """Self-Interaction Corrected (SIC) Functionals.

        nspins: int
            Number of spins.

        xcname: string
            Name of LDA/GGA functional which acts as
            a starting point for the construction of
            the SIC functional

        """

        self.nspins = nspins
        self.xcbasisname = xcname
        self.xcname      = xcname + '-SIC'
        if nspins==2:
            self.xcbasis     = XCFunctional(self.xcbasisname, 2)
            self.xcsic       = self.xcbasis
        else:
            self.xcbasis     = XCFunctional(self.xcbasisname, nspins)
            self.xcsic       = XCFunctional(self.xcbasisname, 2)
        self.gga = self.xcbasis.gga
        self.mgga = not True
        self.orbital_dependent = True
        self.hybrid = 0.0
        self.uses_libxc = self.xcbasis.uses_libxc
        self.gllb = False
        self.xcbasisname = xcname
        self.xcname = xcname + '-SIC'
        
        self.coufac = coufac
        self.excfac = excfac
        
    def set_non_local_things(self, density, hamiltonian, wfs, atoms,
                             energy_only=False):
        self.gd = density.gd
        self.finegd = density.finegd
        self.v_unG     = self.gd.empty((len(wfs.kpt_u), wfs.nbands))
        self.v_cou_unG = self.gd.empty((len(wfs.kpt_u), wfs.nbands))
        self.wfs = wfs
        self.density = density
        self.hamiltonian = hamiltonian

    def is_gllb(self):
        return False

    def get_name(self):
        return self.xcname

    def get_setup_name(self):
        return self.xcbasisname

    def apply_non_local(self, kpt):
        pass

    def get_non_local_kinetic_corrections(self):
        return 0.0

    def adjust_non_local_residual(self, pR_G, dR_G, kpt, n):
        pass

    def get_non_local_force(self, kpt):
        return 0.0
    
    def get_non_local_energy(self, n_g=None, a2_g=None, e_LDAc_g=None,
                             v_LDAc_g=None, v_g=None, deda2_g=None):
        return 0.0
    
    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None):
        
        # the LDA/GGA part of the functional
        self.xcbasis.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
        
        # orbital dependend components of the functional
        if n_g.ndim == 3:
            ESI = self.calculate_sic_potentials()
            if self.density.finegd.comm.rank == 0:
                assert e_g.ndim == 3
                e_g[0, 0, 0] += ESI / self.density.finegd.dv
            print 'SIC=',ESI

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                                a2_g=None, aa2_g=None, ab2_g=None,
                                deda2_g=None, dedaa2_g=None, dedab2_g=None):
        self.xcbasis.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                        a2_g, aa2_g, ab2_g,
                                        deda2_g, dedaa2_g, dedab2_g)
        if na_g.ndim == 3:
            ESI = self.calculate_sic_potentials()
            if self.density.finegd.comm.rank == 0:
                assert e_g.ndim == 3
                e_g[:] = 0.0
                e_g[0, 0, 0] += ESI / self.density.finegd.dv
            print 'SIC=',ESI


    def calculate_sic_potentials(self):
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
        
        ESI = 0.0
        for u, kpt in enumerate(self.wfs.kpt_u):
            for n in range(self.wfs.nbands):
                ESI += self.calculate_sic_potential(kpt, n,
                                                    self.v_unG[u, n],
                                                    self.v_cou_unG[u, n])
        return ESI

                
    def calculate_sic_potential(self, kpt, n, v_G, v_cou_G):
        
        # define some shortcuts to objects
        wfs         = self.wfs
        setups      = self.wfs.setups
        density     = self.density
        hamiltonian = self.hamiltonian
        
        # SIC energy contributions,
        # sic potential on the fine grid and
        # a dummy density rho=0 (needed as
        # SIC contributions are always fully
        # polarized
        Ecou_SI     = 0.0
        Exc_SI      = 0.0
        e_g         = density.finegd.zeros()
        v_g         = density.finegd.zeros()
        v_cou_g     = density.finegd.zeros()
        v_g         = density.finegd.zeros()
        
        nt_g0       = density.finegd.zeros()
        v_g0        = density.finegd.zeros()
        
        # get pseudo density of a state n from the k-point kpt
        psit_G = kpt.psit_nG[n]
        nt_G   = psit_G**2
        
        # interpolate density on the fine grid
        # (note: the total norm has to be
        # conserved explicitly by renormalization
        # to avoid errors due to the interpolation)
        Nt   = density.gd.integrate(nt_G)
        nt_g = density.finegd.empty()
        density.interpolator.apply(nt_G, nt_g)
        Ntfine = density.finegd.integrate(nt_g)
        nt_g *= Nt / Ntfine
        
        # compute the PAW corrections to the
        # orbital density and self-coulomb energy
        D_aii = {}
        Q_aL = {}
        for a, P_ni in kpt.P_ani.items():
            P_i = P_ni[n]
            D_aii[a] = np.outer(P_i, P_i)
            D_p = pack(D_aii[a])
            
            # corrections to the orbital density
            Q_aL[a] = np.dot(D_p, setups[a].Delta_pL)
            
            # corrections to the self-coulomb integral
            #Ecou_SI += self.coufac*np.dot(D_p, np.dot(setups[a].M_pp, D_p))
        
        # add corrections to the orbital density
        # to obtain the single-particle density of the
        # valence orbital
        density.ghat.add(nt_g, Q_aL)
        
        # compute the contribution from E_xc[rho,0] to the
        # self-interaction potential
        if self.excfac==0.0:
            v_g[:] = 0.0
            Exc_SI = 0.0
        else:
            v_g[:] = 0.0
            self.xcsic.calculate_spinpolarized(e_g, nt_g, v_g, nt_g0, v_g0)
            Exc_SI = -self.excfac*e_g.ravel().sum() * density.finegd.dv
            v_g[:] *= self.excfac

        for a in density.D_asp:
            xccorr = density.setups[a].xc_correction
            
            # Obtain the atomic density matrix for state
            D_sp = wfs.get_orbital_density_matrix(a, kpt, n)
            
            # PAW correction to pseudo Hartree-energy
            Ecou_SI += self.coufac*np.sum([np.dot(D_p, np.dot(setups[a].M_pp, D_p))
                                           for D_p in D_sp])
            
            # Expand the density matrix to spin-polarized case
            if len(D_sp) == 1:
                D_p2 = D_sp[0].copy()
                D_p2[:] = 0.0
                D_sp = [ D_sp[0], D_p2 ]

                
        # compute the coulomb-self-interaction potential 
        if self.coufac==0.0:
            v_cou_g[:]= 0.0
            Ecou_SI   = 0.0
        else:
            psolver = self.hamiltonian.poisson
            #
            # interpolare on fine grid
            density.interpolator.apply(v_cou_G, v_cou_g)
            #
            # solve poisson equation
            psolver.solve(v_cou_g, nt_g, charge=1, zero_initial_phi=False)
            #
            # add coulomb energy of compensated pseudo densities to integral
            Ecou_SI -= 0.5 * self.coufac*density.finegd.integrate(nt_g * v_cou_g)
            #
            # add to the total SIC potential
            v_g[:] -= self.coufac*v_cou_g[:]
        #
        # restrict to the coarse grid
        hamiltonian.restrictor.apply(v_g, v_G)
        hamiltonian.restrictor.apply(v_cou_g,v_cou_G)
        
        # write some information about the contributions
        print ("%3d : %10.5f  %10.5f %10.5f %10.5f %10.5f" %
               (n,Ecou_SI,Exc_SI,Ecou_SI+Exc_SI, kpt.eps_n[n], kpt.f_n[n]))
        return Exc_SI+Ecou_SI

    def add_non_local_terms(self, psit_nG, Htpsit_nG, s):
        assert self.wfs.kpt_comm.size == 1
        for psit_G, Htpsit_G, v_G in zip(psit_nG, Htpsit_nG, self.v_unG[s]):
            Htpsit_G += psit_G * v_G

