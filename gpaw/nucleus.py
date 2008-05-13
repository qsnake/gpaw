# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Nucleus class.

A Paw object has a list of nuclei. Each nucleus is described by a
``Setup`` object and a scaled position plus some extra stuff..."""

from math import pi, sqrt
from cmath import exp

import numpy as npy

from gpaw.utilities.complex import real, cc
from gpaw.localized_functions import create_localized_functions
from gpaw.utilities import unpack, pack, pack2, unpack2
import gpaw.mpi as mpi


class Nucleus:
    """Nucleus-class.

    The ``Nucleus`` object basically consists of a ``Setup`` object, a
    scaled position and some localized functions.  It takes care of
    adding localized functions to functions on extended grids and
    calculating integrals of functions on extended grids and localized
    functions.

     ============= ========================================================
     ``setup``     ``Setup`` object.
     ``spos_c``    Scaled position.
     ``a``         Index number for this nucleus.
     ``dtype``     Data type of wave functions (``Float`` or ``Complex``).
     ``neighbors`` List of overlapping neighbor nuclei.
     ============= ========================================================

    Localized functions:
     ========== ===========================================================
     ``nct``    Pseudo core electron density.
     ``tauct``  Pseudo kinetic energy density.
     ``ghat_L`` Shape functions for compensation charges.
     ``vhat_L`` Correction potentials for overlapping compensation charges.
     ``pt_i``   Projector functions.
     ``vbar``   Arbitrary localized potential.
     ``phit_i`` Pseudo partial waves used for initial wave function guess.
     ========== ===========================================================

    Arrays:
     ========= ===============================================================
     ``P_uni`` Integral of products of all wave functions and the projector
               functions of this atom (``P_{\sigma\vec{k}ni}^a``).
     ``D_sp``  Atomic density matrix (``D_{\sigma i_1i_2}^a``).
               Packed with pack 1.
     ``dH_sp`` Atomic Hamiltonian correction (``\Delta H_{\sigma i_1i_2}^a``).
               Packed with pack 2.
     ``Q_L``   Multipole moments  (``Q_{\ell m}^a``).
     ``F_c``   Force.
     ========= ===============================================================

    Parallel stuff: ``comm``, ``rank`` and ``in_this_domain``
    """
    def __init__(self, setup, a, dtype):
        """Construct a ``Nucleus`` object."""
        self.setup = setup
        self.a = a
        self.dtype = dtype
        lmax = setup.lmax
        self.Q_L = npy.empty((lmax + 1)**2)
        self.neighbors = []
        self.spos_c = npy.array([-1.0, -1.0, -1.0])

        self.rank = -1
        self.comm = mpi.serial_comm
        self.in_this_domain = False
        self.ready = False
        
        self.pt_i = None
        self.vbar = None
        self.ghat_L = None
        self.vhat_L = None
        self.nct = None
        self.tauct = None
        self.mom = npy.array(0.0)

    def __cmp__(self, other):
        """Ordering of nuclei.

        Use sequence number ``a`` to sort lists."""
        
        return cmp(self.a, other.a)
    
    def allocate(self, nspins, nmyu, nbands):
        ni = self.get_number_of_partial_waves()
        np = ni * (ni + 1) // 2
        self.D_sp = npy.empty((nspins, np))
        self.H_sp = npy.empty((nspins, np))
        self.P_uni = npy.empty((nmyu, nbands, ni), self.dtype)
        self.F_c = npy.zeros(3)
        if self.setup.xc_correction.xc.xcfunc.hybrid > 0.0:
            self.vxx_uni = npy.empty((nmyu, nbands, ni), self.dtype)
            self.vxx_unii = npy.zeros((nmyu, nbands, ni, ni), self.dtype)

    def reallocate(self, nbands):
        nu, nao, ni = self.P_uni.shape
        if nbands < nao:
            self.P_uni = self.P_uni[:, :nbands, :].copy()
        else:
            P_uni = npy.empty((nu, nbands, ni), self.dtype)
            P_uni[:, :nao, :] = self.P_uni
            P_uni[:, nao:, :] = 0.0
            self.P_uni = P_uni

    def set_position(self, spos_c, domain, my_nuclei, nspins, nmyu, nbands):
        """Move nucleus.

        """
        self.spos_c = spos_c

        self.comm = domain.comm # ??? XXX

        rank = domain.get_rank_for_position(spos_c)
        in_this_domain = (rank == self.comm.rank)

        if in_this_domain and not self.in_this_domain:
            # Nuclei new on this cpu:
            my_nuclei.append(self)
            my_nuclei.sort()
            self.allocate(nspins, nmyu, nbands)
            if self.rank != -1:
                self.comm.receive(self.D_sp, self.rank, 555)
        elif not in_this_domain and self.in_this_domain:
            # Nuclei moved to other cpu:
            my_nuclei.remove(self)
            if self.rank != -1:
                self.comm.send(self.D_sp, rank, 555)
            del self.D_sp, self.H_sp, self.P_uni, self.F_c
            
        self.in_this_domain = in_this_domain
        self.rank = rank

    def move(self, spos_c, gd, finegd, k_ki, lfbc, pt_nuclei, ghat_nuclei):
        """Move nucleus.

        """
        rank = self.rank
        in_this_domain = self.in_this_domain

        # Shortcut:
        create = create_localized_functions

        # Projectors:
        pt_j = self.setup.pt_j
        pt_i = create(pt_j, gd, spos_c, dtype=self.dtype, lfbc=lfbc)

        if self.dtype == complex and pt_i is not None:
            pt_i.set_phase_factors(k_ki)
        
        # Update pt_nuclei:
        if pt_i is not None and self.pt_i is None:
            pt_nuclei.append(self)
            pt_nuclei.sort()
        if pt_i is None and self.pt_i is not None:
            pt_nuclei.remove(self)

        self.pt_i = pt_i

        # Localized potential:
        vbar = self.setup.vbar
        vbar = create([vbar], finegd, spos_c, lfbc=lfbc)

        self.vbar = vbar

        # Shape functions:
        ghat_l = self.setup.ghat_l
        ghat_L = create(ghat_l, finegd, spos_c, lfbc=lfbc)

        # Step function:
        stepf = self.setup.stepf
        stepf = create([stepf], finegd, spos_c, lfbc=lfbc, forces=False)
        self.stepf = stepf
            
        # Potential:
        vhat_l = self.setup.vhat_l
        if vhat_l is None:
            vhat_L = None
        else:
            vhat_L = create(vhat_l, finegd, spos_c, lfbc=lfbc)
            # ghat and vhat have the same size:
            assert (ghat_L is None) == (vhat_L is None)

        # Update ghat_nuclei:
        if ghat_L is not None and self.ghat_L is None:
            ghat_nuclei.append(self)
            ghat_nuclei.sort()
        if ghat_L is None and self.ghat_L is not None:
            ghat_nuclei.remove(self)

        self.ghat_L = ghat_L
        self.vhat_L = vhat_L

        # Smooth core density:
        nct = self.setup.nct
        self.nct = create([nct], gd, spos_c, cut=True, lfbc=lfbc)

        # Smooth core kinetic energy density:
        tauct = self.setup.tauct
        self.tauct = create([tauct], gd, spos_c, cut=True, lfbc=lfbc)
            
        self.ready = True

        # Moving the atoms in a course grid EXX calculation doesn't
        # work.  Make sure it fails:
        self.Ghat_L = None

    def normalize_shape_function_and_pseudo_core_density(self):
        """Normalize shape function and pseudo core density.

        When these functions are put on a grid, their integrals may
        not be exactly what they should be. We fix that here."""

        if self.ghat_L is not None:
            self.ghat_L.normalize(sqrt(4 * pi))

        # Any core electrons?
        if self.setup.Nc == 0:
            return  # No!

        # Yes.  Normalize smooth core density:
        if self.nct is not None:
            Nct = -(self.setup.Delta0 * sqrt(4 * pi)
                    + self.setup.Z - self.setup.Nc)
            self.nct.normalize(Nct)

    def initialize_atomic_orbitals(self, gd, k_ki, lfbc):
        phit_j = self.setup.phit_j
        self.phit_i = create_localized_functions(
            phit_j, gd, self.spos_c, dtype=self.dtype,
            cut=True, forces=False, lfbc=lfbc)
        if self.dtype == complex and self.phit_i is not None:
            self.phit_i.set_phase_factors(k_ki)

    def get_number_of_atomic_orbitals(self):
        return self.setup.niAO

    def get_number_of_partial_waves(self):
        return self.setup.ni
    
    def create_atomic_orbitals(self, psit_iG, k):
        if self.phit_i is None:
            # Nothing to do in this domain:
            return

        coefs_ii = npy.identity(len(psit_iG), psit_iG.dtype.char)
        self.phit_i.add(psit_iG, coefs_ii, k)

    def add_atomic_density(self, nt_sG, magmom, hund):
        if self.phit_i is None:
            # Nothing to do in this domain:
            return

        ns = len(nt_sG)
        ni = self.get_number_of_partial_waves()
        niao = self.get_number_of_atomic_orbitals()
        
        if hasattr(self, 'f_si'):
            # Convert to ndarray:
            self.f_si = npy.asarray(self.f_si, float)
        else:
            self.f_si = self.calculate_initial_occupation_numbers(ns, niao,
                                                                  magmom, hund)

        if self.in_this_domain:
            D_sii = npy.zeros((ns, ni, ni))
            for i in range(min(ni, niao)):
                D_sii[:, i, i] = self.f_si[:, i]
            for s in range(ns):
                self.D_sp[s] = pack(D_sii[s])

        for s in range(ns):
            self.phit_i.add_density(nt_sG[s], self.f_si[s])

    def calculate_initial_occupation_numbers(self, ns, niao, magmom, hund):
        f_si = npy.zeros((ns, niao))
        i = 0
        nj = len(self.setup.n_j)
        for j, phit in enumerate(self.setup.phit_j):
            l = phit.get_angular_momentum_number()
            if j < nj and self.setup.n_j[j] >= 0:
                f = self.setup.f_j[j]
            else:
                f = 0
            degeneracy = 2 * l + 1
            f = int(f)
            if hund:
                # Use Hunds rules:
                f_si[0, i:i + min(f, degeneracy)] = 1.0      # spin up
                f_si[1, i:i + max(f - degeneracy, 0)] = 1.0  # spin down
                if f < degeneracy:
                    magmom -= f
                else:
                    magmom -= 2 * degeneracy - f
            else:
                if ns == 1:
                    f_si[0, i:i + degeneracy] = 1.0 * f / degeneracy
                else:
                    maxmom = min(f, 2 * degeneracy - f)
                    mag = magmom
                    if abs(mag) > maxmom:
                        mag = cmp(mag, 0) * maxmom
                    f_si[0, i:i + degeneracy] = 0.5 * (f + mag) / degeneracy
                    f_si[1, i:i + degeneracy] = 0.5 * (f - mag) / degeneracy
                    magmom -= mag
                
            i += degeneracy

        if magmom != 0:
            raise RuntimeError('Bad magnetic moment %g for %s atom!' %
                               (magmom, self.setup.symbol))
        assert i == niao
        return f_si
    
    def add_smooth_core_density(self, nct_G, nspins):
        if self.nct is not None:
            self.nct.add(nct_G, npy.array([1.0 / nspins]))

    def add_smooth_core_kinetic_energy_density(self, tauct_G, nspins):
        if self.tauct is not None:
            self.tauct.add(tauct_G, npy.array([1.0 / nspins]))

    def add_compensation_charge(self, nt2):
        self.ghat_L.add(nt2, self.Q_L)
        
    def add_hat_potential(self, vt2):
        if self.vhat_L is not None:
            self.vhat_L.add(vt2, self.Q_L)

    def add_localized_potential(self, vt2):
        if self.vbar is not None:
            self.vbar.add(vt2, npy.array([1.0]))
        
    def calculate_projections(self, kpt):
        """Iterator for calculation of wave-function projections.

        This iterator must be called twice, and after that the result
        will be in self.P_uni on the node that owns the atom::

                             ~    _  ~a _ _a  
          P_uni[u, n, i] = <psi  (r)|p (r-R )>
                               un     i

        """
        
        if self.in_this_domain:
            P_ni = self.P_uni[kpt.u]
            P_ni[:] = 0.0
        else:
            P_ni = None

        for x in self.pt_i.iintegrate(kpt.psit_nG, P_ni, kpt.k):
            yield None

    def calculate_multipole_moments(self):
        if self.in_this_domain:
            self.Q_L[:] = npy.dot(self.D_sp.sum(0), self.setup.Delta_pL)
            self.Q_L[0] += self.setup.Delta0
        self.comm.broadcast(self.Q_L, self.rank)

    def calculate_magnetic_moments(self):
        if self.in_this_domain:
            dif = self.D_sp[0,:] - self.D_sp[1,:]
            self.mom = npy.array(sqrt(4 * pi) *
                                 npy.dot(dif, self.setup.Delta_pL[:,0]))
        self.comm.broadcast(self.mom, self.rank)
        
    def calculate_hamiltonian(self, nt_g, vHt_g, vext=None):
        if self.in_this_domain:
            s = self.setup
            W_L = npy.zeros((s.lmax + 1)**2)
            for neighbor in self.neighbors:
                W_L += npy.dot(neighbor.v_LL, neighbor.nucleus().Q_L)
            U = 0.5 * npy.dot(self.Q_L, W_L)

            if self.vhat_L is not None:
                for x in self.vhat_L.iintegrate(nt_g, W_L):
                    yield None
            for x in self.ghat_L.iintegrate(vHt_g, W_L):
                yield None

            D_p = self.D_sp.sum(0)
            dH_p = (s.K_p + s.M_p + s.MB_p + 2.0 * npy.dot(s.M_pp, D_p) +
                    npy.dot(s.Delta_pL, W_L))

            Exc = s.xc_correction.calculate_energy_and_derivatives(
                self.D_sp, self.H_sp, self.a)

            Ekin = npy.dot(s.K_p, D_p) + s.Kc

            Ebar = s.MB + npy.dot(s.MB_p, D_p)
            Epot = U + s.M + npy.dot(D_p, (s.M_p + npy.dot(s.M_pp, D_p)))

            if s.HubU is not None:
##                 print '-----'
                nspins = len(self.D_sp)
                i0 = s.Hubi
                i1 = i0 + 2 * s.Hubl + 1
                for D_p, H_p in zip(self.D_sp, self.H_sp):
                    N_mm = unpack2(D_p)[i0:i1, i0:i1] / 2 * nspins 
                    Eorb = s.HubU/2. * (N_mm - npy.dot(N_mm,N_mm)).trace()
                    Vorb = s.HubU * (0.5 * npy.eye(i1-i0) - N_mm)
##                     print '========='
##                     print 'occs:',npy.diag(N_mm)
##                     print 'Eorb:',Eorb
##                     print 'Vorb:',npy.diag(Vorb)
##                     print '========='
                    Exc += Eorb                    
                    Htemp = unpack(H_p)
                    Htemp[i0:i1,i0:i1] += Vorb
                    H_p[:] = pack2(Htemp)

            # Note that the external potential is assumed to be
            # constant inside the augmentation spheres.
            Eext = 0.0
            if vext is not None:
                Eext += vext * sqrt(4 * pi) * (self.Q_L[0] + s.Z)
                dH_p += vext * sqrt(4 * pi) * s.Delta_pL[:, 0]
            
            for H_p in self.H_sp:
                H_p += dH_p

            # Move this kinetic energy contribution to Paw.py: ????!!!!
            Ekin -= npy.dot(self.D_sp[0], self.H_sp[0])
            if len(self.D_sp) == 2:
                Ekin -= npy.dot(self.D_sp[1], self.H_sp[1])

            yield Ekin, Epot, Ebar, Eext, Exc
        
        else:
            if self.vhat_L is not None:
                for x in self.vhat_L.iintegrate(nt_g, None):
                    yield None
            for x in self.ghat_L.iintegrate(vHt_g, None):
                yield None

            yield 0.0, 0.0, 0.0, 0.0, 0.0

    def update_core_eigenvalues(self, vHt_g):
        # TODO: How to calculate just W_0? Get it from calculate hamiltonian?
        W_L = npy.zeros((self.setup.lmax + 1)**2)
        for x in self.ghat_L.iintegrate(vHt_g, W_L):
            yield None
    
        all_Et = W_L[0] / sqrt(4*pi) 
        all_Eat = - self.Q_L[0] * self.setup.core_B - self.setup.core_C

        D_p = self.D_sp.sum(0)

        # Start with pure kinetic energy + external potential contribution
        self.coreref_k = self.setup.coreref_k.copy()

        # Calculate the eigenvalue dependent contributions
        for k in range(0, self.setup.njcore):
            # From E^a
            I = npy.dot(D_p, self.setup.core_A_kp[k])
            # From \tilde{E}^a
            It = npy.dot(D_p, self.setup.core_At_kp[k])
            # Include all corrections to eigenvalue
            self.coreref_k[k] += all_Et + all_Eat + I - It

        yield None
            
    def adjust_residual(self, R_nG, eps_n, s, u, k):
        if self.in_this_domain:
            H_ii = unpack(self.H_sp[s])
            P_ni = self.P_uni[u]
            coef_ni = (npy.dot(P_ni, H_ii) -
                       npy.dot(P_ni * eps_n[:, None], self.setup.O_ii))

            if self.setup.xc_correction.xc.xcfunc.hybrid > 0.0:
                coef_ni += self.vxx_uni[u]
                
        else:
            coef_ni = None

        for x in self.pt_i.iadd(R_nG, coef_ni, k, communicate=True):
            yield None
            
    def adjust_residual2(self, pR_G, dR_G, eps, u, s, k, n):
        if self.in_this_domain:
            ni = self.get_number_of_partial_waves()
            dP_i = npy.zeros(ni, self.dtype)
        else:
            dP_i = None

        for x in self.pt_i.iintegrate(pR_G, dP_i, k):
            yield None

        if self.in_this_domain:
            H_ii = unpack(self.H_sp[s])
            coef_i = (npy.dot(dP_i, H_ii) -
                      npy.dot(dP_i * eps, self.setup.O_ii))

            if self.setup.xc_correction.xc.xcfunc.hybrid > 0.0:
                coef_i += npy.dot(self.vxx_unii[u, n], dP_i)
        else:
            coef_i = None
            
        for x in self.pt_i.iadd(dR_G, coef_i, k, communicate=True):
            yield None

    def apply_hamiltonian(self, a_nG, b_nG, kpt, calculate_P_uni=True):
        """Apply non-local part of Hamiltonian.

        Non-local part of the Hamiltonian is applied to ``a_nG``
        and added to ``b_nG``.If calcualte_P_uni == False, existing
        P_uni's are used.

        """

        k, u, s = kpt.k, kpt.u, kpt.s
        if self.in_this_domain:
            if calculate_P_uni:
                n = len(a_nG)
                ni = self.get_number_of_partial_waves()
                P_ni = npy.zeros((n, ni), self.dtype)
                for x in self.pt_i.iintegrate(a_nG, P_ni, k):
                    yield None
            else:
                P_ni = self.P_uni[u]
            H_ii = unpack(self.H_sp[s])
            coefs_ni = npy.dot(P_ni, H_ii)
            for x in self.pt_i.iadd(b_nG, coefs_ni, k, communicate=True):
                yield None
        else:
            if calculate_P_uni:
                for x in self.pt_i.iintegrate(a_nG, None, k):
                    yield None
            for x in self.pt_i.iadd(b_nG, None, k, communicate=True):
                yield None

    def apply_overlap(self, a_nG, b_nG, kpt, calculate_P_uni=True):
        """Apply non-local part of the overlap operator.

        Non-local part of the Overlap is applied to ``a_nG``
        and added to ``b_nG``. If calcualte_P_uni == False, existing
        P_uni's are used.
        
        """

        k, u = kpt.k, kpt.u
        if self.in_this_domain:
            if calculate_P_uni:
                n = len(a_nG)
                ni = self.get_number_of_partial_waves()
                P_ni = npy.zeros((n, ni), self.dtype)
                for x in self.pt_i.iintegrate(a_nG, P_ni, k):
                    yield None
            else:
                P_ni = self.P_uni[u]
            coefs_ni = npy.dot(P_ni, self.setup.O_ii)
            for x in self.pt_i.iadd(b_nG, coefs_ni, k, communicate=True):
                yield None
        else:
            if calculate_P_uni:
                for x in self.pt_i.iintegrate(a_nG, None, k):
                    yield None
            for x in self.pt_i.iadd(b_nG, None, k, communicate=True):
                yield None
#         if self.in_this_domain:
#             n = len(a_nG)
#             ni = self.get_number_of_partial_waves()
#             P_ni = npy.zeros((n, ni), self.dtype)
#             self.pt_i.integrate(a_nG, P_ni, k)
#             coefs_ni = npy.dot(P_ni, self.setup.O_ii)
#             self.pt_i.add(b_nG, coefs_ni, k, communicate=True)
#         else:
#             self.pt_i.integrate(a_nG, None, k)
#             self.pt_i.add(b_nG, None, k, communicate=True)
            
    def apply_inverse_overlap(self, a_nG, b_nG, k):
        """Apply non-local part of the approximative inverse overlap operator.

        Non-local part of the overlap operator is applied to ``a_nG``
        and added to ``b_nG``."""

        if self.in_this_domain:
            n = len(a_nG)
            ni = self.get_number_of_partial_waves()
            P_ni = npy.zeros((n, ni), self.dtype)
            self.pt_i.integrate(a_nG, P_ni, k)
            coefs_ni = npy.dot(P_ni, self.setup.C_ii)
            self.pt_i.add(b_nG, coefs_ni, k, communicate=True)
        else:
            self.pt_i.integrate(a_nG, None, k)
            self.pt_i.add(b_nG, None, k, communicate=True)


    def apply_polynomial(self, a_nG, b_nG, k, poly):
        """Apply non-local part of the polynomial operator.

        Currently supports only linear:
        p(x,y,z) = a + b_x x + b_y y + b_z z 

        Non-local part of the polynomial operator is applied to ``a_nG``
        and added to ``b_nG``. K-point wavevector is given by ``k`` and 
        polynomial by ``poly``."""
        
        if self.in_this_domain:
            # number of wavefunctions, psit_nG
            n = len(a_nG)
            # number of partial waves, pt_nG
            ni = self.get_number_of_partial_waves()
            # allocate memory and calculate coefficients P_ni = <pt_i|psit_nG>
            P_ni = npy.zeros((n, ni), self.dtype)
            self.pt_i.integrate(a_nG, P_ni, k)
            
            # indexes of Delta_L,i_1,i_2
            l0 = 0
            ly = 1
            lz = 2
            lx = 3
            lxy   = 4
            lyz   = 5
            lz2r2 = 6
            lxz   = 7
            lx2y2 = 8

            # calculate coefficient 
            # ---------------------
            #
            # coeffs_ni =
            #   sum_i,j ( P_nj * c0 * 1_ij
            #             + P_nj * cx * x_ij
            #             + P_nj * cy * y_ij
            #             + P_nj * cz * z_ij
            #             + ...
            #
            # where (see spherical_harmonics.py)
            #
            #   1_ij = sqrt(4pi) Delta_0ij
            #   y_ij = sqrt(4pi/3) Delta_1ij
            #   z_ij = sqrt(4pi/3) Delta_2ij
            #   x_ij = sqrt(4pi/3) Delta_3ij
            #   xy_ij = sqrt(4pi/15) Delta_4ij
            #   yz_ij = sqrt(4pi/15) Delta_5ij
            #   xz_ij = sqrt(4pi/15) Delta_7ij 
            # ...

            Delta_Lii = self.setup.Delta_Lii

            #   1_ij = sqrt(4pi) Delta_0ij
            #   y_ij = sqrt(4pi/3) Delta_1ij
            #   z_ij = sqrt(4pi/3) Delta_2ij
            #   x_ij = sqrt(4pi/3) Delta_3ij
            oneij = npy.sqrt(4.*npy.pi) \
                * npy.dot(P_ni, Delta_Lii[:,:,0])
            yij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,1])
            zij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,2])
            xij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,3])

            # coefficients
            # coefs_ni = sum_j ( <phi_i| f(x,y,z) | phi_j>
            #                    - <phit_i| f(x,y,z) | phit_j> ) P_nj
            coefs_ni = \
                poly.coeff(0,0,0) * oneij \
                + poly.coeff(0,0,1) * zij \
                + poly.coeff(1,0,0) * xij \
                + poly.coeff(0,1,0) * yij \
#                + poly.coeff(2,0,0) * x2ij \
#                + poly.coeff(1,1,0) * xyij \
#                + poly.coeff(0,2,0) * y2ij \
#                + poly.coeff(0,1,1) * yzij \
#                + poly.coeff(0,0,2) * z2ij \
#                + poly.coeff(1,0,1) * xzij

            # add partial wave pt_nG to psit_nG with proper coefficient
            self.pt_i.add(b_nG, coefs_ni, k, communicate=True)
        else:
            self.pt_i.integrate(a_nG, None, k)
            self.pt_i.add(b_nG, None, k, communicate=True)


    def apply_linear_field(self, a_nG, b_nG, k, c0, cxyz):
        """Apply non-local part of the linear field."""
        
        if self.in_this_domain:
            # number of wavefunctions, psit_nG
            n = len(a_nG)
            # number of partial waves, pt_nG
            ni = self.get_number_of_partial_waves()
            # allocate memory and calculate coefficients P_ni = <pt_i|psit_nG>
            P_ni = npy.zeros((n, ni), self.dtype)
            self.pt_i.integrate(a_nG, P_ni, k)
            
            # indexes of Delta_L,i_1,i_2
            l0 = 0
            ly = 1
            lz = 2
            lx = 3

            # calculate coefficient 
            # ---------------------
            #
            # coeffs_ni =
            #   P_nj * c0 * 1_ij
            #   + P_nj * cx * x_ij
            #
            # where (see spherical_harmonics.py)
            #
            #   1_ij = sqrt(4pi) Delta_0ij
            #   y_ij = sqrt(4pi/3) Delta_1ij
            #   z_ij = sqrt(4pi/3) Delta_2ij
            #   x_ij = sqrt(4pi/3) Delta_3ij
            # ...

            Delta_Lii = self.setup.Delta_Lii

            #   1_ij = sqrt(4pi) Delta_0ij
            #   y_ij = sqrt(4pi/3) Delta_1ij
            #   z_ij = sqrt(4pi/3) Delta_2ij
            #   x_ij = sqrt(4pi/3) Delta_3ij
            oneij = npy.sqrt(4.*npy.pi) \
                * npy.dot(P_ni, Delta_Lii[:,:,0])
            yij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,1])
            zij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,2])
            xij = npy.sqrt(4.*npy.pi / 3.) \
                * npy.dot(P_ni, Delta_Lii[:,:,3])

            # coefficients
            # coefs_ni = sum_j ( <phi_i| f(x,y,z) | phi_j>
            #                    - <phit_i| f(x,y,z) | phit_j> ) P_nj
            coefs_ni = ( c0 * oneij 
                         + cxyz[0] * xij + cxyz[1] * yij + cxyz[2] * zij )

            # add partial wave pt_nG to psit_nG with proper coefficient
            self.pt_i.add(b_nG, coefs_ni, k, communicate=True)
        else:
            self.pt_i.integrate(a_nG, None, k)
            self.pt_i.add(b_nG, None, k, communicate=True)


    def symmetrize(self, D_aii, map_sa, s, response=False):
        D_ii = self.setup.symmetrize(self.a, D_aii, map_sa)
        if response:
            self.Dresp_sp[s] = pack(D_ii)
        else:
            self.D_sp[s] = pack(D_ii)

    def calculate_force_kpoint(self, kpt):
        f_n = kpt.f_n
        eps_n = kpt.eps_n
        psit_nG = kpt.psit_nG
        s = kpt.s
        u = kpt.u
        k = kpt.k
        if self.in_this_domain:
            P_ni = cc(self.P_uni[u])
            nb = P_ni.shape[0]
            H_ii = unpack(self.H_sp[s])
            O_ii = self.setup.O_ii
            ni = self.setup.ni
            F_nic = npy.zeros((nb, ni, 3), self.dtype)
            # ???? Optimization: Take the real value of F_nk * P_ni early.
            self.pt_i.derivative(psit_nG, F_nic, k)
            F_nic.shape = (nb, ni * 3)
            F_nic *= f_n[:, None]
            F_iic = npy.dot(H_ii, npy.dot(npy.transpose(P_ni), F_nic))
            F_nic *= eps_n[:, None]
            F_iic -= npy.dot(O_ii, npy.dot(npy.transpose(P_ni), F_nic))
            F_iic *= 2.0
            F = self.F_c
            F_iic.shape = (ni, ni, 3)
            for i in range(ni):
                F += real(F_iic[i, i])
        else:
            self.pt_i.derivative(psit_nG, None, k)

    def calculate_force(self, vHt_g, nt_g, vt_G):
        if self.in_this_domain:
            lmax = self.setup.lmax
            # ???? Optimization: do the sum over L before the sum over g and G.
            F_Lc = npy.zeros(((lmax + 1)**2, 3))
            self.ghat_L.derivative(vHt_g, F_Lc)
            if self.vhat_L is not None:
                self.vhat_L.derivative(nt_g, F_Lc) 
            
            Q_L = self.Q_L
            F = self.F_c
            F[:] += npy.dot(Q_L, F_Lc)

            # Force from smooth core charge:
##            self.nct.derivative(vt_G, F[npy.newaxis, :]) 
            self.nct.derivative(vt_G, npy.reshape(F, (1, 3)))  # numpy!

            # Force from zero potential:
            self.vbar.derivative(nt_g, npy.reshape(F, (1, 3)))

            dF = npy.zeros(((lmax + 1)**2, 3))
            for neighbor in self.neighbors:
                for c in range(3):
                    dF[:, c] += npy.dot(neighbor.dvdr_LLc[:, :, c],
                                        neighbor.nucleus().Q_L)
            F += npy.dot(self.Q_L, dF)
        else:
            if self.ghat_L is not None:
                self.ghat_L.derivative(vHt_g, None)
                if self.vhat_L is not None:
                    self.vhat_L.derivative(nt_g, None)
                
            if self.nct is not None:
                self.nct.derivative(vt_G, None)
                
            if self.vbar is not None:
                self.vbar.derivative(nt_g, None)

    def get_nearest_grid_point(self, gd):
        """Return index of nearest grid point.
        
        The nearest grid point can be on a different CPU than the one the
        nucleus belongs to (i.e. return can be negative, or larger than
        gd.end_c), in which case something clever should be done.
        """
        return npy.around(gd.N_c * self.spos_c).astype(int) - gd.beg_c

    def get_density_correction(self, spin, nspins):
        """Integrated atomic density correction.

        Get the integrated correction to the pseuso density relative to
        the all-electron density.
        """
        return sqrt(4 * pi) * (
            npy.dot(self.D_sp[spin], self.setup.Delta_pL[:, 0])
            + self.setup.Delta0 / nspins)

    def add_density_correction(self, n_sg, nspins, gd, splines={}):
        """Add atomic density correction function.

        Add the function correcting the pseuso density to the all-electron
        density, to the density array `n_sg`.
        """
        # Load splines
        symbol = self.setup.symbol
        if not symbol in splines:
            phi_j, phit_j, nc, nct = self.setup.get_partial_waves()[:4]
            splines[symbol] = (phi_j, phit_j, nc, nct)
        else:
            phi_j, phit_j, nc, nct = splines[symbol]

        # Create localized functions from splines
        create = create_localized_functions
        phi_i = create(phi_j, gd, self.spos_c)
        phit_i = create(phit_j, gd, self.spos_c)

        nc = create([nc], gd, self.spos_c)
        nct = create([nct], gd, self.spos_c)

        # The correct normalizations are:
        Nc = self.setup.Nc
        Nct = -(self.setup.Delta0 * sqrt(4 * pi)
                + self.setup.Z - self.setup.Nc)

        # Actual normalizations:
        Nc0 = nc.norm()[0, 0]
        Nct0 = nct.norm()[0, 0]

        for s in range(nspins):
            # Numeric and analytic integrations of density corrections:
            Inum = (Nc0 - Nct0) / nspins
            Iana = ((Nc - Nct) / nspins +
                    sqrt(4 * pi) * npy.dot(self.D_sp[s],
                                           self.setup.Delta_pL[:, 0]))

            # Add density corrections to input array n_G
            Inum += phi_i.add_density2(n_sg[s], self.D_sp[s])
            Inum += phit_i.add_density2(n_sg[s], -self.D_sp[s])
            if Nc != 0:
                nc.add(n_sg[s], npy.ones(1) / nspins)
                nct.add(n_sg[s], -npy.ones(1) / nspins)

            # Correct density, such that correction is norm-conserving
            g_c = tuple(self.get_nearest_grid_point(gd) % gd.N_c)
            n_sg[s][g_c] += (Iana - Inum) / gd.dv
        
    def wannier_correction(self, G, c, u, u1):
        """
        Calculate the correction to the wannier integrals Z,
        given by (Eq. 27 ref1)::

                          -i G.r    
            Z   = <psi | e      |psi >
             nm       n             m
                            
                           __                __
                   ~      \              a  \     a  a    a  *
            Z    = Z    +  ) exp[-i G . R ]  )   P  O   (P  )
             nmx    nmx   /__            x  /__   ni ii'  mi'

                           a                 ii'

        Note that this correction is an approximation assuming that the
        exponential varies slowly over the extent of the augmentation sphere.

        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005) 
        """
        P_ni = self.P_uni[u]
        P1_ni = self.P_uni[u1]
        O_ii = self.setup.O_ii
        e = exp(-2.j * pi * G * self.spos_c[c])
        Z_nn = e * npy.dot(npy.dot(P_ni, O_ii), cc(npy.transpose(P1_ni)))

        return Z_nn
