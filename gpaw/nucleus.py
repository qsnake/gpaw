# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Nucleus class.

A Paw object has a list of nuclei. Each nucleus is described by a
``Setup`` object and a scaled position plus some extra stuff..."""

from math import pi, sqrt

import Numeric as num

from gpaw.utilities.complex import real, cc
from gpaw.localized_functions import create_localized_functions
from gpaw.utilities import unpack, pack
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
     ``typecode``  Data type of wave functions (``Float`` or ``Complex``).
     ``neighbors`` List of overlapping neighbor nuclei.
     ============= ========================================================

    Localized functions:
     ========== ===========================================================
     ``nct``    Pseudo core electron density.
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
     ``dH_sp`` Atomic Hamiltonian correction (``\Delta H_{\sigma i_1i_2}^a``).
     ``Q_L``   Multipole moments  (``Q_{\ell m}^a``).
     ``F_c``   Force.
     ========= ===============================================================

    Parallel stuff: ``comm``, ``rank`` and ``in_this_domain``
    """
    def __init__(self, setup, a, typecode):
        """Construct a ``Nucleus`` object."""
        self.setup = setup
        self.a = a
        self.typecode = typecode
        lmax = setup.lmax
        self.Q_L = num.zeros((lmax + 1)**2, num.Float)
        self.neighbors = []
        self.spos_c = num.array([-1.0, -1.0, -1.0])

        self.rank = -1
        self.comm = mpi.serial_comm
        self.in_this_domain = False

        self.pt_i = None
        self.vbar = None
        self.ghat_L = None
        self.vhat_L = None
        self.nct = None
        self.mom = num.array(0.0)

    def __cmp__(self, other):
        """Ordering of nuclei.

        Use sequence number ``a`` to sort lists."""
        
        return cmp(self.a, other.a)
    
    def allocate(self, nspins, nmyu, nbands):
        ni = self.get_number_of_partial_waves()
        np = ni * (ni + 1) // 2
        self.D_sp = num.zeros((nspins, np), num.Float)
        self.H_sp = num.zeros((nspins, np), num.Float)
        self.P_uni = num.zeros((nmyu, nbands, ni), self.typecode)
        self.F_c = num.zeros(3, num.Float)
        if self.setup.xcname == 'EXX':
            self.vxx_sni = num.zeros((nspins, nbands, ni), self.typecode)

    def reallocate(self, nbands):
        nu, nao, ni = self.P_uni.shape
        if nbands < nao:
            self.P_uni = self.P_uni[:, :nbands, :].copy()
        else:
            P_uni = num.zeros((nu, nbands, ni), self.typecode)
            P_uni[:, :nao, :] = self.P_uni
            self.P_uni = P_uni

    def move(self, spos_c, gd, finegd, k_ki, lfbc, domain,
             my_nuclei, pt_nuclei, ghat_nuclei,
             nspins, nmyu, nbands):
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

        # Shortcut:
        create = create_localized_functions

        # Projectors:
        pt_j = self.setup.pt_j
        pt_i = create(pt_j, gd, spos_c, typecode=self.typecode, lfbc=lfbc)

        if self.typecode == num.Complex and pt_i is not None:
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

        #Step function
        stepf = self.setup.stepf
        stepf = create([stepf], finegd, spos_c, lfbc=lfbc)
        self.stepf = stepf
            
        # Potential:
        vhat_l = self.setup.vhat_l
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
        if self.nct is not None:
            self.nct.set_communicator(self.comm, rank)

        if self.comm.size > 1:
            # Make MPI-group communicators:
            flags = num.array([1 * (pt_i is not None) +
                               2 * (vbar is not None) +
                               4 * (ghat_L is not None)])

            flags_r = num.zeros((self.comm.size, 1), num.Int)
            self.comm.all_gather(flags, flags_r)
            for mask, lfs in [(1, [pt_i]),
                              (2, [vbar, stepf]),
                              (4, [ghat_L, vhat_L])]:
                group = [r for r, flags in enumerate(flags_r) if flags & mask]
                root = group.index(rank)
                comm = domain.get_communicator(group)
                for lf in lfs:
                    if lf is not None:
                        lf.set_communicator(comm, root)

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
            Nct = -(self.setup.Delta0 * sqrt(4 * pi) + self.setup.Nv)
            self.nct.normalize(Nct)
        else:
            self.comm.sum(0.0)

    def initialize_atomic_orbitals(self, gd, k_ki, lfbc):
        phit_j = self.setup.phit_j
        self.phit_i = create_localized_functions(
            phit_j, gd, self.spos_c, typecode=self.typecode,
            cut=True, forces=False, lfbc=lfbc)
        if self.typecode == num.Complex and self.phit_i is not None:
            self.phit_i.set_phase_factors(k_ki)

    def get_number_of_atomic_orbitals(self):
        return self.setup.niAO

    def get_number_of_partial_waves(self):
        return self.setup.ni
    
    def create_atomic_orbitals(self, psit_iG, k):
        if self.phit_i is None:
            # Nothing to do in this domain:
            return

        coefs = num.identity(len(psit_iG), psit_iG.typecode())
        self.phit_i.add(psit_iG, coefs, k)

    def add_atomic_density(self, nt_sG, magmom, hund):
        if self.phit_i is None:
            # Nothing to do in this domain:
            print 'self.phitj is none'
            return

        ns = len(nt_sG)
        ni = self.get_number_of_partial_waves()
        niao = self.get_number_of_atomic_orbitals()
        f_si = num.zeros((ns, niao), num.Float)
        
        i = 0
        for n, l, f in zip(self.setup.n_j, self.setup.l_j, self.setup.f_j):
            degeneracy = 2 * l + 1
            f = int(f)
            if n < 0:
                break
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
                    if f == 2 * degeneracy:
                        mag = 0.0
                    else:
                        mag = min(magmom, degeneracy, f)
                    f_si[0, i:i + degeneracy] = 0.5 * (f + mag) / degeneracy
                    f_si[1, i:i + degeneracy] = 0.5 * (f - mag) / degeneracy
                    magmom -= mag
                
            i += degeneracy
        assert i == niao

        if self.in_this_domain:
            D_sii = num.zeros((ns, ni, ni), num.Float)
            for i in range(niao):
                D_sii[:, i, i] = f_si[:, i]
            for s in range(ns):
                self.D_sp[s] = pack(D_sii[s])

        for s in range(ns):
            self.phit_i.add_density(nt_sG[s], f_si[s])

    def add_smooth_core_density(self, nct_G, nspins):
        if self.nct is not None:
            self.nct.add(nct_G, num.array([1.0 / nspins]))

    def add_compensation_charge(self, nt2):
        self.ghat_L.add(nt2, self.Q_L)
        
    def add_hat_potential(self, vt2):
        self.vhat_L.add(vt2, self.Q_L)

    def add_localized_potential(self, vt2):
        if self.vbar is not None:
            self.vbar.add(vt2, num.array([1.0]))
        
    def calculate_projections(self, kpt):
        if self.in_this_domain:
            P_ni = self.P_uni[kpt.u]
            P_ni[:] = 0.0 # why????
            self.pt_i.integrate(kpt.psit_nG, P_ni, kpt.k)
        else:
            self.pt_i.integrate(kpt.psit_nG, None, kpt.k)

    def calculate_multipole_moments(self):
        if self.in_this_domain:
            self.Q_L[:] = num.dot(num.sum(self.D_sp), self.setup.Delta_pL)
            self.Q_L[0] += self.setup.Delta0
        self.comm.broadcast(self.Q_L, self.rank)

    def calculate_magnetic_moments(self):
        if self.in_this_domain:
            dif = self.D_sp[0,:] - self.D_sp[1,:]
            self.mom = num.array(sqrt(4 * pi) *
                                 num.dot(dif, self.setup.Delta_pL[:,0]))
        self.comm.broadcast(self.mom, self.rank)
        
    def calculate_hamiltonian(self, nt_g, vHt_g):
        if self.in_this_domain:
            s = self.setup
            W_L = num.zeros((s.lmax + 1)**2, num.Float)
            for neighbor in self.neighbors:
                W_L += num.dot(neighbor.v_LL, neighbor.nucleus().Q_L)
            U = 0.5 * num.dot(self.Q_L, W_L)

            self.vhat_L.integrate(nt_g, W_L)
            self.ghat_L.integrate(vHt_g, W_L)

            D_p = num.sum(self.D_sp)
            dH_p = (s.K_p + s.M_p + s.MB_p + 2.0 * num.dot(s.M_pp, D_p) +
                    num.dot(s.Delta_pL, W_L))

            Exc = s.xc.calculate_energy_and_derivatives(self.D_sp, self.H_sp)
            if s.xcname == 'EXX': # XXX EXX hack
                Exc = s.ExxC - num.dot(D_p, (s.X_p + num.dot(s.M_pp, D_p)))
                self.H_sp -= s.X_p + 2.0 * num.dot(s.M_pp, D_p)

            Ekin = num.dot(s.K_p, D_p) + s.Kc

            Ebar = s.MB + num.dot(s.MB_p, D_p)
            Epot = U + s.M + num.dot(D_p, (s.M_p + num.dot(s.M_pp, D_p)))
            for H_p in self.H_sp:
                H_p += dH_p

            # Move this kinetic energy contribution to Paw.py: ????!!!!
            Ekin -= num.dot(self.D_sp[0], self.H_sp[0])
            if len(self.D_sp) == 2:
                Ekin -= num.dot(self.D_sp[1], self.H_sp[1])

            return Ekin, Epot, Ebar, Exc
        
        else:
            self.vhat_L.integrate(nt_g, None)
            self.ghat_L.integrate(vHt_g, None)
            return 0.0, 0.0, 0.0, 0.0

    def adjust_residual(self, R_nG, eps_n, s, u, k):
        if self.in_this_domain:
            H_ii = unpack(self.H_sp[s])
            P_ni = self.P_uni[u]
            coefs_ni =  (num.dot(P_ni, H_ii) -
                         num.dot(P_ni * eps_n[:, None], self.setup.O_ii))
            self.pt_i.add(R_nG, coefs_ni, k, communicate=True)
        else:
            self.pt_i.add(R_nG, None, k, communicate=True)
            
    def adjust_residual2(self, pR_G, dR_G, eps, s, k):
        if self.in_this_domain:
            ni = self.get_number_of_partial_waves()
            dP_i = num.zeros(ni, self.typecode)
            self.pt_i.integrate(pR_G, dP_i, k)
        else:
            self.pt_i.integrate(pR_G, None, k)

        if self.in_this_domain:
            H_ii = unpack(self.H_sp[s])
            coefs_i = (num.dot(dP_i, H_ii) -
                       num.dot(dP_i * eps, self.setup.O_ii))
            self.pt_i.add(dR_G, coefs_i, k, communicate=True)
        else:
            self.pt_i.add(dR_G, None, k, communicate=True)

    def apply_hamiltonian(self, psit_G, Htpsit_G, s, k):
        """Applies the non-local part of the Hamiltonian to the
           wave function psit_G and adds the result to Htpsit_G"""
        if self.in_this_domain:
            ni = self.get_number_of_partial_waves()
            P_i = num.zeros(ni, self.typecode)
            self.pt_i.integrate(psit_G, P_i, k)
        else:
            self.pt_i.integrate(psit_G, None, k)

        if self.in_this_domain:
            H_ii = unpack(self.H_sp[s])
            coefs_i = (num.dot(P_i, H_ii))
            self.pt_i.add(Htpsit_G, coefs_i, k, communicate=True)
        else:
            self.pt_i.add(Htpsit_G, None, k, communicate=True)

    def apply_overlap(self, psit_G, Spsit_G, k):
        """Applies the non-local part of the overlap operator to the
           wave function psit_G and adds the result to Spsit_G"""
        if self.in_this_domain:
            ni = self.get_number_of_partial_waves()
            P_i = num.zeros(ni, self.typecode)
            self.pt_i.integrate(psit_G, P_i, k)
        else:
            self.pt_i.integrate(psit_G, None, k)

        if self.in_this_domain:
            coefs_i = (num.dot(P_i, self.setup.O_ii))
            self.pt_i.add(Spsit_G, coefs_i, k, communicate=True)
        else:
            self.pt_i.add(Spsit_G, None, k, communicate=True)

    def symmetrize(self, D_aii, map_sa, s):
        D_ii = self.setup.symmetrize(self.a, D_aii, map_sa)
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
            F_nic = num.zeros((nb, ni, 3), self.typecode)
            # ???? Optimization: Take the real value of F_nk * P_ni early.
            self.pt_i.derivative(psit_nG, F_nic, k)
            F_nic.shape = (nb, ni * 3)
            F_nic *= f_n[:, None]
            F_iic = num.dot(H_ii, num.dot(num.transpose(P_ni), F_nic))
            F_nic *= eps_n[:, None]
            F_iic -= num.dot(O_ii, num.dot(num.transpose(P_ni), F_nic))
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
            F_Lc = num.zeros(((lmax + 1)**2, 3), num.Float)
            self.ghat_L.derivative(vHt_g, F_Lc)
            self.vhat_L.derivative(nt_g, F_Lc) 
            
            Q_L = self.Q_L
            F = self.F_c
            F[:] += num.dot(Q_L, F_Lc)

            # Force from smooth core charge:
##            self.nct.derivative(vt_G, F[num.NewAxis, :]) 
            self.nct.derivative(vt_G, num.reshape(F, (1, 3)))  # numpy!

            # Force from zero potential:
            self.vbar.derivative(nt_g, num.reshape(F, (1, 3)))

            dF = num.zeros(((lmax + 1)**2, 3), num.Float)
            for neighbor in self.neighbors:
                for c in range(3):
                    dF[:, c] += num.dot(neighbor.dvdr_LLc[:, :, c],
                                        neighbor.nucleus().Q_L)
            F += num.dot(self.Q_L, dF)
        else:
            if self.ghat_L is not None:
                self.ghat_L.derivative(vHt_g, None)
                self.vhat_L.derivative(nt_g, None)
                
            if self.nct is None:
                self.comm.sum(num.zeros(3, num.Float), self.rank)
            else:
                self.nct.derivative(vt_G, None)
                
            if self.vbar is not None:
                self.vbar.derivative(nt_g, None)

    def add_density_correction(self, n_sg, nspins, gd, splines={}):
        # Load splines
        symbol = self.setup.symbol
        if not symbol in splines:
            phi_j, phit_j, nc, nct = self.setup.get_partial_waves()
            splines[symbol] = (phi_j, phit_j, nc, nct)
        else:
            phi_j, phit_j, nc, nct = splines[symbol]

        # Create localized functions from splines
        create = create_localized_functions
        phi_i = create(phi_j, gd, self.spos_c)
        phit_i = create(phit_j, gd, self.spos_c)
        nc = create([nc], gd, self.spos_c)
        nct = create([nct], gd, self.spos_c)

        # Normalize core densities:
        Nc = self.setup.Nc
        Nct = -(self.setup.Delta0 * sqrt(4 * pi) + self.setup.Nv)
        if Nc != 0:
            nc.normalize(Nc)
            nct.normalize(Nct)
        
        for s in range(nspins):
            # Numeric and analytic integrations of density corrections
            Inum = 0.0
            Ianal = sqrt(4 * pi) * num.dot(self.D_sp[s],
                                           self.setup.Delta_pL[:,0])

            # Add density corrections to input array n_G
            Inum += phi_i.add_density2(n_sg[s], self.D_sp[s])
            Inum += phit_i.add_density2(n_sg[s], -self.D_sp[s])
            if Nc != 0:
                nc.add(n_sg[s], num.ones(1, num.Float) / nspins)
                nct.add(n_sg[s], -num.ones(1, num.Float) / nspins)

            # Correct density, such that correction is norm-conserving
            Core_c = num.around(gd.N_c * self.spos_c).astype(num.Int) % gd.N_c
            n_sg[s][Core_c] += (Ianal - Inum) / gd.dv
        
