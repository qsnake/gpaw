# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import numpy as np
from ase.units import Bohr, Hartree

from gpaw.paw import PAW
from gpaw.xc_functional import XCFunctional

class GPAW(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """
    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(self)
        return atoms

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        if atoms is None:
            atoms = self.atoms

        self.calculate(atoms, converge=True)

        if force_consistent:
            # Free energy:
            return Hartree * self.hamiltonian.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return Hartree * (self.hamiltonian.Etot + 0.5 * self.hamiltonian.S)

    def get_reference_energy(self):
        return self.wfs.setups.Eref * Hartree
    
    def get_forces(self, atoms):
        """Return the forces for the current state of the atoms."""
        # I believe that the force_call_to_set_positions must be set
        # in order to make sure that psit_nG is correctly initialized
        # from a tarfile reference.
        #
        # TODO: improve i/o interface so the rest of the program doesn't need
        # to distinguish between the ways in which wave functions were obtained
        if (self.forces.F_av is None and
            hasattr(self.wfs, 'kpt_u') and
            not hasattr(self.wfs, 'tci') and
            not isinstance(self.wfs.kpt_u[0].psit_nG, np.ndarray)):
            force_call_to_set_positions = True
        else:
            force_call_to_set_positions = False
        self.calculate(atoms, converge=True,
                       force_call_to_set_positions=force_call_to_set_positions)
        F_av = self.forces.calculate(self.wfs, self.density, self.hamiltonian)
        self.print_forces()
        return F_av * (Hartree / Bohr)
      
    def get_stress(self, atoms):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def calculation_required(self, atoms, quantities):
        if 'stress' in quantities:
            quantities.remove('stress')

        if len(quantities) == 0:
            return False

        if not (self.initialized and self.scf.converged):
            return True

        if (len(atoms) != len(self.atoms) or
            (atoms.get_positions() != self.atoms.get_positions()).any() or
            (atoms.get_atomic_numbers() !=
             self.atoms.get_atomic_numbers()).any() or
            (atoms.get_cell() != self.atoms.get_cell()).any() or
            (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            return True

        if 'forces' in quantities:
            return self.forces.F_av is None

        return False

    def get_number_of_bands(self):
        """Return the number of bands."""
        return self.wfs.nbands 
  
    def get_xc_functional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def get_bz_k_points(self):
        """Return the k-points."""
        return self.wfs.bzk_kc
 
    def get_number_of_spins(self):
        return self.wfs.nspins

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return self.wfs.nspins == 2
    
    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.wfs.ibzk_kc

    def get_k_point_weights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.wfs.weight_k

    def get_pseudo_density(self, spin=None, gridrefinement=1,
                           pad=True, broadcast=True):
        """Return pseudo-density array.

        If *spin* is not given, then the total density is returned.
        Otherwise, the spin up or down density is returned (spin=0 or
        1)."""

        if gridrefinement == 1:
            nt_sG = self.density.nt_sG
            gd = self.density.gd
        elif gridrefinement == 2:
            if self.density.nt_sg is None:
                self.density.interpolate()
            nt_sG = self.density.nt_sg
            gd = self.density.finegd
        else:
            raise NotImplementedError

        if spin is None:
            if self.wfs.nspins == 1:
                nt_G = nt_sG[0]
            else:
                nt_G = nt_sG.sum(axis=0)
        else:
            if self.wfs.nspins == 1:
                nt_G = 0.5 * nt_sG[0]
            else:
                nt_G = nt_sG[spin]

        nt_G = gd.collect(nt_G, broadcast)

        if nt_G is None:
            return None
        
        if pad:
            nt_G = gd.zero_pad(nt_G)

        return nt_G / Bohr**3

    get_pseudo_valence_density = get_pseudo_density  # Don't use this one!
    
    def get_effective_potential(self, spin=0, pad=True):
        """Return pseudo effective-potential."""
        vt_G = self.hamiltonian.vt_sG[spin]
        if pad:
            vt_G = self.gd.zero_pad(vt_G)
        return vt_G * Hartree
    
    def get_pseudo_density_corrections(self):
        """Integrated density corrections.

        Returns the integrated value of the difference between the pseudo-
        and the all-electron densities at each atom.  These are the numbers
        you should add to the result of doing e.g. Bader analysis on the
        pseudo density."""
        if self.wfs.nspins == 1:
            return np.array([self.density.get_correction(a, 0)
                             for a in range(len(self.atoms))])
        else:
            return np.array([[self.density.get_correction(a, spin)
                              for a in range(len(self.atoms))]
                             for spin in range(2)])

    def get_all_electron_density(self, spin=None, gridrefinement=2,
                                 pad=True, broadcast=True):
        """Return reconstructed all-electron density array."""
        n_sG, gd = self.density.get_all_electron_density(
            self.atoms, gridrefinement=gridrefinement)

        if spin is None:
            if self.wfs.nspins == 1:
                n_G = n_sG[0]
            else:
                n_G = n_sG.sum(axis=0)
        else:
            if self.wfs.nspins == 1:
                n_G = 0.5 * n_sG[0]
            else:
                n_G = n_sG[spin]

        n_G = gd.collect(n_G, broadcast)

        if n_G is None:
            return None
        
        if pad:
            n_G = gd.zero_pad(n_G)

        return n_G / Bohr**3

    def get_fermi_level(self):
        """Return the Fermi-level."""
        eFermi = self.occupations.get_fermi_level()
        if eFermi is not None:
            eFermi *= Hartree
        return eFermi

    def get_wigner_seitz_densities(self, spin):
        """Get the weight of the spin-density in Wigner-Seitz cells
        around each atom.

        The density assigned to each atom is relative to the neutral atom,
        i.e. the density sums to zero.
        """
        from gpaw.utilities import wignerseitz
        atom_index = self.gd.empty(dtype=int)
        atom_ac = self.atoms.get_scaled_positions() * self.gd.N_c
        wignerseitz(atom_index, atom_ac, self.gd.beg_c, self.gd.end_c)

        nt_G = self.density.nt_sG[spin]
        weight_a = np.empty(len(self.atoms))
        for a in range(len(self.atoms)):
            # XXX Optimize! No need to integrate in zero-region
            smooth = self.gd.integrate(np.where(atom_index == a, nt_G, 0.0))
            correction = self.density.get_correction(a, spin)
            weight_a[a] = smooth + correction
            
        return weight_a

    def get_dos(self, spin=0, npts=201, width=None):
        """The total DOS.

        Fold eigenvalues with Gaussians, and put on an energy grid.

        returns an (energies, dos) tuple, where energies are relative to the
        vacuum level for non-periodic systems, and the average potentail for
        periodic systems.
        """
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        w_k = self.wfs.weight_k
        Nb = self.wfs.nbands
        energies = np.empty(len(w_k) * Nb)
        weights  = np.empty(len(w_k) * Nb)
        x = 0
        for k, w in enumerate(w_k):
            energies[x:x + Nb] = self.get_eigenvalues(k, spin)
            weights[x:x + Nb] = w
            x += Nb
            
        from gpaw.utilities.dos import fold
        return fold(energies, weights, npts, width)

    def get_wigner_seitz_ldos(self, a, spin=0, npts=201, width=None):
        """The Local Density of States, using a Wigner-Seitz basis function.

        Project wave functions onto a Wigner-Seitz box at atom ``a``, and
        use this as weight when summing the eigenvalues."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold(energies * Hartree, weights, npts, width)
    
    def get_orbital_ldos(self, a,
                         spin=0, angular='spdf', npts=201, width=None):
        """The Local Density of States, using atomic orbital basis functions.

        Project wave functions onto an atom orbital at atom ``a``, and
        use this as weight when summing the eigenvalues.

        The atomic orbital has angular momentum ``angular``, which can be
        's', 'p', 'd', 'f', or any combination (e.g. 'sdf').

        An integer value for ``angular`` can also be used to specify a specific
        projector function to project onto.
        """
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import raw_orbital_LDOS, fold
        energies, weights = raw_orbital_LDOS(self, a, spin, angular)
        return fold(energies * Hartree, weights, npts, width)

    def get_all_electron_ldos(self, mol, spin=0, npts=201, width=None,
                              wf_k=None, P_aui=None, lc=None, raw=False):
        """The Projected Density of States, using all-electron wavefunctions.

        Projects onto a pseudo_wavefunctions (wf_k) corresponding to some band
        n and uses P_aui ([paw.nuclei[a].P_uni[:,n,:] for a in atoms]) to
        obtain the all-electron overlaps.
        Instead of projecting onto a wavefunctions a molecular orbital can
        be specified by a linear combination of weights (lc)
        """
        from gpaw.utilities.dos import all_electron_LDOS, fold

        if raw:
            return all_electron_LDOS(self, mol, spin, lc=lc,
                                     wf_k=wf_k, P_aui=P_aui)
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        energies, weights = all_electron_LDOS(self, mol, spin,
                                              lc=lc, wf_k=wf_k, P_aui=P_aui)
        return fold(energies * Hartree, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0, broadcast=True,
                                 pad=True):
        """Return pseudo-wave-function array.

        Units: 1/Angstrom^(3/2)
        """
        if pad:
            return self.gd.zero_pad(self.get_pseudo_wave_function(
                band, kpt, spin, broadcast, False))
        psit_G = self.wfs.get_wave_function_array(band, kpt, spin)
        if broadcast:
            if not self.wfs.world.rank == 0:
                psit_G = self.gd.empty(dtype=self.wfs.dtype, global_array=True)
            self.wfs.world.broadcast(psit_G, 0)
            return psit_G / Bohr**1.5
        elif self.wfs.world.rank == 0:
            return psit_G / Bohr**1.5

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        """Return eigenvalue array."""
        eps_n = self.wfs.collect_eigenvalues(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                assert eps_n is None
                eps_n = np.empty(self.wfs.nbands)
            self.wfs.world.broadcast(eps_n, 0)
        if eps_n is not None:
            return eps_n * Hartree

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True):
        """Return occupation array."""
        f_n = self.wfs.collect_occupations(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                assert f_n is None
                f_n = np.empty(self.wfs.nbands)
            self.wfs.world.broadcast(f_n, 0)
        return f_n
    
    def get_xc_difference(self, xcname):
        xcfunc = XCFunctional(xcname, self.hamiltonian.nspins)
        if xcfunc.mgga or xcfunc.orbital_dependent:
            self.converge_wave_functions()
        return self.hamiltonian.get_xc_difference(xcfunc, self.wfs,
                                                  self.density,
                                                  self.atoms) * Hartree

    def get_nonselfconsistent_eigenvalues(self, xcname):
        from gpaw.xc_functional import XCFunctional
        wfs = self.wfs
        oldxc = self.hamiltonian.xc.xcfunc

        def shiftxc(calc, xc, energy_only=False):
            if isinstance(xc, str):
                xc = XCFunctional(xc, calc.wfs.nspins)
            elif isinstance(xc, dict):
                xc = XCFunctional(xc.copy(), calc.wfs.nspins)
            xc.set_non_local_things(calc.density, calc.hamiltonian, calc.wfs,
                                    calc.atoms, energy_only=energy_only)
            calc.hamiltonian.xc.set_functional(xc)
            calc.hamiltonian.xc.set_positions(
                calc.atoms.get_scaled_positions() % 1.0)
            for setup in calc.wfs.setups:
                setup.xc_correction.xc.set_functional(xc)
                if xc.mgga:
                    setup.xc_correction.initialize_kinetic(setup.data)

        # Read in stuff from the file
        assert wfs.kpt_u[0].psit_nG is not None, 'gpw file must contain wfs!'
        wfs.initialize_wave_functions_from_restart_file()
        self.set_positions()
        for kpt in wfs.kpt_u:
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        # Change the xc functional
        shiftxc(self, xcname)

        # Recalculate the effective potential
        self.hamiltonian.update(self.density)
        if not wfs.eigensolver.initialized:
            wfs.eigensolver.initialize(wfs)

        # Apply Hamiltonian and get new eigenvalues, occupation, and energy
        for kpt in wfs.kpt_u:
            wfs.eigensolver.subspace_diagonalize(
                self.hamiltonian, wfs, kpt, rotate=False)

        # Change xc functional back to the original
        shiftxc(self, oldxc)

        eig_skn = np.array([[self.get_eigenvalues(kpt=k, spin=s)
                             for k in range(wfs.nibzkpts)]
                            for s in range(wfs.nspins)])
        return eig_skn

    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin):
        """Initial guess for the shape of wannier functions.

        Use initial guess for wannier orbitals to determine rotation
        matrices U and C.
        """
        if not self.wfs.gamma:
            raise NotImplementedError
        from ase.dft.wannier import rotation_from_projection
        proj_knw = self.get_projections(initialwannier, spin)
        U_ww, C_ul = rotation_from_projection(proj_knw[0],
                                              fixedstates[0],
                                              ortho=True)
        return [C_ul], U_ww[np.newaxis]

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        c = dirG.tolist().index(1)
        k_kc = self.wfs.bzk_kc
        G = k_kc[nextkpoint, c] - k_kc[kpoint, c] - G_I[c]

        return self.get_wannier_integrals(c, spin, kpoint,
                                          nextkpoint, G, nbands)

    def get_wannier_integrals(self, c, s, k, k1, G, nbands=None):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.wfs.nspins
        kpt_rank, u = divmod(k + len(self.wfs.ibzk_kc) * s,
                             len(self.wfs.kpt_u))
        kpt_rank1, u1 = divmod(k1 + len(self.wfs.ibzk_kc) * s,
                               len(self.wfs.kpt_u))
        kpt_u = self.wfs.kpt_u

        # XXX not for the kpoint/spin parallel case
        assert self.wfs.kpt_comm.size == 1

        # If calc is a save file, read in tar references to memory
        self.wfs.initialize_wave_functions_from_restart_file()
        
        # Get pseudo part
        Z_nn = self.gd.wannier_matrix(kpt_u[u].psit_nG,
                                      kpt_u[u1].psit_nG, c, G, nbands)

        # Add corrections
        self.add_wannier_correction(Z_nn, G, c, u, u1, nbands)

        self.gd.comm.sum(Z_nn, 0)
            
        return Z_nn

    def add_wannier_correction(self, Z_nn, G, c, u, u1, nbands=None):
        """
        Calculate the correction to the wannier integrals Z,
        given by (Eq. 27 ref1)::

                          -i G.r    
            Z   = <psi | e      |psi >
             nm       n             m
                            
                           __                __
                   ~      \              a  \     a*  a    a   
            Z    = Z    +  ) exp[-i G . R ]  )   P   O    P  
             nmx    nmx   /__            x  /__   ni  ii'  mi'

                           a                 ii'

        Note that this correction is an approximation that assumes the
        exponential varies slowly over the extent of the augmentation sphere.

        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005) 
        """

        if nbands is None:
            nbands = self.wfs.nbands
            
        P_ani = self.wfs.kpt_u[u].P_ani
        P1_ani = self.wfs.kpt_u[u1].P_ani
        spos_av = self.atoms.get_scaled_positions()
        for a, P_ni in P_ani.items():
            P_ni = P_ani[a][:nbands]
            P1_ni = P1_ani[a][:nbands]
            O_ii = self.wfs.setups[a].O_ii
            e = np.exp(-2.j * np.pi * G * spos_av[a, c])
            Z_nn += e * np.dot(np.dot(P_ni.conj(), O_ii), P1_ni.T)

    def get_projections(self, locfun, spin=0):
        """Project wave functions onto localized functions

        Determine the projections of the Kohn-Sham eigenstates
        onto specified localized functions of the format::

          locfun = [[spos_c, l, sigma], [...]]

        spos_c can be an atom index, or a scaled position vector. l is
        the angular momentum, and sigma is the (half-) width of the
        radial gaussian.

        Return format is::

          f_kni = <psi_kn | f_i>

        where psi_kn are the wave functions, and f_i are the specified
        localized functions.

        As a special case, locfun can be the string 'projectors', in which
        case the bound state projectors are used as localized functions.
        """

        wfs = self.wfs
        
        if locfun == 'projectors':
            f_kin = []
            for kpt in wfs.kpt_u:
                if kpt.s == spin:
                    f_in = []
                    for a, P_ni in kpt.P_ani.items():
                        i = 0
                        setup = wfs.setups[a]
                        for l, n in zip(setup.l_j, setup.n_j):
                            if n >= 0:
                                for j in range(i, i + 2 * l + 1):
                                    f_in.append(P_ni[:, j])
                            i += 2 * l + 1
                    f_kin.append(f_in)
            f_kni = np.array(f_kin).transpose(0, 2, 1)
            return f_kni.conj()

        from gpaw.lfc import LocalizedFunctionsCollection as LFC
        from gpaw.spline import Spline
        from gpaw.utilities import fac

        nkpts = len(wfs.ibzk_kc)
        nbf = np.sum([2 * l + 1 for pos, l, a in locfun])
        f_kni = np.zeros((nkpts, wfs.nbands, nbf), wfs.dtype)

        bf = 0

        spos_ac = self.atoms.get_scaled_positions() % 1.0
        spos_xc = []
        splines_x = []
        for spos_c, l, sigma in locfun:
            if isinstance(spos_c, int):
                spos_c = spos_ac[spos_c]
            spos_xc.append(spos_c)
            
            alpha = .5 * Bohr**2 / sigma**2
            r = np.linspace(0, 6. * sigma, 500)
            f_g = (fac[l] * (4 * alpha)**(l + 3 / 2.) *
                   np.exp(-a * r**2) /
                   (np.sqrt(4 * np.pi) * fac[2 * l + 1]))
            splines_x.append([Spline(l, rmax=r[-1], f_g=f_g, points=61)])
            
        lf = LFC(wfs.gd, splines_x, wfs.kpt_comm, dtype=wfs.dtype)
        if not wfs.gamma:
            lf.set_k_points(wfs.ibzk_qc)
        lf.set_positions(spos_xc)

        k = 0
        f_ani = lf.dict(wfs.nbands)
        for kpt in wfs.kpt_u:
            if kpt.s != spin:
                continue
            lf.integrate(kpt.psit_nG[:], f_ani, kpt.q)
            i1 = 0
            for x, f_ni in f_ani.items():
                i2 = i1 + f_ni.shape[1]
                f_kni[k, :, i1:i2] = f_ni
                i1 = i2
            k += 1

        return f_kni.conj()

    def get_dipole_moment(self, atoms=None):
        """Return the total dipole moment in ASE units."""
        rhot_g = self.density.rhot_g
        return self.density.finegd.calculate_dipole_moment(rhot_g) * Bohr

    def get_magnetic_moment(self, atoms=None):
        """Return the total magnetic moment."""
        return self.occupations.magmom

    def get_magnetic_moments(self, atoms=None):
        """Return the local magnetic moments within augmentation spheres"""
        magmom_a = self.density.estimate_magnetic_moments()
        M = self.occupations.magmom
        if abs(M) > 1e-7:
            magmom_a *= M / magmom_a.sum()
        return magmom_a
        
    def get_number_of_grid_points(self):
        return self.gd.N_c

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.get_xc_difference('XC-9-1.0')
        coefs = (E + E0,
                 self.get_xc_difference('XC-0-1.0') - E0,
                 self.get_xc_difference('XC-1-1.0') - E0,
                 self.get_xc_difference('XC-2-1.0') - E0)
        self.text('BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs)
        return np.array(coefs)

    def get_electronic_temperature(self):
        return self.occupations.kT * Hartree

    def get_electrostatic_corrections(self):
        """Calculate PAW correction to average electrostatic potential."""
        dEH_a = np.zeros(len(self.atoms))
        for a, D_sp in self.density.D_asp.items():
            setup = self.wfs.setups[a]
            dEH_a[a] = setup.dEH0 + np.dot(setup.dEH_p, D_sp.sum(0))
        self.wfs.gd.comm.sum(dEH_a)
        return dEH_a * Hartree * Bohr**3

    def get_grid_spacings(self):
        return Bohr * self.wfs.gd.h_c

    def read_wave_functions(self, mode='gpw'):
        """Read wave functions one by one from seperate files"""

        from gpaw.io import read_wave_function
        for u, kpt in enumerate(self.wfs.kpt_u):
            #kpt = self.kpt_u[u]
            kpt.psit_nG = self.gd.empty(self.wfs.nbands, self.wfs.dtype)
            # Read band by band to save memory
            s = kpt.s
            k = kpt.k
            for n, psit_G in enumerate(kpt.psit_nG):
                psit_G[:] = read_wave_function(self.gd, s, k, n, mode)
                
