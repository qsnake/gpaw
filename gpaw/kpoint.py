# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class and the derived ``GlobalKPoint``."""

import numpy as np

from gpaw.operators import Gradient
from gpaw.utilities.blas import axpy


class KPoint:
    """Class for a single k-point.

    The KPoint class takes care of all wave functions for a
    certain k-point and a certain spin.

    XXX This needs to be updated.

    Attributes:

    phase_cd: complex ndarray
        Bloch phase-factors for translations - axis c=0,1,2
        and direction d=0,1.
    eps_n: float ndarray
        Eigenvalues.
    f_n: float ndarray
        Occupation numbers.
    psit_nG: ndarray
        Wave functions.
    nbands: int
        Number of bands.

    Parallel stuff:

    comm: Communicator object
        MPI-communicator for domain.
    root: int
        Rank of the CPU that does the matrix diagonalization of
        H_nn and the Cholesky decomposition of S_nn.
    """
    
    def __init__(self, weight, s, k, q, phase_cd):
        """Construct k-point object.

        Parameters:

        gd: GridDescriptor object
            Descriptor for wave-function grid.
        weight: float
            Weight of this k-point.
        s: int
            Spin index: up or down (0 or 1).
        k: int
            k-point index.
        q: int
            local k-point index.
        k_c: float-ndarray of shape (3,)
            scaled **k**-point vector (coordinates scaled to
            [-0.5:0.5] interval).
        dtype: type object
            Data type of wave functions (float or complex).
        timer: Timer object
            Optional.

        Note that s and k are global spin/k-point indices,
        whereas u is a local spin/k-point pair index for this
        processor.  So if we have `S` spins and `K` k-points, and
        the spins/k-points are parallelized over `P` processors
        (kpt_comm), then we have this equation relating s,
        k and u::

           rSK
           --- + u = sK + k,
            P

        where `r` is the processor rank within kpt_comm.  The
        total number of spin/k-point pairs, `SK`, is always a
        multiple of the number of processors, `P`.
        """

        self.weight = weight
        self.s = s  # spin index
        self.k = k  # k-point index
        self.q = q  # local k-point index
        self.phase_cd = phase_cd
        
        self.eps_n = None
        self.f_n = None
        self.P_ani = None

        # Only one of these two will be used:
        self.psit_nG = None  # wave functions on 3D grid
        self.C_nM = None     # LCAO coefficients for wave functions XXX

        self.rho_MM = None
        
        self.P_aMi = None
        self.S_MM = None
        self.T_MM = None

    def add_to_density(self, nt_G, use_lcao, basis_functions):
        raise DeprecationWarning
        """Add contribution to pseudo electron-density."""
        self.add_to_density_with_occupation(nt_G, use_lcao, self.f_n,
                                            basis_functions)
        
    def add_to_density_with_occupation(self, nt_G, use_lcao, f_n,
                                       basis_functions):
        raise DeprecationWarning
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        print 'nnnnnnnnnnnnnnoooooooooooo'
        if use_lcao:
            C_nM = self.C_nM
            rho_MM = np.dot(C_nM.conj().T * f_n, C_nM)
            basis_functions.construct_density(rho_MM, nt_G)
        else:
            if self.dtype == float:
                for f, psit_G in zip(f_n, self.psit_nG):
                    axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
            else:
                for f, psit_G in zip(f_n, self.psit_nG):
                    nt_G += f * (psit_G * np.conjugate(psit_G)).real
        
    def add_to_kinetic_density(self, taut_G):
        raise DeprecationWarning
        """Add contribution to pseudo kinetic energy density."""
        ddr = [Gradient(self.gd, c, dtype=self.dtype).apply for c in range(3)]
        d_G = self.gd.empty(dtype=self.dtype)
        for f, psit_G in zip(self.f_n, self.psit_nG):
            for c in range(3):
                if self.dtype == float:
                    ddr[c](psit_G,d_G)
                    axpy(0.5*f, d_G**2, taut_G) #taut_G += 0.5*f * d_G**2
                else:
                    ddr[c](psit_G,d_G,self.phase_cd)
                    taut_G += 0.5* f * (d_G * np.conjugate(d_G)).real

    def create_atomic_orbitals(self, nao, nuclei):
        """Initialize the wave functions from atomic orbitals.

        Create nao atomic orbitals."""
        raise DeprecationWarning
        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)
        self.psit_nG = self.gd.zeros(nao, self.dtype)

        # fill in the atomic orbitals:
        nao0 = 0
        for nucleus in nuclei:
            nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
            nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k)
            nao0 = nao1
        assert nao0 == nao

    def create_random_orbitals(self, nbands):
        """Initialize all the wave functions from random numbers"""
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.allocate(nbands)
        self.psit_nG = self.gd.zeros(nbands, self.dtype)
        self.random_wave_functions(self.psit_nG)                   


class GlobalKPoint(KPoint):

    def update(self, wfs):
        """Distribute requested kpoint data across the kpoint communicator."""

        # Locate rank and index of requested k-point
        nks = len(wfs.ibzk_kc)
        mynu = len(wfs.kpt_u)
        kpt_rank, u = divmod(self.k + nks * self.s, mynu)
        kpt_comm = wfs.kpt_comm

        my_atom_indices = np.argwhere(wfs.rank_a == wfs.gd.comm.rank).ravel()
        mynproj = sum([wfs.setups[a].ni for a in my_atom_indices])
        my_P_ni = np.empty((wfs.mynbands, mynproj), wfs.dtype)

        self.P_ani = {}

        if self.phase_cd is None:
            self.phase_cd = np.empty((3,2), wfs.dtype)

        if self.psit_nG is None:
            self.psit_nG = wfs.gd.empty(wfs.mynbands, wfs.dtype)

        reqs = []

        # Do I have the requested kpoint?
        if kpt_comm.rank == kpt_rank:
            self.phase_cd[:] = wfs.kpt_u[u].phase_cd
            self.psit_nG[:] = wfs.kpt_u[u].psit_nG

            # Compress entries in requested P_ani dict into my_P_ni ndarray
            i = 0
            for a,P_ni in wfs.kpt_u[u].P_ani.items():
                ni = wfs.setups[a].ni
                my_P_ni[:,i:i+ni] = P_ni
                i += ni

            assert (my_atom_indices == wfs.kpt_u[u].P_ani.keys()).all()

            # Send phase_cd, psit_nG and my_P_ni to kpoint slaves
            for rank in range(kpt_comm.size):
                if rank != kpt_rank:
                    reqs.append(kpt_comm.send(self.phase_cd, rank, 256, False))
                    reqs.append(kpt_comm.send(self.psit_nG, rank, 257, False))
                    reqs.append(kpt_comm.send(my_P_ni, rank, 258, False))
        else:
            # Receive phase_cd, psit_nG and my_P_ni from kpoint master
            reqs.append(kpt_comm.receive(self.phase_cd, kpt_rank, 256, False))
            reqs.append(kpt_comm.receive(self.psit_nG, kpt_rank, 257, False))
            reqs.append(kpt_comm.receive(my_P_ni, kpt_rank, 258, False))

        for request in reqs:
            kpt_comm.wait(request)

        # Decompress my_P_ni ndarray into entries in my P_ani dict
        i = 0
        for a in my_atom_indices:
            ni = wfs.setups[a].ni
            self.P_ani[a] = my_P_ni[:,i:i+ni] #copy?
            i += ni

