# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""K-point/spin combination-descriptors

This module contains classes for defining combinations of two indices:

* Index k for irreducible kpoints in the 1st Brillouin zone.
* Index s for spin up/down if spin-polarized (otherwise ignored).

"""

import numpy as np
from ase.units import Bohr
from ase.dft.kpoints import get_monkhorst_shape

from gpaw.symmetry import Symmetry
from gpaw.kpoint import KPoint

class KPointDescriptor:
    """Descriptor-class for k-points."""

    def __init__(self, bzk_kc, nspins):
        """Construct descriptor object for kpoint/spin combinations (ks-pair).

        Parameters
        ----------
        bzk_kc: ndarray
            Coordinates of the k-points in the Brillouin zone in units of the
            reciprocal lattice vectors.
        nspins: int
            Number of spins.

        Attributes
        ============  ======================================================
        ``nspins``    Number of spins.
        ``nibzkpts``  Number of irreducible kpoints in 1st Brillouin zone.
        ``nks``       Number of k-point/spin combinations in total.
        ``mynks``     Number of k-point/spin combinations on this CPU.
        ``gamma``     Boolean indicator for gamma point calculation.
        ``comm``      MPI-communicator for kpoint distribution.
        ============  ======================================================
        
        """

        self.bzk_kc = bzk_kc
        self.nspins = nspins
        self.nbzkpts = len(bzk_kc)
        
        # Gamma-point calculation
        self.gamma = self.nbzkpts == 1 and not bzk_kc[0].any()

        self.symmetry = None
        self.comm = None
        self.ibzk_kc = None
        self.weight_k = None
        self.nibzkpts = None

    def __len__(self):
        """Return number of k-point/spin combinations of local CPU."""
        
        return self.mynks

    def set_symmetry(self, atoms, setups, usesymm):
        """Create symmetry object and construct irreducible Brillouin zone.

        Parameters
        ----------
        atoms: Atoms object
            Defines atom positions and types and also unit cell and
            boundary conditions.
        setups: instance of class Setups
            PAW setups for the atoms.
        usesymm: bool
            Symmetry flag.

        """

        if (~atoms.pbc & self.bzk_kc.any(0)).any():
            raise ValueError('K-points can only be used with PBCs!')

        if self.gamma:
            self.symmetry = None
            self.weight_k = np.array([1.0])
            self.ibzk_kc = np.zeros((1, 3))

        elif usesymm is None:
            # Point group and time-reversal symmetry neglected
            nkpts = len(self.bzk_kc)
            self.symmetry = None
            self.weight_k = np.ones(nkpts) / nkpts
            self.ibzk_kc = self.bzk_kc.copy()

        else:
            # Round off
            magmom_a = atoms.get_initial_magnetic_moments().round(decimals=3)
            id_a = zip(magmom_a, setups.id_a)

            # Construct a Symmetry instance containing the identity operation
            # only
            self.symmetry = Symmetry(id_a, atoms.get_cell() / Bohr,
                                     atoms.get_pbc())

            if usesymm:
                # Find symmetry operations of atoms
                self.symmetry.analyze(atoms.get_scaled_positions())
            else:
                self.symmetry.prune_symmetries(atoms.get_scaled_positions())

            # Reduce the set of k-points and add inversion if not already
            # detected
            self.ibzk_kc, self.weight_k = self.symmetry.reduce(self.bzk_kc)

            if usesymm:
                setups.set_symmetry(self.symmetry)
            else:
                self.symmetry = None

        self.nibzkpts = len(self.ibzk_kc)
        
    def set_communicator(self, comm):
        """Set k-point communicator."""

        self.comm = comm
        # Total number of k-point/spin combinations:
        nks = self.nibzkpts * self.nspins

        # Ranks < self.rank0 have mynks0 k-point/spin combinations and
        # ranks >= self.rank0 have mynks+1 k-point/spin combinations.
        self.mynks0, x = divmod(nks, comm.size)
        self.rank0 = comm.size - x
        self.ks0 = comm.rank * self.mynks0
        # My number of k-point/spin combinations
        self.mynks = self.mynks0  
        if comm.rank >= self.rank0:
            self.ks0 += comm.rank - self.rank0
            self.mynks += 1

        if self.nspins == 2 and comm.size == 1:
            # Avoid duplicating k-points in local list of k-points.
            self.ibzk_qc = self.ibzk_kc.copy()
        else:
            self.ibzk_qc = np.vstack((self.ibzk_kc,
                                      self.ibzk_kc))[self.ks0:self.ks0 +
                                                     self.mynks]

    def create_k_points(self, gd):
        """Return a list of KPoints."""
 
        sdisp_cd = gd.sdisp_cd

        kpt_u = []
        for ks in range(self.ks0, self.ks0 + self.mynks):
            s, k = divmod(ks, self.nibzkpts)
            q = (ks - self.ks0) % self.nibzkpts
            weight = self.weight_k[k] * 2 / self.nspins
            if self.gamma:
                phase_cd = np.ones((3, 2), complex)
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * self.ibzk_kc[k, :, np.newaxis])
            kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        return kpt_u

    def find_k_plus_q(self, q_c):
        """Find the indices of k+q for all kpoints in the Brillouin zone.

        In case that k+q is outside the BZ, the k-point inside the BZ
        corresponding to k+q is given.
        
        Parameters
        ----------
        q_c: ndarray
            Scaled coordinates for the q-vector in units of the reciprocal
            lattice vectors.

        """

        kplusq_k = []
        
        nkptxyz = get_monkhorst_shape(self.bzk_kc)
        
        dk = 1. / nkptxyz 
        kmax = (nkptxyz - 1) * dk / 2.
        N = np.zeros(3, dtype=int)

        for k, k_c in enumerate(self.bzk_kc):
            
            kplusq_c = k_c + q_c
            
            for dim in range(3):
                if kplusq_c[dim] > 0.5:
                    kplusq_c[dim] -= 1.
                elif kplusq_c[dim] < -0.5:
                    kplusq_c[dim] += 1.
    
                N[dim] = int(np.round((kplusq_c[dim] + kmax[dim])/dk[dim]))
    
            kplusq_k.append(N[2] + N[1] * nkptxyz[2] + N[0] * nkptxyz[2] * nkptxyz[1])

            # Check the k+q vector index
            k_c = self.bzk_kc[kplusq_k[k]]

            assert abs(kplusq_c - k_c).sum() < 1e-8, 'k+q index not correct!'
    
        return kplusq_k

    def get_rank_and_index(self, k, s):
        """Find rank and local index."""
        
        ks = k + self.nibzkpts * s
        if ks < self.mynks * self.rank0:
            rank, u = divmod(ks, self.mynks)
        else:
            rank, u = divmod(ks - self.mynks * self.rank0, self.mynks + 1)
            rank += self.rank0
            
        return rank, u
   
    def get_slice(self, rank=None):
        """Return the slice of global ks-pairs which belong to a given rank."""
        
        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)

        ks0 = rank * self.mynks0
        mynks = self.mynks0
        
        if rank >= self.rank0:
            ks0 += rank - self.rank0
            mynks += 1
            
        uslice = slice(ks0, ks0 + mynks)
        return uslice

    def get_ks_pair_indices(self, rank=None):
        """Return the global ks-pair indices which belong to a given rank."""
        
        uslice = self.get_slice(rank)
        return np.arange(*uslice.indices(self.nks))

    def get_ks_pair_ranks(self):
        """Return array of ranks as a function of global ks-pair indices."""
        
        rank_u = np.empty(self.nks, dtype=int)
        for rank in range(self.comm.size):
            uslice = self.get_slice(rank)
            rank_u[uslice] = rank
        assert (rank_u >= 0).all() and (rank_u < self.comm.size).all()
        return rank_u

    def who_has(self, u):
        """Convert global index to rank information and local index."""
        
        rank, myu = divmod(u, self.mynks)
        return rank, myu

    def global_index(self, myu, rank=None):
        """Convert rank information and local index to global index."""
        
        if rank is None:
            rank = self.comm.rank
        u = rank * self.mynks + myu
        return u

    def what_is(self, u):
        """Convert global index to corresponding kpoint/spin combination."""
        
        s, k = divmod(u, self.nibzkpts)
        return s, k

    def where_is(self, s, k):
        """Convert kpoint/spin combination to the global index thereof."""
        
        u = k + self.nibzkpts * s
        return u

    def who_has_and_where_is(self, s, k):
        """Redundant function is redundant.""" #XXX most often used though!
        
        u = self.where_is(s, k)
        rank, myu = self.who_has(u)
        return rank, myu

    #def get_size_of_global_array(self):
    #    return (self.nspins*self.nibzkpts,)
    #
    #def ...



class KPointDescriptorOld:
    """Descriptor-class for ordered lists of kpoint/spin combinations

    TODO
    """ #XXX

    def __init__(self, nspins, nibzkpts, comm=None, gamma=True, dtype=float):
        """Construct descriptor object for kpoint/spin combinations (ks-pair).

        Parameters:

        nspins: int
            Number of spins.
        nibzkpts: int
            Number of irreducible kpoints in 1st Brillouin zone.
        comm: MPI-communicator
            Communicator for kpoint-groups.
        gamma: bool
            More to follow.
        dtype: NumPy dtype
            More to follow.

        Note that if comm.size is greater than the number of spins, then
        the kpoints cannot all be located at the gamma point and therefor
        the gamma boolean loses its significance.

        Attributes:

        ============  ======================================================
        ``nspins``    Number of spins.
        ``nibzkpts``  Number of irreducible kpoints in 1st Brillouin zone.
        ``nks``       Number of k-point/spin combinations in total.
        ``mynks``     Number of k-point/spin combinations on this CPU.
        ``gamma``     Boolean indicator for gamma point calculation.
        ``dtype``     Data type appropriate for wave functions.
        ``beg``       Beginning of ks-pair indices in group (inclusive).
        ``end``       End of ks-pair indices in group (exclusive).
        ``step``      Stride for ks-pair indices between ``beg`` and ``end``.
        ``comm``      MPI-communicator for kpoint distribution.
        ============  ======================================================
        """
        
        if comm is None:
            comm = mpi.serial_comm
        self.comm = comm
        self.rank = self.comm.rank

        self.nspins = nspins
        self.nibzkpts = nibzkpts
        self.nks = self.nibzkpts * self.nspins

        # XXX Check from distribute_cpus in mpi/__init__.py line 239 rev. 4187
        if self.nks % self.comm.size != 0:
            raise RuntimeError('Cannot distribute %d k-point/spin ' \
                               'combinations to %d processors' % \
                               (self.nks, self.comm.size))

        self.mynks = self.nks // self.comm.size

        # TODO Move code from PAW.initialize in paw.py lines 319-328 rev. 4187
        self.gamma = gamma
        self.dtype = dtype

        uslice = self.get_slice()
        self.beg, self.end, self.step = uslice.indices(self.nks)

    #XXX u is global kpoint index

    def __len__(self):
        return self.mynks

    def get_slice(self, rank=None):
        """Return the slice of global ks-pairs which belong to a given rank."""
        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        ks0 = rank * self.mynks
        uslice = slice(ks0, ks0 + self.mynks)
        return uslice

    def get_ks_pair_indices(self, rank=None):
        """Return the global ks-pair indices which belong to a given rank."""
        uslice = self.get_slice(rank)
        return np.arange(*uslice.indices(self.nks))

    def get_ks_pair_ranks(self):
        """Return array of ranks as a function of global ks-pair indices."""
        rank_u = np.empty(self.nks, dtype=int)
        for rank in range(self.comm.size):
            uslice = self.get_slice(rank)
            rank_u[uslice] = rank
        assert (rank_u >= 0).all() and (rank_u < self.comm.size).all()
        return rank_u

    def who_has(self, u):
        """Convert global index to rank information and local index."""
        rank, myu = divmod(u, self.mynks)
        return rank, myu

    def global_index(self, myu, rank=None):
        """Convert rank information and local index to global index."""
        if rank is None:
            rank = self.comm.rank
        u = rank * self.mynks + myu
        return u

    def what_is(self, u):
        """Convert global index to corresponding kpoint/spin combination."""
        s, k = divmod(u, self.nibzkpts)
        return s, k

    def where_is(self, s, k):
        """Convert kpoint/spin combination to the global index thereof."""
        u = k + self.nibzkpts * s
        return u

    def who_has_and_where_is(self, s, k):
        """Redundant function is redundant.""" #XXX most often used though!
        u = self.where_is(s, k)
        rank, myu = self.who_has(u)
        return rank, myu

    #def get_size_of_global_array(self):
    #    return (self.nspins*self.nibzkpts,)
    #
    #def ...


