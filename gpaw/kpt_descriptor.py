# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""K-point/spin combination-descriptors

This module contains classes for defining combinations of two indices:

* Index k for irreducible kpoints in the 1st Brillouin zone.
* Index s for spin up/down if spin-polarized (otherwise ignored).
"""

import numpy as np

class KPointDescriptor:
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

    def get_slice(self, kpt_rank=None):
        """Return the slice of global ks-pairs which belong to a given rank."""
        if kpt_rank is None:
            kpt_rank = self.comm.rank
        assert kpt_rank in xrange(self.comm.size)
        ks0 = kpt_rank * self.mynks
        uslice = slice(ks0, ks0 + self.mynks)
        return uslice

    def get_ks_pair_indices(self, kpt_rank=None):
        """Return the global ks-pair indices which belong to a given rank."""
        uslice = self.get_slice(kpt_rank)
        return np.arange(*uslice.indices(self.nks))

    def get_ks_pair_ranks(self):
        """Return array of ranks as a function of global ks-pair indices."""
        rank_u = np.empty(self.nks, dtype=int)
        for kpt_rank in range(self.comm.size):
            uslice = self.get_slice(kpt_rank)
            rank_u[uslice] = kpt_rank
        assert (rank_u >= 0).all() and (rank_u < self.comm.size).all()
        return rank_u

    def who_has(self, u):
        """Convert global index to rank information and local index."""
        kpt_rank, myu = divmod(u, self.mynks)
        return kpt_rank, myu

    def global_index(self, myu, kpt_rank=None):
        """Convert rank information and local index to global index."""
        if kpt_rank is None:
            kpt_rank = self.comm.rank
        u = kpt_rank * self.mynks + myu
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
        kpt_rank, myu = self.who_has(u)
        return kpt_rank, myu

    #def get_size_of_global_array(self):
    #    return (self.nspins*self.nibzkpts,)
    #
    #def ...

