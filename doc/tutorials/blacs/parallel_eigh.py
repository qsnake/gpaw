import numpy as np
from gpaw.mpi import world, MASTER
from gpaw.blacs import BlacsGrid
from gpaw.blacs import Redistributor


def parallel_eigh(matrixfile, blacsgrid=(4, 2), blocksize=64):
    """Diagonalize matrix in parallel"""
    assert np.prod(blacsgrid) == world.size
    grid = BlacsGrid(world, *blacsgrid)

    if world.rank == MASTER:
        H_MM = np.load(matrixfile)
        assert H_MM.ndim == 2
        assert H_MM.shape[0] == H_MM.shape[1]
        NM = len(H_MM)
    else:
        H_MM = None
        NM = 0
    NM = world.sum(NM)

    # descriptor for the individual blocks
    block_desc = grid.new_descriptor(NM, NM, blocksize, blocksize)

    # descriptor for global array on MASTER
    local_desc = grid.new_descriptor(NM, NM, NM, NM)

    # The local version of the matrix
    H_mm = block_desc.empty()

    # Distribute global array to smaller blocks
    redistributor = Redistributor(world, local_desc, block_desc)
    redistributor.redistribute(H_MM, H_mm)

    # Allocate arrays for eigenvalues and -vectors
    eps_M = np.empty(NM)
    C_mm = block_desc.empty()
    block_desc.diagonalize_ex(H_mm, C_mm, eps_M)

    # Collect eigenvectors on MASTER
    C_MM = local_desc.empty()
    redistributor2 = Redistributor(world, block_desc, local_desc)
    redistributor2.redistribute(C_mm, C_MM)

    # Return eigenvalues and -vectors on Master
    if world.rank == MASTER:
        return eps_M, C_MM
    else:
        return None, None


if __name__ == '__main__':
    # Test script which should be run on 4 CPUs
    
    if world.rank == MASTER:
        a = np.diag(range(1,51)).astype(float)
        a.dump('H_50x50.pckl')
        eps, U = np.linalg.eigh(a)
        print eps

    eps, U = parallel_eigh(matrixfile='H_50x50.pckl',
                           blacsgrid=(2, 1), blocksize=6)
    if world.rank == MASTER:
        print eps
        print U
