import Numeric as num

import gridpaw.utilities.mpi as mpi


def new_communicator(ranks):
    if len(ranks) == 1:
        return mpi.serial_comm
    elif len(ranks) == mpi.size:
        return mpi.world
    else:
        return mpi.world.new_communicator(num.array(ranks))

def distribute_kpoints_and_spins(nspins, nkpts):
    ntot = nspins * nkpts
    size = mpi.size
    rank = mpi.rank

    for gcd in range(min(ntot, size), 0, -1):
        if ntot % gcd == 0 and size % gcd == 0:
            break

    ndomains = size / gcd

    r0 = (rank // ndomains) * ndomains
    ranks = range(r0, r0 + ndomains)
    domain_comm = new_communicator(ranks)

    r0 = rank % ndomains
    ranks = range(r0, r0 + size, ndomains)
    kpt_comm = new_communicator(ranks)

    return domain_comm, kpt_comm
