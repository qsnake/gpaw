import Numeric as num

import gridpaw.utilities.mpi as mpi


def distribute_kpoints_and_spins(nspins, ibzk_kc, weights_k):

    # Distribute the spins and kpoints among processors a new comm
    # will be defined for each process.

    p = mpi.size
    if p == 1: 
        return range(nspins), ibzk_kc, weights_k, mpi.world, mpi.world

    nkpts = len(ibzk_kc)
    tot = nspins*nkpts

    # find greatest common divisor of the number of
    # processors and the spins*kpoints
    # this will be the number of domain_comms 
    for i in range(1,min(p,tot)+1):
        if (p%i==0) and (tot%i==0):
            gcd = i

    # two new comms are now define:
    #   domain_comm : a subset of the k-points for a complete domain
    #   kpt_comm    : all kpoints for a part of the domain.

    # number of domain comms
    number_domain_comms = gcd

    # number of processors per domain_comm:
    nproc_per_group = p/number_domain_comms

    # find group members for domain
    group_members = []
    for igroup in range(number_domain_comms):
        members = [n+igroup*nproc_per_group for n in range(nproc_per_group)]
        group_members.append(members)
        if mpi.rank in members:
            mygroup = igroup

    my_domain_comm = mygroup

    if nproc_per_group>1: 
        domain_comm = mpi.world.new_communicator(num.array(group_members[mygroup]))
    else:
        domain_comm = mpi.serial_comm

    # should spin be distributed
    if (gcd%nspins) == 0 and (nspins>1):
        if num.remainder(my_domain_comm,2)==0:
            myspins = [0]
        else:
            myspins = [1]
    else:
        myspins = range(nspins)

    # now define kpt_word

    # number of different kpt_comms is equal to the
    # number of processors in a domain_comm
    number_kpt_comms = nproc_per_group

    # number of processors for each kpt_comm
    nproc_per_kpt_comm = p/number_kpt_comms

    # find group members for this kpt_comm
    group_members = []
    for igroup in range(number_kpt_comms):
        members = [n*number_kpt_comms + igroup for n in range(nproc_per_kpt_comm)]
        group_members.append(members)
        if mpi.rank in members:
            mygroup = igroup


    if nproc_per_kpt_comm > 1: 
        kpt_comm = mpi.world.new_communicator(num.array(group_members[mygroup],
                                                        num.Int))
    else:
        kpt_comm = mpi.serial_comm

    # assign kpoints to each domain_comm
    # example with 4 domains, 4 k-points and two spins

    #      0           1            2            3
    # *-------*    +-------+    +-------+    +-------+
    # * k1,k2 *    + k2,k3 +    + k1,k2 +    + k2,k3 +
    # * s = 0 *    + s = 0 +    + s = 1 +    + s = 1 +
    # * ------+    +-------+    +-------+    +-------+


    # number of kpoints per domain_comm
    kpoints_per_domain_comm = (len(ibzk_kc)*nspins)/(len(myspins)*number_domain_comms)

    # index into k-points
    kpt_index = (my_domain_comm*len(myspins))/nspins

    n = kpt_index*kpoints_per_domain_comm
    m = (kpt_index+1)*kpoints_per_domain_comm
    my_ibzk_kc = ibzk_kc[n:m]
    my_weights_k = weights_k[n:m]

    return myspins, my_ibzk_kc, my_weights_k, domain_comm, kpt_comm


