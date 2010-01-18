.. _performance:

======================
Maximizing performance
======================

If you haven't done so already, this is a good point to read up on the
GPAW parallelization strategies and the BG/P architecture. Band parallelization
will be needed to scale your calculation to large number of cores. The BG/P
systems at Argonne National Laboratory uses Cobalt for scheduling and
it will be referred to frequently below. Other schedulers should have
similar functionality.

Physical Topology <-> Compute Topology
========================================
The BG/P partition dimensions (Px, Py, Pz, T) for surveyor and intrepid at the
Argonne Leadership Computing Facility are `available here 
<https://wiki.alcf.anl.gov/index.php/Running#What_are_the_sizes_and_dimensions_of_the_partitions_on_the_system.3F>`_,
where T represents the number of cores utilized on each node (not whether 
you have a torus network). These are number of cores per node which run MPI
tasks and is specified by Cobalt flag::

  --mode={smp,dual,vn}

where smp, dual and vn are for 1, 2, and 4, MPI tasks respectively. Note that
there are 4 cores per node and 2 GB per node on BG/P.

It is essential to think of the BG/P network as a 4-dimensional object with
3 spatial dimentions and a T-dimension. For optimum efficiency, we
must simultaneously satisfy at least two distinct communications patterns
due to different parts of the DFT calculation: a) H*Psi products 
b) diagonalization. This can only be accomplished with a 4-dimensional
(or higher) network.

The domain decomposition can be specified on the 
``gpaw-python`` command line with ``--domain-decomposition=Nx,Ny,Nz``
and band parallelization with ``--state-parallelization=B``. Here *N* bands
are divided into *B* groups. It was empirically determined that you need to
have *N/B > 256*. It will be necessary to have the combined band-domain
decomposition match the partition dimension exactly, i.e. ::

  {Nx, Ny, Nz, B} = {Px, Py, Pz, T},
  {Nx, Ny, Nz, B} = {T, Px, Py, Pz},
  {Nx, Ny, Nz, B} = {Px, T, Py, Pz}, 
  or another permutation.

This can be accomplised with the help of ``tools/mapfile.py.`` You will
want to use ``band`` mode to generate a BG/P mapfile for a  DFT calculation.
Since there is no diagonalization in the rTDDFT method, one can use 
``domain`` mode as a 3-dimensional network  is sufficient to satisfy the
communiation pattern of the H*Psi products. You will then need to specify the
mapfile via Cobalt::

  --env=BG_MAPPING=<mapfile>

Simultaneous parallelization on k-points, spins, bands and domains
=====================================================================
This is currently in the works. However, even when this is finished we will
need to figure out how to map this correctly to the torus. What is really
needed is a 5-dimensional network.

Important DCMF environment variables
===============================================
`DCMF <http://dcmf.anl-external.org/wiki/index.php/Main_Page>`_  is one
of the lower layers in the BG/P implementation of MPI software stack. 


To understand th DCMF environment variables in greater detail, please read the
appropriate sections of the  IBM System Blue Gene Solution:  
`Blue Gene/P Application Development <http://www.redbooks.ibm.com/abstracts/sg247287.html?Open>`_ 

DCMF_EAGER
============
The computation of the hamiltonian and overlap matrix elements, as well as
the computation of the new wavefunctions, is accomplished by a hand-coded 
parallel matrix-multiply ``hs_operators.py`` employing a 1D ring algorithm.
Please refer to the details of :ref:`band parallelization <band_parallelization>`.

Communication and computation is overlapped to the extent allowed by the
hardware by using non-block sends (Isend)and receives (Irecv). It will be
necessary to select appropriate values for the number of blocks ``nblocks``::

  from gpaw.hs_operators import Operator
  Operator.nblocks = K
  Operator.async = True (default)

where the ``B`` groups of bands are further divided into ``K`` blocks for
efficiency and to save on memory. The value of ``K`` should be selected
so that 4 - 8 MB of wavefunctions are sent/received. It will be also be
necessary to pass to Cobalt::

  --env=DCMF_EAGER=8388608

which corresponds to the size of this block of wavefunctions. Note that the
number is specified in bytes and not megabytes.

For larger blocks of wavefunctions, it will be necessary to increase
DCMF_RECFIFO as well. This will depend on whether you are using smp, dual
or vn mode. 

DCMF_REUSE_STORAGE
====================
If you receive receive allocation error on MPI_Allreduce, please add the following
environment variables::

  --env=DCMF_REDUCE_REUSE_STORAGE=N:DCMF_ALLREDUCE_REUSE_STORAGE=N:DCMF_REDUCE=RECT
