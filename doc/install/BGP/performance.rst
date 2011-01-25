.. _bgp_performance:

==============================
Maximizing performance on BG/P
==============================

Begin by reading up on the GPAW parallelization strategies 
(:ref:`parallel_runs`) and the `BG/P
architecture <https://wiki.alcf.anl.gov/index.php/References>`_.  In
particular,  :ref:`band_parallelization` will be needed to scale your
calculation to large number of cores. The BG/P systems at the `Argonne
Leadership Computing Facility <http://www.alcf.anl.gov>`_
uses Cobalt for scheduling and it will be referred to frequently below. Other schedulers should have similar functionality.

There are four key aspects that require careful considerations:

1) Choosing a parallelization strategy.

#) Selecting the correct partition size (number of nodes) and mapping.

#) Choosing an appropriate value of ``nblocks``. The constraints on ``nbands`` are also summarized.

#) Setting the appropriate DCMF environmental variables.

In the sections that follow, we aim to cultivate an understanding of
how to choose these parameters.

Parallelization Strategy
====================================

Parallelization options are specified at the ``gpaw-python`` command
line.  Domain decomposition  with ``--domain-decomposition=Nx,Ny,Nz``
and band parallelization with ``--state-parallelization=B``.
Additionally, the ``parallel`` keyword is also available.

The smallest calculation that can benefit from band/state
parallelization is *nbands = 1000*. If you are using fewer bands, you
are possibly *not* in need of a leadership class computer facility. 
Note that only :ref:`RMM-DIIS` eigensolver is compatible with band
parallelization. Furthermore, the RMM-DIIS eigensolver requires 
some unoccupied bands in order to converge properly. Recommend range is::

  nbands = valence electrons/spin*[1.0 - 1.2], where
  spin = 1 for spinpol = True
  spin = 2 for spinpol = False
  
It was empirically determined that you need to have *nbands/B
> 256*  for reasonable performance. It is also possible use smaller groups, 
*N/B < 256*, but this may require large domains. It is *required* that
*nbands/B* be integer-divisible. The best values for B =2, 4, 8, 16,
32, 64, and 128.

Obviously, the number of total cores must equal::
  
   Nx*Ny*Nz*B

The parallelization strategy will require careful consideration of the
partition size and mapping. Obviously, also memory!

Partition size and Mapping 
========================================
The BG/P partition dimensions (Px, Py, Pz, T) for Surveyor and Intrepid at the
Argonne Leadership Computing Facility are `available here 
<https://wiki.alcf.anl.gov/index.php/Running#What_are_the_sizes_and_dimensions_of_the_partitions_on_the_system.3F>`_,
where T represents the number of MPI tasks (not whether a
torus network is available). The number of cores per node which
execute MPI tasks is specified by the Cobalt flag::

  --mode={smp,dual,vn}

Hence, the possible values of T are::

  T = 1 for smp
  T = 2 for dual
  T = 4 for vn

Note that there are 4 cores per node and 2 GB per node on BG/P. As GPAW is
presently an MPI-only code, vn mode is preferred since all cores will
be perform computational work.

It is essential to think of the BG/P network as a 4-dimensional object with
3 spatial dimentions and a T-dimension. For optimum scalability it
would seem necessary to maximize the locality of two distinct
communications patterns arising in the canonical O(N^3) DFT algorithm: 
a) H*Psi products  b) parallel matrix multiplies. However, it turns
out this is *not*  necessary. The mesh network can handle small messages
rather efficiently such that the time to send a small message to a
nearest-neighbor node versus a node half-way across the machine is
comparable. Hence, it is only necessary to optimize the mapping for
the communication arising from the parallel matrix multiply which is
a simple 1D systolic communication pattern.

Here we show the examples of different mappings on a 512-node BG/P
partition. Band groups are colored coded. *(Left)* Inefficient mapping
for four groups of bands (B = 4). This mapping leads to contention on
network links in the z-direction. *(Right)*  Efficient mapping for eight
groups of bands (B=8). Correct mapping maximizes scalability and
single-core peak performance. 

|mapping1|		|mapping2|

.. |mapping1| image:: bgp_mapping1.png	
   :width: 40 % 
  
.. |mapping2| image:: bgp_mapping2.png
   :width: 40 %

For the mapping on the *(Right)* above image, there are 
two communication patterns (and hence mappings) that are worth
distinguishing.

|intranode|

.. |intranode| image:: bgp_mapping_intranode.png
   :width: 60 %

The boxes in these images represent a node and the numbers inside
the box repesent the distinct cores in the node (four for BG/P).
Intuitively, the communication pattern of the *(Left)* image should
lead to less network contention than the *(Right)*. However, this is
not the case due to lack of optimization in the intranode
implementation of MPI. The performance of these communications
patterns is presently identical, though this may change in future
version of the BG/P implementation of MPI. 

Mapping is accomplished by the Cobalt flag::
  
   --env=BG_MAPPING=<mapping>

where *<mapping>* can be one of the standard BG/P mappings or
a mapfile.

Lastly, it is important to note that GPAW code orders the MPI tasks as
follows::
  
   Z, Y, X, bands, kpoints, and spins.

B = 2
--------
Simply set the followin submission script setting, noting that the
domain decomposition must match up *exactly* with the partition dimensions::

  mode = dual
  any mapping ending with a T
  {Nz, Ny, Nx} = {Px, Py, Pz}; BG_MAPPING = XYZT
  {Nz, Ny, Nx} = {Pz, Px, Py}; BG_MAPPING = ZXYT
  {Nz, Ny, Nx} = {Py, Pz, Px}; BG_MAPPING = YZXT
  plus other permutations of the right-hand side which end with a T
  

B = 4
--------
Similar to the *B=2* case, but with::

  mode = vn

B = 8 or 16
---------------
It will be necessary to have the combined band-domain decomposition
match the partition dimension exactly, i.e.::

  {Nz, Ny, Nx, B} = {T, Px, Py, Pz}
  {Nz, Ny, Nx, B} = {Px, T, Py, Pz}
  {Nz, Ny, Nx, B} = {Px, Py, T, Pz}
  plus other permutations of the right-hand side which do not end with
  a T

As only mappings with T at the beginning or end are standard mappings
on BG/P, a mapfile must be provided for many of the *B=8 or 16*
case. This can be accomplised with ``tools/mapfile.py.`` You will want
to use ``band``  mode to generate a BG/P mapfile for a  DFT
calculation. Since there is no orthogonalization in the rTDDFT method,
one can use ``domain`` mode to satisfy the communiation pattern of the
H*Psi products.  Remember to specify the mapfile via Cobalt::

  --env=BG_MAPPING=<mapfile>

B = 32, 64, or 128
------------------
For *B=32*, a mapfile can be generated as in the *B=8 or 16* case. But
it is much easier to fold the T-dimension into one of the three
spatial dimensions and use this as the band parallelization
direction. The three-dimensional physical domain can then be flattened into
the two remaining spatial dimensions of the network. The constraints
can be summarized as follows::

  mode = vn
  mapping =  any mapping end with T
  T*[X,Y,Z] = B
  product of remaining two dimensions = Nx*Ny*Nz

Setting the value of nblocks
============================
The documentation below will be partially deprecated in the
future. Use ``buffer_size=2048`` and nblocks will be automatically
computed on the fly in the SVN version 7520 and higher. Refer to 
:ref:`manual_parallel` for more information about the ``buffer_size``
keyword.

The computation of the hamiltonian and overlap matrix elements, as well as
the computation of the new wavefunctions, is accomplished by a hand-coded 
parallel matrix-multiply ``hs_operators.py`` employing a 1D systolic
ring algorithm. 

It will be necessary to select appropriate values for the number of blocks ``nblocks``::

  from gpaw.hs_operators import MatrixOperator
  MatrixOperator.nblocks = K
  MatrixOperator.async = True (default)

where the ``B`` groups of bands are further divided into ``K``
blocks. It is also required that *nbands/B/K* be integer-divisible. 
The value of ``K`` should be chosen so that 2 MB of wavefunctions are
interchanged.  The special cases of B=2, 4 as described
above permit the use blocks of wavefunctions larger than 2 MB to be
interchanged since there is only intranode communication. Larger
blocks of wavefunctions can be interchanged by adjusting the 
Cobalt environment variables: DMCF_RECFIFO.

The size of the wavefunction being interchanged is given by::

  gpts = (Gx, Gy, Gz)
  size of wavefunction block in MB = (Gx/Nx, Gy/Ny, Gz/Nz)*(nbands/B/K)*8/1024^2

There are thus a number of constraints on the value of nbands:

1) ``nbands/B`` must be integer divisible

#) ``nbands/B/K`` must be integer divisible. No longer applicable in SVN verison 7520 and higher.

#) size of wavefunction block ~ 2 MB

#) ``nbands`` must be sufficient largely so that the RMM-DIIS eigensolver converges


Important DCMF environment variables
===============================================
`DCMF <http://dcmf.anl-external.org/wiki/index.php/Main_Page>`_  is one
of the lower layers in the BG/P implementation of MPI software stack. 

To understand th DCMF environment variables in greater detail, please read the
appropriate sections of the  IBM System Blue Gene Solution:  
`Blue Gene/P Application Development <http://www.redbooks.ibm.com/abstracts/sg247287.html?Open>`_ 

DCMF_EAGER and DCMF_RECFIFO
-----------------------------------
Communication and computation is overlapped to the extent allowed by the
hardware by using non-blocking sends (Isend) and receives (Irecv). It will be also be necessary to pass to Cobalt::

  --env=DCMF_EAGER=8388608

which corresponds to the larger size message that can be overlapped
(8 MB). Note that the number is specified in bytes and not
megabytes. This is larger than the target 2 MB size, but we keep this
for historical reasons since it is possible to use larger blocks of
wavefunctions in the case of *smp* or *dual* mode. This is also
equal to the default size of the DCMF_RECFIFO. If the following
warning is obtained,::

  A DMA unit reception FIFO is full.  Automatic recovery occurs
  for this event, but performance might be improved by increasing the FIFO size

the default value of the DCMF_RECFIFO should be increased::

   --env=DCMF_RECFIFO=<size in bytes>

DCMF_REUSE_STORAGE
-------------------------
If you receive allocation error on MPI_Allreduce, please add the following
environment variables::

  --env=DCMF_REDUCE_REUSE_STORAGE=N:DCMF_ALLREDUCE_REUSE_STORAGE=N:DCMF_REDUCE=RECT

It is very likely that your calculation is low on memory. Simply try using more nodes.
