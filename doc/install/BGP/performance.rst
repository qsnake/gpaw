.. _performance:

======================
Maximizing performance
======================

If you haven't done so already, this is a good point to read up on the
GPAW parallelization strategies and the BG/P architecture. The BG/P
systems at Argonne National Laboratory uses Cobalt for scheduling and
it will be referred to frequently below. Other schedulers should have
similar functionality.

Physical Topology <-> Compute Topology
========================================
The domain decomposition (physical) can be specified on the 
``gpaw-python`` command line with ``--domain-decomposition=Nx,Ny,Nz``
flag. It is desirable to have the BG/P partition (compute) topology
*exactly* match the domain decomposition to minimize the number of node hops
during point-to-point communication. The BG/P partition dimensions (Px,
Py, Pz) for surveyor and intrepid at Argonne Leadership Computing
Facility are available here.
https://wiki.alcf.anl.gov/index.php/Running#What_are_the_sizes_and_dimensions_of_the_partitions_on_the_system.3F

There are several things to note:

* For small BG/P partition sizes (less than 1028 nodes), a mismatch
  between  the physical and compute topology may only slightly hinder
  performance. It will certainly become important above 4096 nodes.
* If *pure* domain decomposition and *smp* mode are used, it is
  necessary to use MAPPING=ZYXT. This is the order in which the ranks
  are assigned by GPAW. Here T=0 (torus dimension) since we are in smp
  mode. In general, you want Nx=Px, Ny=Py, Nz=Pz.
* If you use fewer nodes that are available in your partition, then
  adjacent subdomains will not end up on adjacent compute nodes. It is
  recommend to either increase your grid points to allow you to
  use all the nodes or use a *MAPFILE*.

It is very difficult to use all the cores on the nodes using only
domain-decomposition. Here one can think of the T-dimension as a 4th
dimension to be folded into any of the other physical
dimensions (X,Y,Z) or to exploit it for another layer of
parallezation, e.g. spin, band or k-point parallelization

Band parallelization
====================
Band parallelization is enabled on the
``gpaw-python`` command line with ``--state-parallelization=B``
flag. Here *N* bands are divided into *B* groups. Efficient band parallelization
is obtained by having the *N/B* bands end up "together" on BG/P partition. This is
accomplished through the use of a *MAPFILE* which can be easily
generated with ``tools/mapfile.py`` and then passed to Cobalt with
``-env mapfile=<MAPFILE>``

The syntax for the use of ``mapfile.py`` is::

  python mapfile.py <number of nodes> Nx, Ny, Nz

Let's take a concreate example. Suppose you want to run on 128 nodes,
which has partition dimensions Px=Py=4 and Pz=8, the physical problem has N
= 2000  and it the grid points can be easily decomposed into
Nx=Ny=Nz=4.  You also have enough memory to run in *vn* mode, then we
generate the **MAPFILE** with::

  python mapfile.py 128 4,4,4

Now let's do some simple math, in *vn* mode there
are Px*Py*Pz*4 = 512 ranks. We are only using Nx\*Ny\*Nz = 64 for domain
decompsition. Where do the rest go? The answer is band
parallelization. We take our N bands and divide them into this many groups::

 B = Px*Py*Pz*4/(Nz*Ny*Nz) = 8

If you problem cannot run in *vn* mode, change the ``ppn`` parameter in
``mapfile.py`` to reflect the number of processors per node that you
can use, and repeat this analysis to determine the correct value that
should be passed to ``--state-parallelization=B``.

Spin Polaralization
====================
In progress...

