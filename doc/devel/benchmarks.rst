.. _benchmarks:

==========
Benchmarks
==========

Memory benchmark
================

Goal
----

It is known that gpaw puts a heavy load on the RAM memory subsystem.
This benchmark will test system's
`memory bandwidth <http://en.wikipedia.org/wiki/Memory_bandwidth>`_.

Prerequisites
-------------

This benchmark requires approximately 1.5 GB of RAM memory per core.
The amount of disk space required is minimal.

The following packages are required (names given for RHEL 5 system):

 - python, python-devel
 - numpy
 - python-matplotlib
 - openmpi, openmpi-devel
 - bash
 - `campos-gpaw <https://wiki.fysik.dtu.dk/gpaw/install/installationguide.html>`_
 - `campos-ase3 <https://wiki.fysik.dtu.dk/ase/download.html>`_

Please refer to :ref:`platforms_and_architectures` for hints on
installing gpaw on different platforms.

Results
-------

Multiple instances of the gpaw code are executed in serial
using OpenMPI in order to benchmark a number of processes that ranges from
1, through integer powers of 2 and up to the total number of CPU cores
(NCORES - number of cores available on the test machine).

The benchmark result is the average execution time in seconds when running
1, 2, up to NCORES processes, respectively, on the test machine.
The scaling of execution time with the number of processes is part of
the benchmark result.

Getting the results
-------------------

Please perform the following steps:

 - make sure that no other resources consuming processes are running,
 - set (as root) ulimit's cpu time to 5 hours::

    ulimit -t 18000

 - use the following commands to setup the benchmark::

    bash
    export NCORES=8 # default; set this variable to the number of cores in your machine
    export MACHINE=TEST # default; optional: set this to the name of you machine
    mkdir /tmp/benchmark.$$; cd /tmp/benchmark.*
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/H2Al110.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/prepare.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/run.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/twiny.py
    sh prepare.sh

 - run with (it takes 6-10 hours with NCORES=8)::

    cd $MACHINE; nohup sh ../run.sh 2>&1 | tee $MACHINE.log&

 - analyse the results::

    python ../memory_bandwidth.py

   A typical (rather bad) output may look like
   (example given for Intel Xeon dual-socket, quad-core L5k CPUs, 2.5 GHz,
   gpaw linked with Intel mkl)::

    No. of processes 1 Runtime 380.89 sec
    No. of processes 2 Runtime 396.16 sec
    No. of processes 4 Runtime 457.62 sec
    No. of processes 6 Runtime 595.06 sec
    No. of processes 8 Runtime 806.88 sec

   Ideally the time should not increase with the No. of processes,
   and increase of ~20% on 8 cores compared to 1 core can be considered
   satisfactory.

Strong scaling benchmark of a medium size system
================================================

Goal
----

Fix the problem size, vary the number of processors, and measure the speedup.
The system used in this benchmark is of medium size,
typical in state-of-the-art calculations in the year 2008,
and consists of 256 water molecules in a box of ~20**3 Angstrom**3,
120**3 grid points (grid spacing of ~0.16) and 1440 bands.
LCAO initialization step is performed, then 3 SCF steps with a constant
potential and 3 full SCF steps.
The initialization step and the full SCF steps are timed separately,
due to their different scaling.

Prerequisites
-------------

This benchmark requires approximately 1 GB of RAM memory per core
and at least 60 cores, up to 480.
The amount of disk space required is minimal.

The following packages are required (names given for FC 10 system):

 - python, python-devel
 - numpy
 - python-matplotlib
 - openmpi, openmpi-devel
 - blacs, scalapack
 - bash
 - `campos-gpaw <https://wiki.fysik.dtu.dk/gpaw/install/installationguide.html>`_
 - `campos-ase3 <https://wiki.fysik.dtu.dk/ase/download.html>`_

**Note** that gpaw has to built with scalapack enabled -
please refer to :ref:`platforms_and_architectures` for hints on
installing gpaw on different platforms.

Results
-------

to be written

on surveyor submit :svn:`gpaw/doc/devel/256H2O/b256H2O.py` using
:svn:`gpaw/doc/devel/256H2O/surveyor.sh` 


Getting the results
-------------------

to be written
