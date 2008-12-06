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
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/H2Al110.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/prepare.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/run.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/memory_bandwidth.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/twiny.py
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
