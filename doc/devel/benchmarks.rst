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
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth/H2Al110.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth/prepare.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth/run.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth/memory_bandwidth.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/memory_bandwidth/twiny.py
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

Strong scaling benchmarks
=========================

Goal
----

Fix the problem size, vary the number of processors, and measure the speedup.

1) Medium size system
+++++++++++++++++++++

The system used in this benchmark is of medium size, as for the year 2008,
and consists of 256 water molecules in a box of ~20**3 Angstrom**3,
2048 electrons, and 1056 bands, and 112**3 grid points (grid spacing of ~0.18).
LCAO initialization stage is performed, then 3 SCF steps with a constant
potential and 2 full SCF steps.
All the stages are timed separately, due to their different scaling.

**Note** that the size of the system can be changed easily by modifying
just one variable in :file:`~/doc/devel/256H2O/b256H2O.py`::

  r = [2, 2, 2]

Prerequisites
+++++++++++++

This benchmark requires approximately 2 GB of RAM memory per core
and at least 32 cores, up to 512.
The amount of disk space required is minimal.

The following packages are required (names given for Fedora Core 10 system):

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
+++++++

gpaw code is executed in parallel in order to benchmark a number of processes that ranges from
32, through integer powers of 2 and up to the total number of CPU 512 cores.
The number of bands (1056) and cores are chosen to make comparisons
of different band parallelizations (:ref:`band_parallelization`) possible.
**Note** also that a default domain decomposition is appplied, and different
results can be obtained by tuning ``--domain-decomposition`` argument
to your platform (see :ref:`submit_tool_on_niflheim`).

The results of the benchmark is scaling of execution time of different stages
of gpaw run with the number of processes (CPU cores).


Getting the results
+++++++++++++++++++

Please perform the following steps:

 - use the following commands to setup the benchmark::

    bash
    mkdir 256H2O; cd 256H2O
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/b256H2O.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/akka.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/surveyor.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/prepare.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/scaling.py
    # set the prefix directory: results will be in $PATTERN_*_
    export PATTERN=b256H2O_112_04x04m64.grid
    sh prepare.sh

   **Warning**: the choice of the directory names is not free in the sense that
   the number of processes has to come at the end of directory name,
   and be delimited by two underscores.

 - run with, for example:

    - on akka::

       cd $PATTERN_00032_; qsub -l nodes=4:8 ../akka.sh; cd ..
       cd $PATTERN_00064_; qsub -l nodes=8:8 ../akka.sh; cd ..
       cd $PATTERN_00128_; qsub -l nodes=16:8 ../akka.sh; cd ..
       cd $PATTERN_00256_; qsub -l nodes=32:8 ../akka.sh; cd ..
       cd $PATTERN_00512_; qsub -l nodes=64:8 ../akka.sh; cd ..

   **Warning**: on Linux clusters it s desirable to repeat these runs 2-3 times
   to make sure that they give reproducible time.

 - analyse the results::

    python -v --dir=. --pattern="b256H2O_112_04x04m64.grid_*_" b256H2O

   A typical output may look like
   (example given for Intel Xeon dual-socket, quad-core L5k CPUs, 2.5 GHz,
   gpaw linked with Intel mkl, infiniband)::

    # p - processes, p0 - reference processes, t - time [sec], s - speedup, e - efficiency
    # GPAW version 3340: stages: 1 - initialization, 2 - fixdensity, 3 - SCF, 4 - forces, 5 - total
    # p     p/p0   t1      s1      e1    t2      s2      e2    t3      s3      e3    t4      s4      e4    t5      s5      e5
         32   1.00   100.0    32.0  1.00   339.0    32.0  1.00   235.0    32.0  1.00     0.0     0.0  0.00   674.0    32.0  1.00
         64   2.00    81.5    39.3  0.61   166.5    65.2  1.02   114.0    66.0  1.03     0.0     0.0  0.00   362.0    59.6  0.93
        128   4.00    62.0    51.6  0.40    87.0   124.7  0.97    59.0   127.5  1.00     0.0     0.0  0.00   208.0   103.7  0.81
        256   8.00    44.0    72.7  0.28    48.0   226.0  0.88    32.0   235.0  0.92     0.0     0.0  0.00   124.0   173.9  0.68
        512  16.00    43.0    74.4  0.15    36.0   301.3  0.59    24.0   313.3  0.61     0.0     0.0  0.00   103.0   209.4  0.41
 
   Clearly SCF part scales better than the initialization stage. Superscaling comes probably
   from inacurracies in printed time, and should disappear if more SCF steps were timed (verify!).


2. Medium size system
+++++++++++++++++++++

The system used in this benchmark is another one of medium size, as for the year 2008,
and consists of a gold cluster interacting with organic groups
(see `<http://www.pnas.org/cgi/content/abstract/0801001105v1>`_) in a box of 32**3 Angstrom**3,
3366 electrons, and 1920 bands, and 240**3 grid points (grid spacing of ~0.13).
LCAO initialization stage is performed, then 3 SCF steps with a constant
potential and 2 full SCF steps.
All the stages are timed separately, due to their different scaling.

**Note** that the size of the system can be changed easily by modifying
just one variable in :file:`~/doc/devel/Au_cluster/Au_cluster.py`::

  r = [1, 1, 1]

Prerequisites
+++++++++++++

This benchmark requires approximately 2 GB of RAM memory per core
and at least 512 cores, up to 4096.
The amount of disk space required is minimal.

The following packages are required (names given for Fedora Core 10 system):

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
+++++++

gpaw code is executed in parallel in order to benchmark a number of processes that ranges from
512, through integer powers of 2 and up to the total number of CPU 4096 cores.
The number of bands (1920) and cores are chosen to make comparisons
of different band parallelizations (:ref:`band_parallelization`) possible.
**Note** also that a default domain decomposition is appplied, and different
results can be obtained by tuning ``--domain-decomposition`` argument
to your platform (see :ref:`submit_tool_on_niflheim`).

The results of the benchmark is scaling of execution time of different stages
of gpaw run with the number of processes (CPU cores).


Getting the results
+++++++++++++++++++

Please perform the following steps:

 - use the following commands to setup the benchmark::

    bash
    mkdir Au_cluster; cd Au_cluster
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/Au_cluster/Au102_revised.xyz
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/Au_cluster/Au_cluster.py
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/Au_cluster/akka.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/Au_cluster/intrepid.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/Au_cluster/prepare.sh
    wget http://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/scaling.py
    # set the prefix directory: results will be in $PATTERN_*_
    export PATTERN=Au_cluster_240_05x05m64.grid
    sh prepare.sh

   **Warning**: the choice of the directory names is not free in the sense that
   the number of processes has to come at the end of directory name,
   and be delimited by two underscores.

 - run with, for example:

    - on akka::

       cd $PATTERN_00032_; qsub -l nodes=64:8 ../akka.sh; cd ..
       cd $PATTERN_00064_; qsub -l nodes=128:8 ../akka.sh; cd ..
       cd $PATTERN_00128_; qsub -l nodes=256:8 ../akka.sh; cd ..
       cd $PATTERN_00256_; qsub -l nodes=512:8 ../akka.sh; cd ..

   **Warning**: on Linux clusters it s desirable to repeat these runs 2-3 times
   to make sure that they give reproducible time.

 - analyse the results::

    python -v --dir=. --pattern="Au_cluster_240_05x05m64.grid_*_" Au_cluster

   A typical output may look like
   (example given for Intel Xeon dual-socket, quad-core L5k CPUs, 2.5 GHz,
   gpaw linked with Intel mkl, infiniband)::

    # p - processes, p0 - reference processes, t - time [sec], s - speedup, e - efficiency
    # GPAW version 2843: stages: 1 - initialization, 2 - fixdensity, 3 - SCF, 4 - forces, 5 - total
    # p     p/p0   t1      s1      e1    t2      s2      e2    t3      s3      e3    t4      s4      e4    t5      s5      e5
        512   1.00   243.5   512.0  1.00   856.5   512.0  1.00   900.0   512.0  1.00     0.0     0.0  0.00  2000.0   512.0  1.00
       1024   2.00   155.5   801.7  0.78   466.5   940.0  0.92   489.0   942.3  0.92     0.0     0.0  0.00  1111.0   921.7  0.90
       2048   4.00   148.5   839.5  0.41   241.5  1815.9  0.89   256.0  1800.0  0.88     0.0     0.0  0.00   646.0  1585.1  0.77
