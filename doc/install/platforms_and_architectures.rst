.. _platforms_and_architectures:

===========================
Platforms and architectures
===========================

Ubuntu
======

Install these packages:

* python-dev
* lapack3
* lapack3-dev
* refblas3
* refblas3-dev
* build-essential

If using GPAW 0.3, then:

* python-numeric
* python-numeric-ext

Else, if using trunk:

* python-numpy
* python-numpy-ext

Optional:

* atlas3-base
* atlas3-base-dev
* atlas3-headers
* python-scientific

GPAW will use atlas3 if available, which should increase performance. Python-scientific is not strictly necessary, but some tests require it. Some packages in build-essential are likewise not necessary.

Linux cluster Monolith at NSC
=============================

The Monolith_ machine is a cluster of 2.4Ghz Xeon processors with 2GB of
memory.  The ScaMPI implementation of MPI has a problem, but we can
use MPICH.

.. _Monolith: http://www.nsc.liu.se/systems/retiredsystems/monolith/

Add these two line to the :file:`.modules` file::

  python/2.3.3
  mkl/9.0p18

The Numeric Python module on the system is way too old, so we build
our own version with this :file:`customize.py` file::

  use_system_lapack = 1
  mkl = '/usr/local/intel/ict/l_ict_p_3.0.023/cmkl/9.0'
  lapack_library_dirs = [mkl + '/lib/32']
  lapack_libraries = ['mkl', 'mkl_lapack', 'g2c']
  use_dotblas = 1
  dotblas_include_dirs = [mkl + '/include']
  dotblas_cblas_header = '<mkl_cblas.h>'

Set these environment variables in the :file:`.bashrc` file::

  export PYTHONPATH=$HOME/campos-ase-2.3.4:$HOME/gpaw:$HOME/lib/python/Numeric
  export GPAW_SETUP_PATH=$HOME/setups
  export LD_LIBRARY_PATH=$MKL_ROOT

and build GPAW (``python setup.py build_ext``) with this
:file:`customize.py` file::

  extra_compile_args += ['-w1']
  mpicompiler = 'icc -Nmpich'
  custom_interpreter = True
  compiler = 'icc'

Jobs can be submitted like this::

  qsub -l nodes=2:ppn=2 -A <account> -l walltime=2:00:00 \
       -m abe run.sh

where :file:`run.sh` looks like this::

  cd $PBS_O_WORKDIR
  mpirun $HOME/gpaw/build/bin.linux-i686-2.3/gpaw-python gpaw-script.py


Linux cluster Carbon at CNM
===========================

The Carbon machine is a cluster of dual socket, quad-core Intel Xeon
5355 CPUs, 2.66 GHz processors with 2 GB of memory per core.

To build (``python setup.py install --home=~/numpy-1.0.4-1``)
numpy-1.0.4 add these lines to :file:`site.cfg`::

  [DEFAULT]
  library_dirs = /usr/local/lib:/opt/intel/mkl/10.0.2.018/lib/em64t
  include_dirs = /usr/local/include:/opt/intel/mkl/10.0.2.018/include

and, in :file:`numpy/distutils/system_info.py` change the line::

  _lib_mkl = ['mkl','vml','guide']

into::

  _lib_mkl = ['mkl','guide']

and the line::

  lapack_libs = self.get_libs('lapack_libs',['mkl_lapack32','mkl_lapack64'])

into::

  lapack_libs = self.get_libs('lapack_libs',['mkl_lapack'])

Set these environment variables in the :file:`.bashrc` file::

  export OMPI_CC=gcc
  export OMP_NUM_THREADS=1

  export PYTHONPATH=${HOME}/gpaw:${HOME}/ase3k:${HOME}/numpy-1.0.4-1/lib64/python:
  export GPAW_SETUP_PATH=${HOME}/gpaw-setups-0.4.2039

  export LD_LIBRARY_PATH=/usr/lib64/openmpi:/opt/intel/mkl/10.0.2.018/lib/em64t
  export PATH=${HOME}/gpaw/tools:${HOME}/ase3k/tools:/usr/share/openmpi/bin64:${PATH}

  if [ $PBS_ENVIRONMENT ]; then
        cd $PBS_O_WORKDIR
        export PYTHONPATH=${PBS_O_WORKDIR}:${PYTHONPATH}
        return
  fi

and build GPAW (``python setup.py build_ext``) with this
:file:`customize.py` file (comment out experimental ``scalapack`` and ``blacs`` features)::

  extra_compile_args += [
      '-O3'
      ]

  libraries = [
    'mkl',
    'mkl_lapack',
    'guide',
    'mkl_scalapack',
    'mkl_blacs_openmpi_lp64'
    ]

  library_dirs = [
    '/opt/intel/mkl/10.0.2.018/lib/em64t',
    '/usr/lib64/openmpi'
    ]

  define_macros += [
    ('GPAW_MKL', '1')
  ]

  mpi_runtime_library_dirs = [
    '/opt/intel/mkl/10.0.2.018/lib/em64t',
    '/usr/lib64/openmpi'
    ]

A gpaw script :file:`gpaw-script.py` can be submitted like this::

  qsub -l nodes=1:ppn=8 -l walltime=02:00:00 \
       -m abe run.sh

where :file:`run.sh` looks like this::

  cd $PBS_O_WORKDIR
  mpirun -machinefile $PBS_NODEFILE -np 8 -mca btl openib -mca btl_openib_retry_count 14 -x OMP_NUM_THREADS \
         $HOME/gpaw/build/bin.linux-x86_64-2.4/gpaw-python gpaw-script.py

Please make sure that your jobs do not run multi-threaded, e.g. for a
job running on ``n090`` do from a login node::

  ssh n090 ps -fL

you should see **1** in the **NLWP** column. Numbers higher then **1**
mean multi-threaded job.

It's convenient to customize as described on the :ref:`parallel_runs` page.


Linux cluster davinci.ssci.liv.ac.uk
====================================

The machine is a cluster of dual-core Intel Xeon CPUs, 3.2 GHz
processors with 2 GB of memory per core.

To build (``python setup.py install --home=~/numpy-1.1.0-1``) numpy-1.1.0 add this line to :file:`site.cfg`::

  [DEFAULT]
  library_dirs = /usr/local/Cluster-Apps/intel_mkl_7.0.1.006/mkl701/lib/32

and build GPAW (``PYTHONPATH=${HOME}/dulak/numpy-1.1.0-1/usr/local/lib/python2.5/site-packages python setup.py build_ext``) with this
``customize.py`` file::

  home='/home/haiping'

  extra_compile_args += [
      '-O3'
      ]

  libraries = [
    'mkl',
    'mkl_lapack',
    'guide'
    ]

  library_dirs = [
    '/usr/local/Cluster-Apps/intel_mkl_7.0.1.006/mkl701/lib/32'
    ]

  include_dirs += [
    home+'numpy-1.1.0-1/usr/local/lib/python2.5/site-packages/numpy/core/include'
    ]

A gpaw script :file:`test/CH4.py` can be submitted like this::

  qsub submit.sh

where :file:`submit.sh` looks like this::

  #!/bin/bash
  #
  # Script to submit an mpi job

  # ----------------------------
  # Replace these with the name of the executable 
  # and the parameters it needs

  export home=/home/haiping
  export MYAPP=${home}/gpaw-0.4.2063/build/bin.linux-i686-2.5/gpaw-python
  export MYAPP_FLAGS=${home}/gpaw-0.4.2063/test/CH4.py

  export PYTHONPATH="${home}/numpy-1.1.0-1/usr/local/lib/python2.5/site-packages"
  export PYTHONPATH="${PYTHONPATH}:${home}/gpaw-0.4.2063:${home}/python-ase-3.0.0.358"

  # ---------------------------
  # set the name of the job
  #$ -N CH4

  # request 2 slots
  #$ -pe fatmpi 2


  #################################################################
  #################################################################
  # there shouldn't be a need to change anything below this line

  export MPICH_PROCESS_GROUP=no

  # ---------------------------
  # set up the mpich version to use
  # ---------------------------
  # load the module
  if [ -f  /usr/local/Cluster-Apps/Modules/init/bash ]
  then
          .  /usr/local/Cluster-Apps/Modules/init/bash
          module load default-ethernet
  fi


  #----------------------------
  # set up the parameters for qsub
  # ---------------------------

  #  Mail to user at beginning/end/abort/on suspension
  ####$ -m beas
  #  By default, mail is sent to the submitting user 
  #  Use  $ -M username    to direct mail to another userid 

  # Execute the job from the current working directory
  # Job output will appear in this directory
  #$ -cwd
  #   can use -o dirname to redirect stdout 
  #   can use -e dirname to redirect stderr

  #  Export these environment variables
  #$ -v PATH 
  #$ -v MPI_HOME
  #$ -v LD_LIBRARY_PATH
  #$ -v GPAW_SETUP_PATH
  #$ -v PYTHONPATH
  # Gridengine allocates the max number of free slots and sets the
  # variable $NSLOTS.
  echo "Got $NSLOTS slots."

  # Gridengine sets also $TMPDIR and writes to $TMPDIR/machines the
  # corresponding list of nodes. It also generates some special scripts in
  # $TMPDIR. Therefore, the next two lines are practically canonical:
  #
  #
  export PATH=$TMPDIR:$PATH
  #

  echo "Stack size is "`ulimit -S -s`

  # ---------------------------
  # run the job
  # ---------------------------
  date_start=`date +%s`
  $MPI_HOME/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines  $MYAPP $MYAPP_FLAGS
  date_end=`date +%s`
  seconds=$((date_end-date_start))
  minutes=$((seconds/60))
  seconds=$((seconds-60*minutes))
  hours=$((minutes/60))
  minutes=$((minutes-60*hours))
  echo =========================================================   
  echo SGE job: finished   date = `date`   
  echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
  echo ========================================================= 

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at :ref:`parallel_runs`


Linux cluster Niflheim - Infiniband nodes
=========================================

A subset of the Niflheim's nodes is equipped with Infiniband network
`<https://wiki.fysik.dtu.dk/niflheim/Hardware#infiniband-network>`_.

On the login node ``slid`` build GPAW (``python setup.py build_ext``)
with gcc compiler using the following :file:`customize.py` file
(comment out experimental ``scalapack`` and ``blacs`` features)::

  extra_link_args += ['-cc=gcc']
  extra_compile_args += [
    '-cc=gcc',
    '-pthread',
    '-fno-strict-aliasing',
    '-DNDEBUG',
    '-O2',
    '-g',
    '-pipe',
    '-m64',
    '-fPIC',
    '-UNDEBUG'
  ]

  libraries = [
    'pathfortran',
    'gfortran',
    'mpiblacsCinit',
    'acml',
    'mpiblacs',
    'scalapack'
    ]

  library_dirs = [
    '/opt/pathscale/lib/2.5',
    '/opt/acml-4.0.1/gfortran64/lib',
    '/usr/local/blacs-1.1-24.6.infiniband/lib64',
    '/usr/local/scalapack-1.8.0-1.infiniband/lib64',
    '/usr/local/infinipath-2.0/lib64'
    ]

  include_dirs += [
    '/usr/local/infinipath-2.0/include'
   ]

  extra_link_args += [
    '-Wl,-rpath=/opt/pathscale/lib/2.5',
    '-Wl,-rpath=/opt/acml-4.0.1/gfortran64/lib',
    '-Wl,-rpath=/usr/local/blacs-1.1-24.6.infiniband/lib64',
    '-Wl,-rpath=/usr/local/scalapack-1.8.0-1.infiniband/lib64',
    '-Wl,-rpath=/usr/local/infinipath-2.0/lib64'
  ]

  define_macros += [
    ('GPAW_MKL', '1'),
    ('SL_SECOND_UNDERSCORE', '1')
  ]

  mpicompiler = '/usr/local/infinipath-2.0/bin/mpicc'

You can alternatively build on ``slid`` build GPAW (``python setup.py
build_ext``) with pathcc (pathcc looks ~3% slower - check other jobs!)
compiler using the following :file:`customize.py` file (comment out
experimental ``scalapack`` and ``blacs`` features)::

  libraries = [
    'pathfortran',
    'mpiblacsCinit',
    'acml',
    'mpiblacs',
    'scalapack'
    ]

  library_dirs = [
    '/opt/pathscale/lib/2.5',
    '/opt/acml-4.0.1/pathscale64/lib',
    '/usr/local/blacs-1.1-24.6.infiniband/lib64',
    '/usr/local/scalapack-1.8.0-1.infiniband/lib64',
    '/usr/local/infinipath-2.0/lib64'
    ]

  extra_link_args += [
    '-Wl,-rpath=/opt/pathscale/lib/2.5',
    '-Wl,-rpath=/opt/acml-4.0.1/pathscale64/lib',
    '-Wl,-rpath=/usr/local/blacs-1.1-24.6.infiniband/lib64',
    '-Wl,-rpath=/usr/local/scalapack-1.8.0-1.infiniband/lib64',
    '-Wl,-rpath=/usr/local/infinipath-2.0/lib64'
  ]

  define_macros += [
    ('GPAW_MKL', '1'),
    ('SL_SECOND_UNDERSCORE', '1')
  ]

  mpicompiler = '/usr/local/infinipath-2.0/bin/mpicc -Ofast'

A gpaw script :file:`gpaw-script.py` can be submitted like this::

  qsub -l nodes=1:ppn=4:infiniband -l walltime=02:00:00 \
       -m abe run.sh

where :file:`run.sh` for gcc version looks like this::

  cd $PBS_O_WORKDIR
  export LD_LIBRARY_PATH=/opt/pathscale/lib/2.5
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/acml-4.0.1/gfortran64/lib
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/blacs-1.1-24.6.infiniband/lib64
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/scalapack-1.8.0-1.infiniband/lib64
  mpirun -machinefile $PBS_NODEFILE -np 4 \
         $HOME/gpaw/build/bin.linux-x86_64-2.4/gpaw-python gpaw-script.py

and for pathcc version looks like this::

  cd $PBS_O_WORKDIR
  export LD_LIBRARY_PATH=/opt/pathscale/lib/2.5
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/acml-4.0.1/pathscale64/lib
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/blacs-1.1-24.6.infiniband/lib64
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/scalapack-1.8.0-1.infiniband/lib64
  mpirun -machinefile $PBS_NODEFILE -np 4 \
         $HOME/gpaw/build/bin.linux-x86_64-2.4/gpaw-python gpaw-script.py

Please make sure that the threads use 100% of CPU, e.g. for a job running on ``p024`` do from ``audhumbla``::

  ssh p024 ps -fL

Numbers higher then **1** in the **NLWP** column mean multi-threaded job.

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.


Sun Solaris
===========


corona.csc.fi
-------------

Submit jobs like this::

  qsub -pe cre 8 -cwd -V -S /p/bin/python job.py


bohr.gbar.dtu.dk
----------------

Follow instructions from `<http://www.gbar.dtu.dk/index.php/GridEngine>`_ to create :file:`~/.grouprc`.

Download `MPIscript.sh <http://www.hpc.dtu.dk/GridEngine/MPIscript.sh>`_ and edit it, so it resembles::

 #!/bin/sh 
 # (c) 2000 Sun Microsystems, Inc.
 # ---------------------------
 # General options
 #
 #$ -S /bin/sh
 #$ -o $JOB_NAME.$JOB_ID.out
 #$ -e $JOB_NAME.$JOB_ID.err
 # -M User@Domain
 # -m es
 # ---------------------------
 # Execute the job from  the  current  working  directory
 #$ -cwd
 #
 # Parallel environment request
 # ---------------------------
 # do not change the following line
 #$ -l cre
 #
 #      PE_name  CPU_Numbers_requested
 ##$ -pe HPC      2
 # ------------------------------- Program_name_and_options
 LD_LIBRARY_PATH=/opt/csw/lib:${LD_LIBRARY_PATH}
 export LD_LIBRARY_PATH
 gpawpython=/xbar/nas2/home2/fys/v40082/gpaw/build/bin.solaris-2.10-sun4u-2.5/gpaw-python
 /appl/hgrid/current/bin/mprun -np $NSLOTS $gpawpython gpaw/examples/H.py
 # ---------------------------

Submit jobs like this::

  qsub -N test -pe HPC 2 MPIscript.sh


IBM
===

jump.fz-juelich.de
------------------

The only way we found to compile numpy is using python2.3 and
numpy-1.0.4. The next version numpy-1.1.0 did not work
unfortunately. In addition the usage of the generic IBM lapack/blas in
numpy does not work, hence one has to use site.cfg::

  : diff site.cfg site.cfg.example
  58,60c58,60
  < [DEFAULT]
  < library_dirs =
  < include_dirs =
  ---
  > #[DEFAULT]
  > #library_dirs = /usr/local/lib
  > #include_dirs = /usr/local/include

With his change numpy compiles and the installation of ASE and gpaw
does not cause problems.

seaborg.nersc.gov
-----------------

We need to use the mpi-enabled compiler ``mpcc`` and we should link to
LAPACK before ESSL.  Make sure LAPACK is added::

  $ module add lapack

and use this customize.py::

  from os import environ
  mpicompiler = 'mpcc'
  libraries = ['f']
  extra_link_args += [environ['LAPACK'], '-lessl']

The Numeric Python extension is not installed on NERSC, so we should
install it.  Get the Numeric-24.2_ and do this::

  $ gunzip -c Numeric-24.2.tar.gz | tar xf -
  $ cd Numeric-24.2
  $ python setup.py install --home=$HOME

and put the :file:`$HOME/lib/python/Numeric` directory in your
:envvar:`$PYTHONPATH`.

Now we are ready to :ref:`compile GPAW <installationguide>`

.. _Numeric-24.2: http://downloads.sourceforge.net/numpy/Numeric-24.2.tar.gz
.. _numpy-1.0.4: http://downloads.sourceforge.net/numpy/numpy-1.0.4.tar.gz

ibmsc.csc.fi
------------

Debug like this::

  p690m ~/gpaw/demo> dbx /p/bin/python2.3
  Type 'help' for help.
  reading symbolic information ...warning: no source compiled with -g
  
  (dbx) run
  Python 2.3.4 (#4, May 28 2004, 15:30:35) [C] on aix5
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import grr

surveyor.alcf.anl.gov
---------------------

The **0.3** version of gpaw uses Numeric `<https://svn.fysik.dtu.dk/projects/gpaw/tags/0.3/>`_.
The latest version of gpaw uses numpy `<https://svn.fysik.dtu.dk/projects/gpaw/trunk/>`_.

Get the Numeric-24.2_ (only if you want to run the **0.3** version)
and do this::

  $ gunzip -c Numeric-24.2.tar.gz | tar xf -
  $ cd Numeric-24.2
  $ /bgsys/drivers/ppcfloor/gnu-linux/bin/python setup.py install --root=$HOME/Numeric-24.2-1

To build numpy, save the numpy-1.0.4-gnu.py.patch_ patch file 
(modifications required to get mpif77 instead of gfortran compiler),
get and numpy-1.0.4_ and do this::

  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ cd numpy-1.0.4
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ root=$HOME/numpy-1.0.4-1
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"mpicc\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

Set these environment variables in the :file:`.softenvrc` file::

  PYTHONPATH = ${HOME}/Numeric-24.2-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/Numeric
  PYTHONPATH += ${HOME}/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
  PYTHONPATH += ${HOME}/gpaw:${HOME}/CamposASE2:${HOME}/ase3k
  GPAW_SETUP_PATH = ${HOME}/gpaw-setups-0.4.2039

  LD_LIBRARY_PATH += /bgsys/drivers/ppcfloor/runtime/SPI
  LD_LIBRARY_PATH += /opt/ibmcmp/xlf/bg/11.1/bglib:/opt/ibmcmp/lib/bg
  LD_LIBRARY_PATH += /opt/ibmcmp/xlsmp/bg/1.7/bglib:/bgsys/drivers/ppcfloor/gnu-linux/lib
  PATH += ${HOME}/gpaw/tools:${HOME}/CamposASE2/tools:${HOME}/ase3k/tools
  # to enable TAU profiling add also:
  PYTHONPATH += /soft/apps/tau/tau_latest/bgp/lib/bindings-mpi-gnu-python-pdt
  LD_LIBRARY_PATH += /soft/apps/tau/tau_latest/bgp/lib/bindings-mpi-gnu-python-pdt
  TAU_THROTTLE = 1

and do::

  resoft

(to enable TAU profiling do also ``soft add +tau``),
and build GPAW (``/bgsys/drivers/ppcfloor/gnu-linux/bin/python
setup.py build_ext``) with this :file:`customize.py` file (comment out
experimental ``scalapack`` and ``blacs`` features)::

  extra_compile_args += [
      '-O3'
      ]

  libraries = [
             'lapack_bgp',
             'scalapack',
             'blacsCinit_MPI-BGP-0',
             'blacs_MPI-BGP-0',
             'lapack_bgp',
             'goto',
             'xlf90_r',
             'xlopt',
             'xl',
             'xlfmath',
             'xlsmp'
             ]

  library_dirs = [
             '/soft/apps/LAPACK',
             '/soft/apps/LIBGOTO',
             '/soft/apps/BLACS',
             '/soft/apps/SCALAPACK',
             '/opt/ibmcmp/xlf/bg/11.1/bglib',
             '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
             '/bgsys/drivers/ppcfloor/gnu-linux/lib'
             ]

  gpfsdir = '/home/dulak'
  python_site = 'bgsys/drivers/ppcfloor/gnu-linux'

  include_dirs += [gpfsdir+'/Numeric-24.2-1/'+python_site+'/include/python2.5',
                   gpfsdir+'/numpy-1.0.4-1/'+python_site+'/lib/python2.5/site-packages/numpy/core/include']

  extra_compile_args += ['-std=c99']

  define_macros += [
            ('GPAW_AIX', '1'),
            ('GPAW_MKL', '1'),
            ('GPAW_BGP', '1')
            ]

  # uncomment the two following lines to enable TAU profiling
  #tau_make = '/soft/apps/tau/tau_latest/bgp/lib/Makefile.tau-mpi-gnu-python-pdt'
  #mpicompiler = 'tau_cc.sh -tau_options="-optShared -optVerbose" -tau_makefile='+tau_make

Because of missing ``popen3`` function you need to remove all the
contents of the :file:`gpaw/version.py` file after ``version =
'0.4'``.  The same holds for :file:`ase/version.py` in the ase
installation!  Suggestions how to skip the ``popen3`` testing in
:file:`gpaw/version.py` on BGP are welcome!

A gpaw script ``CH4.py`` (fetch it from ``gpaw/test``) can be submitted like this::

  qsub -n 2 -t 10 --mode vn --env \
       OMP_NUM_THREADS=1:GPAW_SETUP_PATH=$GPAW_SETUP_PATH:PYTHONPATH=$PYTHONPATH:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib:LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
       ${HOME}/gpaw/build/bin.linux-ppc64-2.5/gpaw-python ${HOME}/CH4.py

Absolute paths are important!

If you want to perform profiling with TAU submit the following wrapper instead::

  import tau
  from gpaw.mpi import rank

  def OurMain():
      import CH4;

  tau.run('OurMain()')

This TAU run will produce ``profile.*`` files that can be merged into
the default TAU's ``ppk`` format using the command issued from the directory
where the ``profile.*`` files reside::

 paraprof --pack CH4.ppk

The actual analysis can be made on a different machine, by transferring
the ``CH4.ppk`` file from `surveyor`, installing TAU, and launching::

 paraprof CH4.ppk

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.


bcssh.rochester.ibm.com
-----------------------

Instructions below are valid for ``frontend-13`` and the filesystem
:file:`/gpfs/fs2/frontend-13`.

The latest version of gpaw uses numpy
`<https://svn.fysik.dtu.dk/projects/gpaw/trunk/>`_.

To build an optimized? (this does not work completely, see problems
below) numpy, save the
numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran_ patch file
(modifications required to get powerpc-bgp-linux-gfortran instead of
gfortran compiler), the
numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg_ patch file (lapack
section configured to use ``lapack_bgp`` and blas section to use
``esslbg``, the numpy-1.0.4-site.cfg.lapack_bgp_esslbg_ file (contains
paths to ``lapack_bgp``, ``esslbg`` and xlf* related libraries).
**Note** that ``lapack_bgp`` is not available on ``frontend-13``, use
a personal of somebody else's version!  Get numpy-1.0.4_ and do this::

  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ mv numpy-1.0.4 numpy-1.0.4.optimized; cd numpy-1.0.4.optimized
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
  $ patch -p1 < ../numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg
  $ cp ../numpy-1.0.4-site.cfg.lapack_bgp_esslbg site.cfg
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ mkdir /gpfs/fs2/frontend-13/$USER
  $ root=/gpfs/fs2/frontend-13/$USER/numpy-1.0.4-1.optimized
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc -DNO_APPEND_FORTRAN\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

Numpy built in this way does not build the
:file:`$root/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/_dotblas.so`
(numpy requires cblas for this), and running the following python
script (save it as :file:`/gpfs/fs2/frontend-13/$USER/dot.py`) for the
optimized and standard versions of numpy show the same time (~ 329
sec) for ``numpy.dot`` operation::

  import numpy
  print numpy.__file__
  #import Numeric

  from time import time

  N = 1700

  num = numpy.array(numpy.zeros((N,N)))
  #Num = Numeric.array(Numeric.zeros((N,N)))

  t = time()
  numpy.dot(num, num)
  print 'numpy', time()-t

  #t = time()
  #Numeric.dot(Num, Num)
  #print 'Numeric', time()-t

Use the following command to submit this job ``cd
/gpfs/fs2/frontend-13/$USER; llsubmit numpy.llrun``, with the
following :file:`numpy.llrun` file::

  #!/bin/bash

  # @ job_type = bluegene
  # @ requirements = (Machine == "$(host)")
  # @ class = medium
  # @ job_name = $(user).$(host)
  # @ comment = "LoadLeveler llrun script"
  # @ error = $(job_name).$(jobid).err
  # @ output = $(job_name).$(jobid).out
  # @ wall_clock_limit = 00:15:00
  # @ notification = always
  # @ notify_user =
  # @ bg_connection = prefer_torus
  # @ bg_size = 32
  # @ queue

  dir="/gpfs/fs2/frontend-13/${USER}"
  home=$dir
  prog=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  args=${dir}/dot.py

  ldpath="${ldpath}:/bgsys/opt/ibmcmp/lib/bg"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/lib"
  pythonpath=":${home}/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages:"

  export LD_LIBRARY_PATH=\"$ldpath\"
  export PYTHONPATH=\"$pythonpath\"
  export OMP_NUM_THREADS=1

  mpirun=/bgsys/drivers/ppcfloor/bin/mpirun

  runargs="-np 1"
  runargs="$runargs -cwd $PWD"
  runargs="$runargs -exp_env LD_LIBRARY_PATH -exp_env PYTHONPATH -exp_env OMP_NUM_THREADS"
  runargs="$runargs -mode SMP"
  runargs="$runargs -verbose 2"

  echo "Hello. This is `hostname` at `date` `pwd`"

  echo "$mpirun $runargs $prog $args"
  /usr/bin/time $mpirun $runargs $prog $args

  echo "Program completed at `date` with exit code $?."

**Note** the colon before and after the string when setting pythonpath!

Here is how you build the standard numpy::

  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ cd numpy-1.0.4
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ mkdir /gpfs/fs2/frontend-13/$USER
  $ root=/gpfs/fs2/frontend-13/$USER/numpy-1.0.4-1
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

Suggestions on how to build numpy using an optimized blas (preferably
essl) are welcome!

Build GPAW
(``PYTHONPATH=/gpfs/fs2/frontend-13/mdulak/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
LD_LIBRARY_PATH="$ldpath" $p setup.py build_ext``) in
:file:`/gpfs/fs2/frontend-13/$USER/gpaw` (you need to install the ase
also somewhere below :file:`/gpfs/fs2/frontend-13/$USER`!)  with this
:file:`customize.py` file (comment out experimental ``scalapack`` and
``blacs`` features)::

  extra_compile_args += [
      '-DNDEBUG',
      '-g',
      '-O3',
      '-Wall',
      '-Wstrict-prototypes',
      '-dynamic',
      '-fPIC'
      ]

  libraries = [
             'gfortran',
             'lapack_bgp',
             'scalapack',
             'blacs',
             'lapack_bgp',
             'goto',
             'xlf90_r',
             'xlopt',
             'xl',
             'xlfmath',
             'xlsmp'
             ]

  library_dirs = [
             '/home/mdulak/blas-lapack-lib',
             '/home/mdulak/blacs-dev',
             '/home/mdulak/SCALAPACK',
             '/opt/ibmcmp/xlf/bg/11.1/bglib',
             '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
             '/bgsys/drivers/ppcfloor/gnu-linux/lib'
             ]

  gpfsdir = '/gpfs/fs2/frontend-13/mdulak'
  python_site = 'bgsys/drivers/ppcfloor/gnu-linux'

  include_dirs += [gpfsdir+'/Numeric-24.2-1/'+python_site+'/include/python2.5',
                   gpfsdir+'/numpy-1.0.4-1/'+python_site+'/lib/python2.5/site-packages/numpy/core/include']

  extra_compile_args += ['-std=c99']

  define_macros += [
            ('GPAW_AIX', '1'),
            ('GPAW_MKL', '1'),
            ('GPAW_BGP', '1')
            ]

Because of missing ``popen3`` function you need to remove all the
contents of the :file:`gpaw/version.py` file after ``version =
'0.4'``.  The same holds for :file:`ase/version.py` in the ase
installation!  Suggestions how to skip the ``popen3`` testing in
:file:`gpaw/version.py` on BGP are welcome!

Note that only files located below :file:`/gpfs/fs2/frontend-13` are
accesible to the compute nodes (even python scripts!).  A gpaw script
:file:`/gpfs/fs2/frontend-13/$USER/gpaw/test/CH4.py` can be submitted to
32 CPUs in the single mode (SMP) for 30 minutes using `LoadLeveler
<http://www.fz-juelich.de/jsc/ibm-bgl/usage/loadl/>`_ like this::

  cd /gpfs/fs2/frontend-13/$USER
  llsubmit gpaw-script.llrun

where :file:`gpaw-script.llrun` looks like this::

  #!/bin/bash

  # @ job_type = bluegene
  # @ requirements = (Machine == "$(host)")
  # @ class = medium
  # @ job_name = $(user).$(host)
  # @ comment = "LoadLeveler llrun script"
  # @ error = $(job_name).$(jobid).err
  # @ output = $(job_name).$(jobid).out
  # @ wall_clock_limit = 00:30:00
  # @ notification = always
  # @ notify_user =
  # @ bg_connection = prefer_torus
  # @ bg_size = 32
  # @ queue

  dir=/gpfs/fs2/frontend-13/$USER
  home=$dir
  prog=${home}/gpaw/build/bin.linux-ppc64-2.5/gpaw-python
  #prog=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  args="${home}/gpaw/test/CH4.py --sl_diagonalize=2,2,2,4"

  ldpath="${ldpath}:/bgsys/opt/ibmcmp/lib/bg"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/lib"
  pythonpath=":${home}/Numeric-24.2-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/Numeric"
  pythonpath="${pythonpath}:${home}/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages"
  pythonpath="${pythonpath}:${home}/gpaw"
  pythonpath="${pythonpath}:${home}/ase3k:"

  export LD_LIBRARY_PATH=\"$ldpath\"
  export PYTHONPATH=\"$pythonpath\"
  export GPAW_SETUP_PATH="${home}/gpaw-setups-0.4.2039"
  export OMP_NUM_THREADS=1

  mpirun=/bgsys/drivers/ppcfloor/bin/mpirun

  runargs="-np 32"
  runargs="$runargs -cwd $PWD"
  runargs="$runargs -exp_env LD_LIBRARY_PATH -exp_env PYTHONPATH -exp_env GPAW_SETUP_PATH -exp_env OMP_NUM_THREADS"
  runargs="$runargs -mode SMP"
  runargs="$runargs -verbose 1"

  echo "Hello. This is `hostname` at `date` `pwd`"

  echo "$mpirun $runargs $prog $args"
  /usr/bin/time $mpirun $runargs $prog $args

  echo "Program completed at `date` with exit code $?."

Absolute paths are important!

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.


HP
==

sepeli.csc.fi
-------------

The installed subversion in sepeli does not support https-protocol, so
one should use a tar file.

Compile like this::

  # use the following modules and define the right python interpreter
  sepeli ~/gpaw/trunk> use mvapich-gnu64
  mvapich-gnu64 is now in use

  MVAPICH environment set
  MPIHOME=/opt/mvapich//gnu64/

  sepeli ~/gpaw/trunk> use ASE
  Atomic Simulation Environment in use
  [ASE is now in use]
  sepeli ~/gpaw/trunk> alias python 'python-pathscale64'
  sepeli ~/gpaw/trunk> unsetenv CC; unsetenv CFLAGS; unsetenv LDFLAGS
  
On runtime you need the following::

  # make shure, that the right acml library is found
  sepeli> setenv LD_LIBRARY_PATH "/opt/acml/gnu64/lib:${LD_LIBRARY_PATH}"

.. Note::

   The compute nodes have different filesystem than the front end
   node. Especially, :envvar:`$HOME` and :envvar:`$METAWRK` are
   mounted only on the frontend, so one should place gpaw on 
   :envvar:`$WRKDIR`

A sample job script with mvapich (Infiniband) MPI::

   #$ -cwd
   #$ -pe mvapich-gnu64-4 8
   #$ -S /bin/csh
   setenv PYTHONPATH /path_to_ase/:/path_to_gpaw/
   setenv GPAW_SETUP_PATH /path_to_setups/
   setenv PATH "$PATH":/path_to_gpaw-python/
   mpirun -np 8 gpaw-python input.py

In order to use a preinstalled version of gpaw one give the command
``use gpaw`` which sets all the correct environment variables
(:envvar:`PYTHONPATH`, :envvar:`GPAW_SETUP_PATH`, ...)

murska.csc.fi
-------------

We want to use python2.4 and gcc compiler::

  > module load python
  > module swap PrgEnv-pgi  PrgEnv-gnu

and use this :file:`customize.py`::

  libraries = ['acml', 'gfortran']

Then, :ref:`compile GPAW <installationguide>`.

A sample job script::

  #!/bin/csh

  #BSUB -n 4
  #BSUB -W 0:10
  #BSUB -J jobname_%J
  #BSUB -e jobname_err_%J
  #BSUB -o jobname_out_%J

  #set the environment variables PYTHONPATH, etc.
  setenv PYTHONPATH ...
  mpirun -srun gpaw-python input.py

Murska uses LSF-HPC batch system where jobs are submitted as (note the
stdin redirection)::

  > bsub < input.py

In order to use a preinstalled version of gpaw one give the command
``module load gpaw`` which sets all the correct environment variables
(:envvar:`PYTHONPATH`, :envvar:`GPAW_SETUP_PATH`, ...)

SGI
===

batman.chem.jyu.fi
------------------

To prepare the compilation, we need to load the required modules and
clean the environment::

 > module purge # remove all modules
 > module add mpt
 > module add mkl
 > unset CC CFLAGS LDFLAGS

We have to change :file:`customize.py` to get the libs and the right compiler::

 # uncomment and change in customize.py
 libraries += ['mpi','mkl']

 mpicompiler = 'gcc'
 custom_interpreter = True

Then compile as usual (``python setup.py build``). This will build the
custom python interpreter for parallel use also.

Cray XT4
========

louhi.csc.fi
------------

The current operating system in Cray XT4 compute nodes, Compute Linux
Environment (CLE) has some limitations, most notably it does not
support shared libraries. In order to use python in CLE some
modifications to the standard python are needed. Before installing a
special python, there are two packages which are needed by GPAW, but
which are not included in the python distribution. Installation of
expat_ and zlib_ should succee with a standard ``./configure; make;
make install;`` procedure.

.. _expat: http://expat.sourceforge.net/
.. _zlib: http://www.zlib.net/  

Next, one can proceed with the actual python installation. The
following instructions are tested with python 2.5.1, and it is assumed
that one is working in the top level of python source
directory. First, one should create a special dynamic loader for
correct resolution of namespaces. Create a file :file:`dynload_redstorm.c`
in the :file:`Python/` directory::

  /* This module provides the simulation of dynamic loading in Red Storm */

  #include "Python.h"
  #include "importdl.h"

  const struct filedescr _PyImport_DynLoadFiletab[] = {
    {".a", "rb", C_EXTENSION},
    {0, 0}
  };

  extern struct _inittab _PyImport_Inittab[];

  dl_funcptr _PyImport_GetDynLoadFunc(const char *fqname, const char *shortname,
                                      const char *pathname, FILE *fp)
  {
    struct _inittab *tab = _PyImport_Inittab;
    while (tab->name && strcmp(shortname, tab->name)) tab++;

    return tab->initfunc;
  }

dynload_redstorm.c_

Then, one should remove ``sharemods`` from ``all:`` target in
:file:`Makefile.pre.in` and set the correct C compiler and flags,
e.g.::

 setenv CC cc
 setenv OPT '-fastsse'

You should be now ready to run :file:`configure`::

  ./configure --prefix=<install_path> SO=.a DYNLOADFILE=dynload_redstorm.o MACHDEP=redstorm --host=x86_64-unknown-linux-gnu --disable-sockets --disable-ssl --enable-static --disable-shared --without-threads

Now, one should specify which modules will be statically linked in to
the python interpreter by editing :file:`Modules/Setup`. An example can be
loaded here. Setup_. Note that at this point all numpy related stuff
in the example should be commented out. Finally, in order to use
``distutils`` for building extensions the following function should be
added to the end of :file:`Lib/distutils/unixccompiler.py` so that instead
of shared libraries static ones are created::

    def link_shared_object (self,
                         objects,
                         output_filename,
                         output_dir=None,
                         libraries=None,
                         library_dirs=None,
                         runtime_library_dirs=None,
                         export_symbols=None,
                         debug=0,
                         extra_preargs=None,
                         extra_postargs=None,
                         build_temp=None,
                         target_lang=None):

        if output_dir is None:
            (output_dir, output_filename) = os.path.split(output_filename)
        output_fullname = os.path.join(output_dir, output_filename)
        linkline = "%s %s" % (output_filename[:-2], output_fullname)
        for l in library_dirs:
            linkline += " -L" + l
        for l in libraries:
            linkline += " -l" + l
        old_fmt = self.static_lib_format
        self.static_lib_format = "%s%.0s"
        self.create_static_lib(objects,
                               output_filename,
                               output_dir,
                               debug,
                               target_lang)

        self.static_lib_format = old_fmt
        print "Append to Setup: ", linkline

unixccompiler.py_

You should be now ready to run ``make`` and ``make install`` and have
a working python interpreter.

Next, one can use the newly created interpreter for installing
``numpy``. Switch to the ``numpy`` source directory and install it
normally::

  <your_new_python> setup.py install >& install.log

The C-extensions of numpy have to be still added to the python
interpreter. Grep :file:`install.log`::

  grep 'Append to Setup' install.log

and add the correct lines to the :file:`Modules/Setup` in the python
source tree. Switch to the python source directory and run ``make``
and ``make install`` again to get interpreter with builtin numpy.

Final step is naturaly to compile GPAW. Only thing is to specify
``numpy``, ``expat`` and ``zlib`` libraries in :file:`customize.py`
then `compile GPAW <installationguide>` as usual. Here is an example
of :file:`customize.py`, modify according your own directory
structures:

.. literalinclude:: customize.py

Now you should be ready for massively parallel calculations, a sample
job file would be::

  #!/bin/csh
  #
  #PBS -N jobname
  #PBS -l walltime=24:00
  #PBS -l mppwidth=512

  cd /wrk/my_workdir
  # set the environment variables
  SETENV PYTHONPATH ...

  aprun -n 512 /path_to_gpaw_bin/gpaw-python input.py

In order to use a preinstalled version of gpaw one can give the
command ``module load gpaw`` which sets all the correct environment
variables (:envvar:`PYTHONPATH`, :envvar:`GPAW_SETUP_PATH`, ...)

.. _numpy-1.0.4-gnu.py.patch: ../_static/numpy-1.0.4-gnu.py.patch
.. _numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran: ../_static/numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
.. _numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg: ../_static/numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg
.. _numpy-1.0.4-site.cfg.lapack_bgp_esslbg: ../_static/numpy-1.0.4-site.cfg.lapack_bgp_esslbg
.. _dynload_redstorm.c: ../_static/dynload_redstorm.c
.. _unixccompiler.py: ../_static/unixccompiler.py
.. _setup: ../_static/setup

