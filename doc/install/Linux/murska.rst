.. _murska:

=============
murska.csc.fi
=============

Here you find information about the the system
`<http://raketti.csc.fi/english/research/Computing_services/computing/servers/murska>`_.

Installation of packages on murska is recommended under :file:`/v/users/$USER/appl/`.

We want to use python2.4 and gcc compiler::

  > module load python
  > module swap PrgEnv-pgi PrgEnv-gnu

and use this :file:`customize.py`::

  scalapack = True

  compiler = 'gcc'

  extra_compile_args =['-O3', '-std=c99', '-funroll-all-loops']

  libraries =['gfortran','acml','scalapack','mpiblacsF77init','mpiblacs','scalapack','mpi']
  library_dirs =['/opt/hpmpi/lib/linux_amd64','/v/linux26_x86_64/opt/gcc/4.2.4/lib64','/v/users/lanzani/appl/blacs/1.1gnu','/v/linux26_x86_64/opt/scalapack/1.8.0gnu/scalapack-1.8.0']

  include_dirs +=['/opt/hpmpi/include']

  extra_link_args =['-Wl,-rpath=/opt/hpmpi/lib/linux_amd64,-rpath=/v/linux26_x86_64/opt/acml/4.1.0/gfortran64/lib,-rpath=/v/linux26_x86_64/opt/gcc/4.2.4/lib64,-rpath=/v/users/lanzani/appl/blacs/1.1gnu,-rpath=/v/linux26_x86_64/opt/scalapack/1.8.0gnu/scalapack-1.8.0']

  define_macros +=[('GPAW_MKL', '1')]

  mpicompiler = '/opt/hpmpi/bin/mpicc'
  mpilinker = mpicompiler

Then, compile GPAW ``python setup.py build_ext``.

A sample job script::

  #!/bin/csh

  #BSUB -n 4
  #BSUB -W 0:10
  #BSUB -J jobname_%J
  #BSUB -e jobname_err_%J
  #BSUB -o jobname_out_%J

  module unload unload PrgEnv-pgi/7.2-3
  module load python/2.4.3-gcc
  module load mpi/hp
  module load blacs/hpmpi/1.1gnu
  module load scalapack/1.8.0gnu
  module load ASE/svn
  module load gpaw-setups
  #set the environment variables PYTHONPATH, etc.
  setenv PYTHONPATH /v/users/$USER/appl/gpaw:$PYTHONPATH
  setenv PATH /v/users/$USER/appl/gpaw/build/bin.linux-x86_64-2.4:$PATH
  mpirun -srun gpaw-python input.py

Murska uses LSF-HPC batch system where jobs are submitted as (note the
stdin redirection)::

  > bsub < input.py

In order to use a preinstalled version of gpaw one give the command
``module load gpaw`` which sets all the correct environment variables
(:envvar:`PYTHONPATH`, :envvar:`GPAW_SETUP_PATH`, ...)

