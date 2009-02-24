.. _Niflheim:

========
Niflheim
========

Here you find information about the the system
`<https://wiki.fysik.dtu.dk/niflheim>`_.

**On slid and thul machines only**: if you want to use a parallel
version, before ``python setup.py build_ext``,
apply the openmpi environment settings::

  [~]$ source /usr/local/openmpi-1.2.5-gfortran/bin/mpivars-1.2.5.csh

To make it the default setting add the line to your :file:`~/.tcshrc`.
See :wiki:`niflheim/Parallelization` for details
(note however, that it contains instructions for openmpi fortran codes).

A subset of the Niflheim's nodes is equipped with Infiniband network
`<https://wiki.fysik.dtu.dk/niflheim/Hardware#infiniband-network>`_.

On the login node ``slid`` build GPAW (``python setup.py build_ext``)
with gcc compiler using the following :file:`customize.py` file::

  scalapack = True

  extra_link_args += ['-cc=gcc']
  extra_compile_args += [
    '-cc=gcc',
    '-O2',
    '-m64',
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
  mpilinker = mpicompiler

You can alternatively build on ``slid`` build GPAW (``python setup.py
build_ext``) with pathcc (pathcc looks ~3% slower - check other jobs!)
compiler using the following :file:`customize.py` file::

  scalapack = True

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
  mpilinker = mpicompiler

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
