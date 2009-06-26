.. _uranus:

======
uranus
======

The uranus machine is a cluster of dual socket, quad-core AMD Opteron
2354 CPUs, 2.2 GHz processors with 2 GB of memory per core.

Please use the unoptimized numpy.

Set these environment variables in the :file:`.bashrc` file::

  if [ -z "${PYTHONPATH}" ]
  then
      export PYTHONPATH=""
  fi

  export PYTHONPATH=/home/jaherron/lib/python:${PYTHONPATH}

  export OMPI=/usr/local/ompi-ifort
  export OPAL_PREFIX=${OMPI}
  export OMP_NUM_THREADS=1

  if [ -z "${PATH}" ]
  then
      export PATH=${OMPI}/bin 
  else
      export PATH=${OMPI}/bin:${PATH}
  fi

  if [ -z "${LD_LIBRARY_PATH}" ] 
  then
      export LD_LIBRARY_PATH=${OMPI}/lib
  else
      export LD_LIBRARY_PATH=${OMPI}/lib:${LD_LIBRARY_PATH}
  fi

and build GPAW (``python setup.py build_ext``) with this
:file:`customize.py` file::

  scalapack = True

  compiler = 'gcc'

  extra_compile_args += [
      '-O3',
      '-funroll-all-loops',
      '-fPIC',
      ]

  libraries= []

  mkl_lib_path = '/opt/intel/mkl/10.0.4.023/lib/em64t/'

  extra_link_args = [
  mkl_lib_path+'libmkl_intel_lp64.a',
  mkl_lib_path+'libmkl_sequential.a',
  mkl_lib_path+'libmkl_core.a',
  mkl_lib_path+'libmkl_blacs_openmpi_lp64.a',
  mkl_lib_path+'libmkl_scalapack.a',
  mkl_lib_path+'libmkl_blacs_openmpi_lp64.a',
  mkl_lib_path+'libmkl_intel_lp64.a',
  mkl_lib_path+'libmkl_sequential.a',
  mkl_lib_path+'libmkl_core.a',
  mkl_lib_path+'libmkl_intel_lp64.a',
  mkl_lib_path+'libmkl_sequential.a',
  mkl_lib_path+'libmkl_core.a',
  ]

  define_macros += [
    ('GPAW_MKL', '1')
  ]  

**Note**: is case of problems similar to those found on :ref:`akka` static linking is required.

A gpaw script :file:`test.py` can be submitted like this::

  qsub -l nodes=1:ppn=8 -l walltime=00:30:00 -m abe run.sh

where :file:`run.sh` looks like this::

  #!/bin/sh

  #PBS -m ae
  #PBS -M email@email.com
  #PBS -q long
  #PBS -r n
  #PBS -l nodes=1:ppn=8

  cd $PBS_O_WORKDIR
  echo Running on host `hostname` in directory `pwd`
  NPROCS=`wc -l < $PBS_NODEFILE`
  echo This jobs runs on the following $NPROCS processors:
  cat $PBS_NODEFILE

  export PYTHONPATH=~/opt/gpaw-0.5.3667:~/opt/python-ase-3.1.0.846:${PYTHONPATH}
  export PATH=~/opt/gpaw-0.5.3667/build/bin.linux-x86_64-2.4:${PATH}
  export GPAW_SETUP_PATH=~/opt/gpaw-setups-0.5.3574
  export OMP_NUM_THREADS=1

  mpiexec gpaw-python test.py

Please make sure that your jobs do not run multi-threaded, e.g. for a
job running on ``node02`` do from a login node::

  ssh node02 ps -fL

you should see **1** in the **NLWP** column. Numbers higher then **1**
mean multi-threaded job.

It's convenient to customize as described on the :ref:`parallel_runs` page.
