.. _bwgrid:

=========
bwgrid
=========

The bwgrid is an grid of machines located in Baden-Württemberg, Germany.
The installation in Freiburg is an cluster of dual socket, quad-core
Intel Xenon 5440 CPUs, 2.83GHz processors with 2 GB of memory per core.

Instructions assume **bash**, installation under `$HOME/opt`.
Load the necessary modules::

  module load system/python/2.6
  module load compiler/intel/10.1
  module load numlib/mkl/10.0-intel-10.1
  module load compiler/gnu/4.3
  module load mpi/openmpi/1.2.8_static-gnu-4.3
 
Build the unoptimized numpy::

  mkdir $HOME/opt
  cd $HOME/opt

  mkdir -p ${HOME}/opt/python/lib/python2.6/site-packages
  export PYTHONPATH=${HOME}/opt/python/lib/python2.6/site-packages

  wget http://dfn.dl.sourceforge.net/sourceforge/numpy/numpy-1.3.0.tar.gz
  wget http://python-nose.googlecode.com/files/nose-0.11.0.tar.gz
  tar zxf nose-0.11.0.tar.gz
  tar zxf numpy-1.3.0.tar.gz
  cd nose-0.11.0
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ../numpy-1.3.0
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ..
  python -c "import numpy; numpy.test()"

and build GPAW (``python setup.py build_ext | tee build_ext.log``)
with this :file:`customize.py` file
(**Note**: instructions valid from the **5232** release)::

  scalapack = True

  compiler = 'gcc'

  extra_compile_args += [
      '-O3',
      '-funroll-all-loops',
      '-fPIC',
      ]

  libraries= []

  mkl_lib_path = '/opt/bwgrid/compiler/intel/ct_3.1.1/mkl/10.0.3.020/lib/em64t/'

  library_dirs = [mkl_lib_path]

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

  define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
  define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]


**Note**: You also have to set `LD_LIBRARY_PATH=` to::

  LD_LIBRARY_PATH="/opt/system/ofed/1.3/lib:/opt/system/ofed/1.3/lib64:$LD_LIBRARY_PATH"

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > gpaw-runscript test.py 8
  using pbs_bwg
  run.pbs_bwg written
  > qsub run.pbs_bwg

