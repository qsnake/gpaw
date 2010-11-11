.. _bwgrid:

=========
bwgrid
=========

The `BWgrid <http://www.bw-grid.de/>`_
is an grid of machines located in Baden-Württemberg, Germany.
The installation in Freiburg is an cluster of dual socket, quad-core
Intel Xenon 5440 CPUs, 2.83GHz processors with 2 GB of memory per core.

Instructions assume **bash**, installation under $HOME/opt.
Load the necessary modules::

  module load system/python/2.6
  module load mpi/openmpi/1.2.8-gnu-4.1
 
You can also use gcc-4.3 instead of gcc-4.1, but numpys
test will complain about different fortran versions::

  module load system/python/2.6
  module load compiler/gnu/4.3
  module load mpi/openmpi/1.2.8_static-gnu-4.3
 

Build the unoptimized numpy::

  mkdir $HOME/opt
  cd $HOME/opt

  mkdir -p ${HOME}/opt/python/lib/python2.6/site-packages
  export PYTHONPATH=${HOME}/opt/python/lib/python2.6/site-packages

  wget http://dfn.dl.sourceforge.net/sourceforge/numpy/numpy-1.5.0.tar.gz
  wget http://python-nose.googlecode.com/files/nose-0.11.0.tar.gz
  tar zxf nose-0.11.0.tar.gz
  tar zxf numpy-1.5.0.tar.gz
  cd nose-0.11.0
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ../numpy-1.5.0
  python setup.py build --fcompiler=gnu95  | tee build.log
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ..
  python -c "import numpy; numpy.test()"

and build GPAW (``python setup.py build_ext | tee build_ext.log``)

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > gpaw-runscript test.py 8
  using pbs_bwg
  run.pbs_bwg written
  > qsub run.pbs_bwg

