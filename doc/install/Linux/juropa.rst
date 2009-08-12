.. _juropa:

====================
juropa.fz-juelich.de
====================

Here you find information about the the system
`<http://www.fz-juelich.de/jsc/juropa>`_.

The instructions assume **bash** and installation of the python modules 
under `${HOME}/opt/python`. Build the unoptimized numpy::

  mkdir ${HOME}/opt/python
  cd ${HOME}/opt/python

  # nose (needed by numpy)
  # ====
  wget http://python-nose.googlecode.com/files/nose-0.11.0.tar.gz
  tar zxf nose-0.11.0.tar.gz
  cd nose-0.11.0
  python setup.py install --prefix=${HOME}/opt/python | tee install.log
  cd ..

  # numpy
  # =====
  wget http://dfn.dl.sourceforge.net/sourceforge/numpy/numpy-1.3.0.tar.gz
  tar zxf numpy-1.3.0.tar.gz
  cd numpy-1.3.0
  python setup.py install --prefix=${HOME}/opt/python | tee install.log
  cd ..
  python -c "import numpy; numpy.test()"

Build GPAW using **gcc** with this :file:`customize.py` file::

  libraries = ['mkl', 'mkl_gnu_thread', 'iomp5']
  library_dirs += ['/opt/intel/Compiler/11.0/074/lib/intel64']

and execute::

  module unload parastation/intel
  module load parastation/gcc

  python setup.py build

Job scripts can be written using::

  gpaw-runscript -h

Note, that environment variables must be submitted explicitely to 
`mpiexec`.
