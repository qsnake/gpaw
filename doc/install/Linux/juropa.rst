.. _juropa:

====================================================
juropa.fz-juelich.de   (Intel Xeon, Infiniband, MKL)
====================================================

Here you find information about the the system
`<http://www.fz-juelich.de/jsc/juropa>`_.

Numpy is installed system wide, so separate installation is not needed.

Building GPAW with gcc
======================

Build GPAW using **gcc** with the configuration file
:svn:`~doc/install/Linux/customize_juropa_gcc.py`.

.. literalinclude:: customize_juropa_gcc.py

and by executing::

  module unload parastation/intel
  module load parastation/gcc

  python setup.py install --prefix='' --home=MY_INSTALLATION_DIR

Building GPAW with Intel compiler
=================================

Use the compiler wrapper file :svn:`~doc/install/Linux/icc.py`

.. literalinclude:: icc.py

and the configuration file :svn:`~doc/install/Linux/customize_juropa_icc.py`.

.. literalinclude:: customize_juropa_icc.py

Now, default parastation/intel module is used so execute only::

  python setup.py install --prefix='' --home=MY_INSTALLATION_DIR

Execution
=========

General execution instructions can be found at `<http://www.fz-juelich.de/jsc/juropa/usage/quick-intro>`_.

Example batch job script for GPAW (512 cores, 30 minutes)::

  #!/bin/bash -x
  #MSUB -l nodes=64:ppn=8
  #MSUB -l walltime=0:30:00
  
  cd $PBS_O_WORKDIR
  export PYTHONPATH="MY_INSTALLATION_DIR/ase/lib64/python"
  export PYTHONPATH="$PYTHONPATH":"MY_INSTALLATION_DIR/gpaw/svn/lib64/python"
  export GPAW_SETUP_PATH=SETUP_DIR/gpaw-setups-0.5.3574
  export GPAW_PYTHON=MY_INSTALLATION_DIR/bin/gpaw-python

  export PSP_ONDEMAND=1

  mpiexec -np 512 -x $GPAW_PYTHON my_input.py --sl_default=4,4,64

Note that **-x** flag for `mpiexec` is needed for exporting the environment 
variables to MPI tasks. The environment variable ``PSP_ONDEMAND`` can decrease 
the running time with almost a factor of two with large process counts!

Job scripts can be written also using::

  gpaw-runscript -h
