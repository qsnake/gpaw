.. _Niflheim:

========
Niflheim
========

The quick step-by-step
----------------------

First time:

1. Run: ``svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk
   gpaw``, where `gpaw` should be a directory on the niflheim file
   server.

2. Replace the file gpaw/customize.py by
   :svn:`~doc/install/Linux/Niflheim/customize_ethernet.py` (if you
   want to run gpaw on the infiniband nodes, you should use
   :svn:`~doc/install/Linux/Niflheim/customize_infiniband.py` instead).

3. ssh to the login node ``slid`` and go to the gpaw directory.

4. Run ``python setup.py build_ext``

5. Add your gpaw directory to the ``PYTHONPATH`` environment variable,
   and your setups directory to the ``GPAW_SETUP_PATH`` environment
   variable

6. When submitting jobs, use the file
   :svn:`~doc/documentation/parallel_runs/gpaw-qsub` instead of the
   usual qsub. (usage is exactly the same).


When updating the gpaw code in the future:

1. Go to the gpaw directory and run ``svn up``.

2. If any of the c-code changed during the update, log on to ``slid`` and run:
   
   a. ``python setup.py clean``

   b. ``python setup.py build_ext``


Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

Please follow :ref:`developer_installation`.
The detailed settings are given below.

opteron ethernet nodes
======================

On the login node ``slid`` build GPAW (``python setup.py build_ext``)
with gcc compiler using the following :file:`customize.py` file:

.. literalinclude:: customize_ethernet.py

opteron infiniband nodes
========================

A subset of the Niflheim's nodes is equipped with Infiniband network
`<https://wiki.fysik.dtu.dk/niflheim/Hardware#infiniband-network>`_
and denoted by ``infiniband`` batch system property.

On the login node ``slid`` build GPAW (``python setup.py build_ext``)
with gcc compiler using the following :file:`customize.py` file:

.. literalinclude:: customize_infiniband.py

You can alternatively build on ``slid`` build GPAW (``python setup.py
build_ext``) with pathcc (pathcc looks ~3% slower - check other jobs!)
compiler using the following :file:`customize.py` file::

.. literalinclude:: customize_infiniband_pathcc.py

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
