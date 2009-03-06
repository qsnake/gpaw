.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

Preferably use the system default installation of GPAW, and setups:
to be able to do so, please do **not**
overwrite the system default :envvar:`PATH`, :envvar:`PYTHONPATH`,
nor :envvar:`GPAW_SETUP_PATH` environment variables.
When setting the environment variables **prepend** them, i.e.:

 - using csh/tcsh::

    setenv PATH ${HOME}/bin:${PATH}

 - using bash::

    export PATH=${HOME}/bin:${PATH}

If you decide to install a development version of GPAW, this is what you do
when installating GPAW for the first time:

1. On the ``servcamd`` filesystem (login on your workstation)
   go to a directory on the Niflheim filesystem.
   Usually users install GPAW under Niflheim's :envvar:`HOME`,
   i.e. :file:`/home/niflheim/$USER`, and the instructions below assume this:

    - using csh/tcsh::

       set GPAW_TRUNK=https://svn.fysik.dtu.dk/projects/gpaw/trunk
       set GPAW_HOME=/home/niflheim/$USER

    - using bash::

       GPAW_TRUNK=https://svn.fysik.dtu.dk/projects/gpaw/trunk
       GPAW_HOME=/home/niflheim/$USER

   Checkout the GPAW source (make **sure** that
   :file:`$GPAW_HOME/gpaw` does **not** exist, before running the checkout!)::

    svn checkout $GPAW_TRUNK $GPAW_HOME/gpaw

   You may consider adding :envvar:`GPAW_TRUNK` and :envvar:`GPAW_HOME` to
   :file:`/home/niflheim/$USER/.cshrc` (:file:`/home/niflheim/$USER/.bashrc`).

   **Note**: that if you are doing a heavy development (many svn checkins)
   you may consider installing a special development version on workstation's
   local disk (faster), i.e. ``GPAW_HOME=/scratch/$USER``, however this version will
   not be accesible from Niflheim.

2a. Replace the file :file:`gpaw/customize.py` by :svn:`~doc/install/Linux/Niflheim/customize_ethernet.py`::

    wget --no-check-certificate $GPAW_TRUNK/doc/install/Linux/Niflheim/customize_ethernet.py -O $GPAW_HOME/gpaw/customize.py

2b. **Optional**: if you want to run gpaw on the infiniband nodes
   (p-nodes, see `<https://wiki.fysik.dtu.dk/niflheim/Hardware#infiniband-network>`_;
   **note**: that not all users are allowed to run on the p-nodes):

    - make another checkout designed for infiniband nodes::

       svn checkout $GPAW_TRUNK $GPAW_HOME/gpaw_p

    - Replace the file :file:`gpaw/customize.py` by :svn:`~doc/install/Linux/Niflheim/customize_infiniband.py`::

       wget --no-check-certificate $GPAW_TRUNK/doc/install/Linux/Niflheim/customize_infiniband.py -O $GPAW_HOME/gpaw_p/customize.py

3. ssh to the login node ``slid`` and go to the gpaw directory.

4. **Prepend** :envvar:`PYTHONPATH` environment variable:

   - csh/tcsh - add to /home/niflheim/$USER/.cshrc::

      set query=`hostname -s`
      # p (infiniband) nodes
      if ( ! ( `echo ${query} | grep "^p"` ==  "")) then
        setenv PYTHONPATH $GPAW_HOME/gpaw_p:$PYTHONPATH
        setenv PATH $GPAW_HOME/gpaw_p/tools:$PATH
        setenv PATH $GPAW_HOME/gpaw_p/build/bin.linux-x86_64-2.3:$PATH
      else
        setenv PYTHONPATH $GPAW_HOME/gpaw:$PYTHONPATH
        setenv PATH $GPAW_HOME/gpaw/tools:$PATH
        setenv PATH $GPAW_HOME/gpaw/build/bin.linux-x86_64-2.3:$PATH
      endif

   - bash - add to /home/niflheim/$USER/.bashrc::

      query=`hostname -s`
      # p (infiniband) nodes
      if [ ! `echo ${query} | grep "^p"` ==  "" ]; then
        export PYTHONPATH=$GPAW_HOME/gpaw_p:$PYTHONPATH
        export PATH=$GPAW_HOME/gpaw_p/tools:$PATH
        export PATH=$GPAW_HOME/gpaw_p/build/bin.linux-x86_64-2.3:$PATH
      else
        export PYTHONPATH=$GPAW_HOME/gpaw:$PYTHONPATH
        export PATH=$GPAW_HOME/gpaw/tools:$PATH
        export PATH=$GPAW_HOME/gpaw/build/bin.linux-x86_64-2.3:$PATH
      fi

   Make sure that you add these settings above any line that
   causes exit when run in the batch system e.g. ``if ( { tty -s } == 0 ) exit``.
 
5. Run::

    python setup.py build_ext

6. If you prefer to use a personal setup's directory follow
   point 4. from :ref:`installationguide`.

7. When submitting jobs to the batch system, use the file
   :svn:`~doc/documentation/parallel_runs/gpaw-qsub` instead of the
   usual :command:`qsub`.

When updating the gpaw code in the future:

1. Go to the gpaw directory and run::

    svn up

2. If any of the c-code changed during the update, log on to ``slid`` and run::
   
    python setup.py clean

    python setup.py build_ext

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
compiler using the following :file:`customize.py` file:

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
         $GPAW_HOME/gpaw_p/build/bin.linux-x86_64-2.4/gpaw-python gpaw-script.py

and for pathcc version looks like this::

  cd $PBS_O_WORKDIR
  export LD_LIBRARY_PATH=/opt/pathscale/lib/2.5
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/acml-4.0.1/pathscale64/lib
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/blacs-1.1-24.6.infiniband/lib64
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/scalapack-1.8.0-1.infiniband/lib64
  mpirun -machinefile $PBS_NODEFILE -np 4 \
         $GPAW_HOME/gpaw_p/build/bin.linux-x86_64-2.4/gpaw-python gpaw-script.py

Please make sure that the threads use 100% of CPU, e.g. for a job running on ``p024`` do from ``audhumbla``::

  ssh p024 ps -fL

Numbers higher then **1** in the **NLWP** column mean multi-threaded job.
