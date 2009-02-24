.. _murska:

=============
murska.csc.fi
=============

Here you find information about the the system
`<http://raketti.csc.fi/english/research/Computing_services/computing/servers/murska>`_.

We want to use python2.4 and gcc compiler::

  > module load python
  > module swap PrgEnv-pgi PrgEnv-gnu

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

