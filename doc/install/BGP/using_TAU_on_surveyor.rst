.. _using_TAU_on_surveyor:

=====================
Using TAU on surveyor
=====================

This applies to surveyor and intrepid, regardless of doing either
manual or automatic instrumentation.

Add the following lines to you ``.softenvrc``. Note that here we specify
``ppc64`` as the ``TAUARCHITECTURE``. This will allow us to run
``paraprof`` on the front-end nodes::

  TAUARCHITECTURE = ppc64
  TAUVERSION = 2.18.1
  TAU_OPTIONS = '-optShared -optTau="-rn Py_RETURN_NONE" ' 
  PATH += /soft/apps/tau/tau-$TAUVERSION/$TAUARCHITECTURE/bin

Because we have chosen ``-optShared`` as an TAU option we can
choose the TAU library binding at runtime by specifying the
appropriate directory with the ``LD_LIBRARY_PATH`` environment
variable. The bindings can be found under
``/soft/apps/tau/tau/tau-$TAUVERSION/bgp/lib/<name>``. Also note that
this environment variable must be passed to the Cobalt, the job scheduler, via ``qsub``, e.g.:: 

  qsub <args> --env
  <env1>:<env2>:<env3>...LD_LIBRARY_PATH=$LD_LIBRARY_PATH/soft/apps/tau/tau-$TAUVERSION/bgp/lib/bindings-bgptimers-mpi-gnu-compensate-python-pdt

This particular TAU library binding supports BGP timers (a low-level
timer with minimal overhead), MPI, GNU compiler, Python, and the
metric compensation. This is the recommended library binding for
flat profiles.
  

