.. _using_TAU_on_surveyor:

=====================
Using TAU on surveyor
=====================

Start by reading the `main profiling page <https://wiki.fysik.dtu.dk/gpaw/devel/profiling.html>`_

Do **not** using the customize.py from the above page, following the instructions found on this
page instead. The following mostly applies to automatic instrumentation using the TAU compiler
scripts. 

Overhead with the TAU for automatic instrumentation has been measured ate about 20% for the
`b256H2O.py <https://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/256H2O/b256H2O.py>`_.
Use manual instrumentation if this overhead is unacceptable or you might need to 
add more functions to the existing selective instrumentation file
`select.tau <https://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/devel/profiling/select.tau>`_

The version of TAU at ALCF is updated frequently, please check:
`<https://wiki.alcf.anl.gov/index.php/Tuning_and_Analysis_Utilities_(TAU)>`_

GCC
===

This applies to surveyor and intrepid, regardless of doing either
manual or automatic instrumentation.

Add the following lines to you ``.softenvrc``. Note that here we specify
``ppc64`` as the ``TAUARCHITECTURE``. This will allow us to run
``paraprof`` on the front-end nodes::

  TAUARCHITECTURE = ppc64
  TAUVERSION = 2.18.2p2
  TAU_MAKEFILE = /soft/apps/tau/tau-$TAUVERSION/bgp/lib/Makefile.tau-bgptimers-gnu-mpi-python-pdt
  TAU_OPTIONS = '-optVerbose -optShared -optTauSelectFile=select.tau \
  	      -optTau="-rn Py_RETURN_NONE -i/soft/apps/tau/tau-'$TAUVERSION'/include/TAU_PYTHON_FIX.h"'
  PATH += /soft/apps/tau/tau-$TAUVERSION/$TAUARCHITECTURE/bin

The biggest difference between 2.18.2 and 2.18.1, is that many Makefile stub configurations are now available as run time options.

The bindings are located in
``/soft/apps/tau/tau/tau-$TAUVERSION/bgp/lib/<name>``.  This particular TAU library binding supports BGP timers (a low-level
timer with minimal overhead), MPI, GNU compiler, and Python. This is the recommended library binding for flat profiles.

You will also need to change one line in your :svn:`bgp_gcc.py`::

  cmd = "tau_cc.sh %s %s"%(flags, cmd)
  
XLC
===

TAU does not work with XLC for automatic instrumentation because of an issue related to the bgptimers shared library. IBM is working
towards a resolution this fall.

Run time environment variables
================================
Please see:
https://wiki.alcf.anl.gov/index.php/Tuning_and_Analysis_Utilities_(TAU)#Running_With_TAU

Here are the recommended run time environment variables that should be passed to Cobalt via qsub::

  TAU_VERBOSE=1:TAU_THROTTLE=0:TAU_COMPENSATE=1

TAU_COMPENSATE seems to cause problems with manual instrumentation, so do not set it to 0 which
means off. In any case, it should not be particularly relevant unless you have manual timers on a
frequently accessed lightweight functions.

Submitting jobs
==================

The following environment variables must be append and passed to Cobalt via the qsub command::

  PYTHONPATH=/soft/apps/tau/tau-$TAUVERSION/bgp/lib/bindings-bgptimers-mpi-gnu-python-pdt
  LD_LIBRARY_PATH=/soft/apps/tau/tau-$TAUVERSION/bgp/lib/bindings-bgptimers-mpi-gnu-python-pdt

A typically ``qsub`` commands looks like this::

  qsub -A $account -n $nodes -t $time -q $queue --mode $mode \
       --env BG_MAPPING=$mapping:MPIRUN_ENABLE_TTY_REPORTING=0:OMP_NUM_THREADS=1: \
       GPAW_SETUP_PATH=$GPAW_SETUP_PATH:\
       PYTHONPATH=$PYTHONPATH:/soft/apps/tau/tau-$TAUVERSION/bgp/lib/bindings-bgptimers-mpi-gnu-python-pdt: \
       LD_LIBRARY_PATH=$CN_LIBRARY_PATH:/soft/apps/tau/tau-$TAUVERSION/bgp/lib/bindings-bgptimers-mpi-gnu-python-pdt \
       $HOME/gpaw-tau/build/bin.linux-ppc64-2.5/gpaw-python wrapper.py

If you are doing manual instrumentation, simply pass the actual input file to ``gpaw-python`` instead. For automatic instrumentation, you need to ``wrapper.py`` instead::

  import tau

  def OurMain():
      import CH4;

  tau.run('OurMain()')

TAU run will then produce ``profile.*`` files that can be merged into
the default TAU's ``ppk`` format using the command issued from the directory
where the ``profile.*`` files reside::

 paraprof --pack CH4.ppk

The actual analysis can be made on a different machine, by transferring
the ``CH4.ppk`` file from ``surveyor``, installing TAU, and launching::

 paraprof CH4.ppk
