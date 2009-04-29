.. _using_TAU_on_surveyor:

=====================
Using TAU on surveyor
=====================

gcc
===

This applies to surveyor and intrepid, regardless of doing either
manual or automatic instrumentation.

Add the following lines to you ``.softenvrc``. Note that here we specify
``ppc64`` as the ``TAUARCHITECTURE``. This will allow us to run
``paraprof`` on the front-end nodes::

  TAUARCHITECTURE = ppc64
  TAUVERSION = 2.18.1
  TAU_MAKEFILE = /soft/apps/tau/tau-$TAUVERSION/bgp/lib/Makefile.tau-bgptimers-mpi-gnu-compensate-python-pdt
  TAU_OPTIONS = '-optTau="-rn Py_RETURN_NONE"  -i/soft/apps/tau/tau-'$TAUVERSION'/include/TAU_PYTHON_FIX.h"'
  PATH += /soft/apps/tau/tau-$TAUVERSION/$TAUARCHITECTURE/bin

The bindings are located in
``/soft/apps/tau/tau/tau-$TAUVERSION/bgp/lib/<name>``.  This particular TAU library binding supports BGP timers (a low-level
timer with minimal overhead), MPI, GNU compiler, Python, and compensation. This is the recommended library binding for
flat profiles.

You will also need to change one line in your :svn:`bgp_gcc.py`::

  cmd = "tau_cc.sh %s %s"%(flags, cmd)
  
xlc
===

Follow the instructions for **gcc** (above), except:

* use filenames and directories that do not contain *gnu* in their name,
* There is no need to include TAU_PYTHON_FIX.h in *TAU_OPTIONS*,
* At the moment the ``-optShared`` is not working with *bgptimers* binding libraries.


automatic instrumentation
==========================

Overhead with the 2.18.1 version of TAU has been measured to be about
20% for the `b256H2O.py
<https://svn.fysik.dtu.dk/projects/gpaw/doc/devel/256H2O/b256H2O.py>`_
test case. Use manual instrumentation if this overhead is unacceptable.

Submitting jobs
==================

If you are doing manual instrumentation, the following environment variables must be append and passed to Cobalt via the qsub command::

  PYTHONPATH=/soft/apps/tau/tau_latest/bgp/lib/bindings-bgptimers-mpi-gnu-compensate-python-pdt
  LD_LIBRARY_PATH=/soft/apps/tau/tau_latest/bgp/lib/bindings-bgptimers-mpi-gnu-compensate-python-pdt

For automatica instrumentation, also add::

  LD_LIBRARY_PATH=/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib:/bgsys/drivers/ppcfloor/comm/lib  

A typically ``qsub`` commands looks like this::

  qsub -A Gpaw -n $nodes -t $time -q $queue --mode $mode --env BG_MAPPING=$mapping:MPIRUN_ENABLE_TTY_REPORTING=0:OMP_NUM_THREADS=1:GPAW_SETUP_PATH=$GPAW_SETUP_PATH:PYTHONPATH=/home/naromero/ase:/home/naromero/gpaw-tau:/soft/apps/tau/tau_latest/bgp/lib/bindings-bgptimers-mpi-gnu-compensate-python-pdt:$PYTHONPATH:LD_LIBRARY_PATH=/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib:/bgsys/drivers/ppcfloor/comm/lib:/soft/apps/tau/tau_latest/bgp/lib/bindings-bgptimers-mpi-gnu-compensate-python-pdt:$LD_LIBRARY_PATH /home/naromero/gpaw-tau/build/bin.linux-ppc64-2.5/gpaw-python ./$input --sl_inverse_cholesky=4,4,64,4 --sl_diagonalize=4,4,64,4 --domain-decomposition=4,4,4 

If you are doing manual instrumentation, simply then the the true input files is passed to ``gpaw-python``. For automatic instrumentation, you need to submit ``wrapper.py`` instead::

  import tau

  def OurMain():
      import CH4;

  tau.run('OurMain()')

This TAU run will produce ``profile.*`` files that can be merged into
the default TAU's ``ppk`` format using the command issued from the directory
where the ``profile.*`` files reside::

 paraprof --pack CH4.ppk

The actual analysis can be made on a different machine, by transferring
the ``CH4.ppk`` file from ``surveyor``, installing TAU, and launching::

 paraprof CH4.ppk
