.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

Preferably use the system default installation of GPAW and setups:
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

       setenv GPAW_TRUNK https://svn.fysik.dtu.dk/projects/gpaw/trunk
       setenv GPAW_HOME /home/niflheim/$USER/gpaw

    - using bash::

       export GPAW_TRUNK=https://svn.fysik.dtu.dk/projects/gpaw/trunk
       export GPAW_HOME=/home/niflheim/$USER/gpaw

   Checkout the GPAW source (make **sure** that
   :file:`${GPAW_HOME}/gpaw` does **not** exist, before running the checkout!)::

    svn checkout $GPAW_TRUNK ${GPAW_HOME}

   You may consider adding :envvar:`GPAW_TRUNK` and :envvar:`GPAW_HOME` to
   :file:`/home/niflheim/$USER/.cshrc` (:file:`/home/niflheim/$USER/.bashrc`).

   **Note**: that if you are doing a heavy development (many svn checkins)
   you may consider installing a special development version on workstation's
   local disk (faster), i.e. ``GPAW_HOME=/scratch/$USER``, however this version will
   not be accesible from Niflheim.

2. To compile the code, run the shell script
:svn:`~doc/install/Linux/Niflheim/compile.sh`:

.. literalinclude:: compile.sh

If you have login passwords active,
this will force you to type your password four times. It is
possible to remove the need for typing passwords on internal CAMd systems,
using the procedure described at
https://wiki.fysik.dtu.dk/it/SshWithoutPassword.

3. **Prepend** :envvar:`PYTHONPATH` environment variable:

   - csh/tcsh - add to /home/niflheim/$USER/.cshrc::

       source /home/camp/modulefiles.csh
       if ( "`echo $FYS_PLATFORM`" == "AMD-Opteron-el4" ) then # slid
           if ( ! ( `hostname -s | grep "^p"` ==  "")) then # p-node = infiniband
               setenv GPAW_PLATFORM "linux-x86_64-infiniband-2.3"
           else
               setenv GPAW_PLATFORM "linux-x86_64-ethernet-2.3"
           endif
       endif
       if ( "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ) then # fjorm
           module load GPAW
           setenv GPAW_PLATFORM "linux-x86_64-opteron-2.4"
       endif
       if ( "`echo $FYS_PLATFORM`" == "Intel-Nehalem-el5" ) then # thul
           module load GPAW
           setenv GPAW_PLATFORM "linux-x86_64-xeon-2.4"
       endif
       setenv PATH ${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:${PATH}
       setenv PATH ${GPAW_HOME}/tools:${PATH}
       setenv PYTHONPATH ${GPAW_HOME}:${PYTHONPATH}
       setenv PYTHONPATH ${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:${PYTHONPATH}

   - bash - add to /home/niflheim/$USER/.bashrc::

       source /home/camp/modulefiles.sh
       if [ "`echo $FYS_PLATFORM`" == "AMD-Opteron-el4" ]; then # slid
           if [ ! `hostname -s | grep "^p"` ==  "" ]; then # p-node = infiniband
               export GPAW_PLATFORM="linux-x86_64-infiniband-2.3"
           else
               export GPAW_PLATFORM="linux-x86_64-ethernet-2.3"
           fi
       fi
       if [ "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ]; then # fjorm
           module load GPAW
           export GPAW_PLATFORM="linux-x86_64-opteron-2.4"
       fi
       if [ "`echo $FYS_PLATFORM`" == "Intel-Nehalem-el5" ]; then # thul
           module load GPAW
           export GPAW_PLATFORM="linux-x86_64-xeon-2.4"
       fi
       export PATH=${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:${PATH}
       export PATH=${GPAW_HOME}/tools:${PATH}
       export PYTHONPATH=${GPAW_HOME}:${PYTHONPATH}
       export PYTHONPATH=${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:${PYTHONPATH}

   Make sure that you add these settings above any line that
   causes exit when run in the batch system e.g. ``if ( { tty -s } == 0 ) exit``.
 
4. If you prefer to use a personal setup's directory follow
   point 4. from :ref:`installationguide`.

5. When submitting jobs to the batch system, use the file
   :svn:`~doc/documentation/parallel_runs/gpaw-qsub` instead of the
   usual :command:`qsub`.

When updating the gpaw code in the future:

- Go to the gpaw directory and run::

    svn up

- If any of the c-code changed during the update repeat step 2.

.. note::

  Please ask the Niflheim's support staff to verify that gpaw-python runs single-threaded, e.g. for a job running on ``p024`` do from ``audhumbla``::

    ssh p024 ps -fL

  Numbers higher then **1** in the **NLWP** column mean multi-threaded job.
