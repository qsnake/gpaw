.. _newNiflheim:

======================
The new Niflheim nodes
======================

Niflheim has recently been upgraded by a new type of nodes.  This page
describes How to do a simultaneous build on all the Niflheim
architecture types.

In all of the folowing, let :envvar:`GPAW` denote your gpaw base directory.

You need to have the newest
:svn:`~doc/documentation/parallel_runs/gpaw-qsub` script in your
:envvar:`PATH`.

You should put the following four :file:`customize.py` files in your
:envvar:`GPAW` directory:

* :svn:`~doc/install/Linux/Niflheim/new/customize-slid-ethernet.py`
* :svn:`~doc/install/Linux/Niflheim/new/customize-slid-infiniband.py`
* :svn:`~doc/install/Linux/Niflheim/new/customize-fjorm.py`
* :svn:`~doc/install/Linux/Niflheim/new/customize-thul.py`

To compile the code, you should run the shell script
:svn:`~doc/install/Linux/Niflheim/new/compile.sh`:

.. literalinclude:: compile.sh

Here exemplified using tc shell commands.  If you have login passwords
active, this will force you to type your password four times. It is
highly recommended to remove the need for typing passwords on internal
CAMD ssh, using the procedure described at
https://wiki.fysik.dtu.dk/it/SshWithoutPassword.

Since some of the node types have the same architecture, gpaw's
automatic detection of the gpaw/build/ directory does not
work. Therefore you need to specify this manually.

If you use a tc shell, the following should be put in the top of your
:file:`/home/niflheim/$USER/.cshrc` file::

  source /home/camp/modulefiles.csh
  if ( "`echo $FYS_PLATFORM`" == "AMD-Opteron-el4" ) then # slid
      if ( ! ( `hostname -s | grep "^p"` ==  "")) then # p-node = infiniband
          setenv PATH $GPAW/build/bin.linux-x86_64infiniband-2.3:$PATH
          setenv PYTHONPATH $GPAW/build/lib.linux-x86_64infiniband-2.3:$PYTHONPATH
      else
          setenv PATH $GPAW/build/bin.linux-x86_64ethernet-2.3:$PATH
          setenv PYTHONPATH $GPAW/build/lib.linux-x86_64ethernet-2.3:$PYTHONPATH
      endif
  endif
  if ( "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ) then # fjorm
      module load GPAW
      setenv PATH $GPAW/build/bin.linux-x86_64amd-2.4:$PATH
      setenv PYTHONPATH $GPAW/build/lib.linux-x86_64amd-2.4:$PYTHONPATH
  endif
  if ( "`echo $FYS_PLATFORM`" == "Intel-Nehalem-el5" ) then # thul
      module load GPAW
      setenv PATH $GPAW/build/bin.linux-x86_64intel-2.4:$PATH
      setenv PYTHONPATH $GPAW/build/lib.linux-x86_64intel-2.4:$PYTHONPATH
  endif

You should now be able to submit to all possible Niflheim nodes.
