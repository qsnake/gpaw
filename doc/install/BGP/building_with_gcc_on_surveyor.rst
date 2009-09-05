.. _building_with_gcc_on_surveyor:

==========================
Building with gcc compiler
==========================

NumPy
=====

If you do not wish to build NumPy for yourself, you can use one of the following versions::

  /home/dulak/numpy-1.0.4-1
  /home/naromero/numpy-1.0.4-1.essl
  /home/naromero/numpy-1.0.4-1.goto

Instructions for the ``V1R3M0_460_2008-081112P`` driver,
compilation with gcc compiler.

The **0.3** version of gpaw uses Numeric `<https://svn.fysik.dtu.dk/projects/gpaw/tags/0.3/>`_.

Get the Numeric-24.2 (**only** if you want to run the **0.3** version of gpaw)
and do this::

  $ wget http://downloads.sourceforge.net/numpy/Numeric-24.2.tar.gz
  $ gunzip -c Numeric-24.2.tar.gz | tar xf -
  $ cd Numeric-24.2
  $ /bgsys/drivers/ppcfloor/gnu-linux/bin/python setup.py install --root=$HOME/Numeric-24.2-1

The latest version of gpaw uses numpy `<https://svn.fysik.dtu.dk/projects/gpaw/trunk/>`_.

To build an optimized numpy for the compute nodes, based on ``goto`` blas, save the :svn:`~doc/install/BGP/numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran`
patch file
(modifications required to get powerpc-bgp-linux-gfortran instead of
gfortran compiler),
the :svn:`~doc/install/BGP/numpy-1.0.4-system_info.py.patch.lapack_bgp_goto_esslbg` patch file (lapack
section configured to use ``lapack_bgp`` and
blas section to use ``goto``, ``cblas_bgp``, and ``esslbg``),
and the :svn:`~doc/install/BGP/numpy-1.0.4-site.cfg.lapack_bgp_goto_esslbg` file (contains paths to
``lapack_bgp``, ``goto``, ``esslbg`` , ``cblas_bgp``,
and xlf* related libraries).

**Note** that ``lapack_bgp`` and ``cblas_bgp`` are not available on ``surveyor/intrepid``, to build use instructions from `<http://www.pdc.kth.se/systems_support/computers/bluegene/LAPACK-CBLAS/LAPACK-CBLAS-build>`_. Python requires all libraries to have names like ``liblapack_bgp.a``, so please make the required links for ``lapack_bgp.a``, and ``cblas_bgp.a``. Moreover numpy requires that ``lapack_bgp``, ``goto``, ``esslbg``, and ``cblas_bgp`` reside in the same directory, so choose a directory and edit ``numpy-1.0.4-site.cfg.lapack_bgp_goto_esslbg`` to reflect your installation path (in this example `/home/dulak/from_Nils_Smeds/CBLAS_goto/lib/bgp`). Include the directory containing `cblas.h` in `include_dirs`. Change the locations of the libraries to be used in the makefiles: `/soft/apps/LIBGOTO` and `/opt/ibmcmp/lib/bg`.

**Warning** : If NumPy built using these libraries fails
with errors of kind "R_PPC_REL24 relocation at 0xa3d664fc for symbol sqrt"
- please add ``-qpic`` to compile options for both ``lapack_bgp`` and ``cblas_bgp``. 
After building ``lapack_bgp`` and ``cblas_bgp``, get numpy-1.0.4 and do this::

  $ wget http://downloads.sourceforge.net/numpy/numpy-1.0.4.tar.gz
  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ mv numpy-1.0.4 numpy-1.0.4.optimized; cd numpy-1.0.4.optimized
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
  $ patch -p1 < ../numpy-1.0.4-system_info.py.patch.lapack_bgp_goto_esslbg
  $ cp ../numpy-1.0.4-site.cfg.lapack_bgp_goto_esslbg site.cfg
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ ldflags="-Wl,--allow-multiple-definition -L/opt/ibmcmp/xlmass/bg/4.4/bglib"
  $ root=$HOME/numpy-1.0.4-1.optimized
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc -DNO_APPEND_FORTRAN -L/opt/ibmcmp/xlmass/bg/4.4/bglib\""
  $ MATHLIB="mass" LDFLAGS="$ldflags" LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

NumPy built in this way does contain the
:file:`$root/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/_dotblas.so`
, and running the following python script results
in better time than the standard version of numpy (~156 vs. ~329 sec)
for ``numpy.dot`` operation (:svn:`~doc/install/BGP/numpy_dot.py`):

.. literalinclude:: numpy_dot.py

To build standard numpy, save the :svn:`~doc/install/BGP/numpy-1.0.4-gnu.py.patch` patch file
(modifications required to get mpif77 instead of gfortran compiler),
get and numpy-1.0.4 and do this::

  $ wget http://downloads.sourceforge.net/numpy/numpy-1.0.4.tar.gz
  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ cd numpy-1.0.4
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ root=$HOME/numpy-1.0.4-1
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"mpicc\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

**Note**: instructions may work also for numpy version *1.3.0*.

Build python-nose::

  $ wget http://python-nose.googlecode.com/files/nose-0.11.0.tar.gz
  $ tar zxf nose-0.11.0.tar.gz
  $ cd nose-0.11.0
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ $p setup.py install --root=${HOME}/python-nose-0.11.0-1 2>&1 | tee install.log

GPAW
====

Step 1
======

Download all the necessary packages:

- `ase <https://wiki.fysik.dtu.dk/ase/download.html#latest-stable-release>`_

- `gpaw <https://wiki.fysik.dtu.dk/gpaw/download.html#latest-stable-release>`_

- `gpaw-setups <https://wiki.fysik.dtu.dk/gpaw/setups/setups.html>`_

Step 2
======

Set these environment variables in the :file:`.softenvrc` file::

  PYTHONPATH += $HOME/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
  PYTHONPATH += ${HOME}/python-nose-0.11.0-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
  PYTHONPATH += ${HOME}/ase3k
  PYTHONPATH += $HOME/gpaw
  PYTHONPATH += $HOME/gpaw/build/lib.linux-ppc64-2.5
  GPAW_SETUP_PATH = $HOME/gpaw-setups

  LD_LIBRARY_PATH += /bgsys/drivers/ppcfloor/gnu-linux/lib
  # Compute node
  # qsub --env LD_LIBRARY_PATH=$CN_LD_LIBRARY_PATH
  # some of these are only relevant for gcc, xlc, or TAU
  CN_LD_LIBRARY_PATH = /bgsys/drivers/ppcfloor/runtime/SPI
  CN_LD_LIBRARY_PATH = /opt/ibmcmp/xlf/bg/11.1/bglib:${CN_LD_LIBRARY_PATH}
  CN_LD_LIBRARY_PATH = /opt/ibmcmp/xlsmp/bg/1.7/bglib:${CN_LD_LIBRARY_PATH}
  CN_LD_LIBRARY_PATH = /opt/ibmcmp/lib/bg/bglib:${CN_LD_LIBRARY_PATH}
  CN_LD_LIBRARY_PATH = /bgsys/drivers/ppcfloor/gnu-linux/lib:${CN_LD_LIBRARY_PATH}
  CN_LD_LIBRARY_PATH = /bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib:${CN_LD_LIBRARY_PATH}
  CN_LD_LIBRARY_PATH = /bgsys/drivers/ppcfloor/comm/lib:${CN_LD_LIBRARY_PATH}

  PATH += $HOME/ase3k/tools
  PATH += $HOME/gpaw/tools
  PYTHONPATH += $HOME/gpaw/build/bin.linux-ppc64-2.5

and type::

  resoft

to update your environment in the main login terminal.

Step 3
======

Because the ``popen3`` function is missing, you will need to remove all the
contents of the :file:`gpaw/version.py` file
after ``ase_required_svnrevision =``.
The same holds for :file:`ase/version.py` in the ase installation!
Suggestions how to skip the ``popen3`` testing in
:file:`gpaw/version.py` on BG/P are welcome!

Step 4
======

A number of the GPAW source files in ``gpaw/c`` directory are built using
the ``distutils`` module which makes it difficult to control the flags
which are passed to the gnu compiler.
A workaround is to use the following python
script: :svn:`~doc/install/BGP/bgp_gcc.py`
with the corresponding :svn:`~doc/install/BGP/customize_surveyor_gcc.py` file.

.. literalinclude:: bgp_gcc.py

.. literalinclude:: customize_surveyor_gcc.py

Download these scripts into the top level GPAW directory::

  export GPAW_TRUNK=http://svn.fysik.dtu.dk/projects/gpaw/trunk
  wget --no-check-certificate ${GPAW_TRUNK}/doc/install/BGP/bgp_gcc.py
  chmod u+x bgp_gcc.py
  wget --no-check-certificate ${GPAW_TRUNK}/doc/install/BGP/customize_surveyor_gcc.py

Finally, we build GPAW by typing::

  /bgsys/drivers/ppcfloor/gnu-linux/bin/python setup.py build_ext --ignore-numpy --customize=customize_surveyor_gcc.py 2>&1 | tee build_ext.log

If an optimized version of NumPy is in your $PYTHONPATH you may need to use "--ignore-numpy".

Additional BG/P specific hacks
===============================

A FLOPS (floating point per second) counter and a number of other hardware counters can be enabled with the macro::

  define_macros += [('GPAW_HPM',1)]

This hpm library is available on the BG/P machines at Argonne National Laboratory. It will produce two files for each core: ``hpm_flops.$rank`` and ``hpm_data.$rank``. The latter one contains a number of additional hardware counters. There are four cores per chip and data for only two of the four cores can be collected simultaneously. This is set through an environment variable which is passed to Cobalt with the *--env*  flag. *BGP_COUNTER_MODE=0* specifies core 1 and 2, while *BGP_COUNTER_MODE=1* specifies core 3 and 4. 

A mapfile for the ranks can be generated by adding another macro to customize.py::
  
  define_macros += [('GPAW_MAP',1)]


Submitting jobs
==================

Test numpy with::

  echo "import numpy; numpy.test()" > test_numpy.py
  qsub -n 1 -t 10 --mode smp --env \
       OMP_NUM_THREADS=1:GPAW_SETUP_PATH=$GPAW_SETUP_PATH:PYTHONPATH=$PYTHONPATH:LD_LIBRARY_PATH=$CN_LD_LIBRARY_PATH \
       /bgsys/drivers/ppcfloor/gnu-linux/bin/python test_numpy.py

A gpaw script ``CH4.py`` (fetch it from ``gpaw/test``) can be submitted like this::

  qsub -n 2 -t 10 --mode vn --env \
       OMP_NUM_THREADS=1:GPAW_SETUP_PATH=$GPAW_SETUP_PATH:PYTHONPATH=$PYTHONPATH:LD_LIBRARY_PATH=$CN_LD_LIBRARY_PATH \
       $HOME/gpaw/build/bin.linux-ppc64-2.5/gpaw-python CH4.py

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.
