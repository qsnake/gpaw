.. _building_with_gcc_on_surveyor:

==========================
Building with gcc compiler
==========================

Instructions for the ``V1R3M0_460_2008-081112P`` driver,
compilation with gcc compiler.

**Warning** - for this driver two versions of numpy need to be compiled for
the compute nodes: a standard one used during the building process of gpaw,
and the optimized one used for running gpaw jobs. Using optimized numpy
for building of gpaw results in "ImportError: /opt/ibmcmp/lib/bg/bglib/libxlsmp.so.1: undefined symbol: __process_count" when importing numpy on the frontend node.

The **0.3** version of gpaw uses Numeric `<https://svn.fysik.dtu.dk/projects/gpaw/tags/0.3/>`_.

Get the Numeric-24.2 (**only** if you want to run the **0.3** version of gpaw)
and do this::

  $ wget http://downloads.sourceforge.net/numpy/Numeric-24.2.tar.gz
  $ gunzip -c Numeric-24.2.tar.gz | tar xf -
  $ cd Numeric-24.2
  $ /bgsys/drivers/ppcfloor/gnu-linux/bin/python setup.py install --root=$HOME/Numeric-24.2-1

The latest version of gpaw uses numpy `<https://svn.fysik.dtu.dk/projects/gpaw/trunk/>`_.

To build an optimized numpy for the compute nodes, based on ``goto`` blas, save the :svn:`numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran <doc/install/BGP/numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran>`
patch file
(modifications required to get powerpc-bgp-linux-gfortran instead of
gfortran compiler),
the :svn:`numpy-1.0.4-system_info.py.patch.lapack_bgp_goto_esslbg <doc/install/BGP/numpy-1.0.4-system_info.py.patch.lapack_bgp_goto_esslbg>` patch file (lapack
section configured to use ``lapack_bgp`` and
blas section to use ``goto``, ``cblas_bgp``, and ``esslbg``),
and the :svn:`numpy-1.0.4-site.cfg.lapack_bgp_esslbg <doc/install/BGP/numpy-1.0.4-site.cfg.lapack_bgp_goto_esslbg>` file (contains paths to
``lapack_bgp``, ``goto``, ``esslbg`` , ``cblas_bgp``,
and xlf* related libraries).

**Note** that ``lapack_bgp`` and ``cblas_bgp`` are not available on ``surveyor/intrepid``, to build use instructions from `<http://www.pdc.kth.se/systems_support/computers/bluegene/LAPACK-CBLAS/LAPACK-CBLAS-build>`_. Python requires all librairies to have names like ``liblapack_bgp.a``, so please make the required links for ``lapack_bgp.a``, and ``cblas_bgp.a``. Moreover numpy requires that ``lapack_bgp``, ``goto``, ``esslbg``, and ``cblas_bgp`` reside in the same directory, so choose a directory and edit ``numpy-1.0.4-site.cfg.lapack_bgp_goto_esslbg`` to reflect your installation path (in this example `/home/dulak/from_Nils_Smeds/CBLAS_goto/lib/bgp`). Include the directory containing `cblas.h` in `include_dirs`. Change the locations of the libraries to be used in the makefiles: `/soft/apps/LIBGOTO` and `/opt/ibmcmp/lib/bg`.

**Warning**: - crashes have been reported with numpy compiled against ``goto``
so build numpy agains ``esslbg`` - see :ref:`rbgc`.

**Warning** - if numpy built using these libraries fails
with errors of kind "R_PPC_REL24 relocation at 0xa3d664fc for symbol sqrt"
- please add ``-qpic`` to compile options for both ``lapack_bgp`` and ``cblas_bgp``. 
After bulding ``lapack_bgp`` and ``cblas_bgp``, get numpy-1.0.4 and do this::

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

Numpy built in this way does contain the
:file:`$root/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/_dotblas.so`
, and running the following python script results
in better time than the standard version of numpy (~156 vs. ~329 sec)
for ``numpy.dot`` operation::

  num_string = "numpy"
  #num_string = "Numeric"

  if num_string == "numpy":
      import numpy as num
  elif num_string == "Numeric":
      import Numeric as num
  print num.__file__

  from time import time

  import random

  N = 1700

  A = num.array(num.ones((N,N)))
  Al = A.tolist()
  for item in Al:
      for n,value in enumerate(item):
          if (n % 2) == 0:
              item[n] = random.random()
  Anew = num.array([Al])

  t = time()
  num.dot(Anew, Anew)
  print num_string, time()-t

To build standard numpy, save the :svn:`numpy-1.0.4-gnu.py.patch <doc/install/BGP/numpy-1.0.4-gnu.py.patch>` patch file
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

Set these environment variables in the :file:`.softenvrc` file::

  PYTHONPATH = ${HOME}/Numeric-24.2-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/Numeric
  PYTHONPATH += ${HOME}/numpy-1.0.4-1.optimized/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
  PYTHONPATH += ${HOME}/gpaw:${HOME}/CamposASE2:${HOME}/ase3k
  GPAW_SETUP_PATH = ${HOME}/gpaw-setups-0.4.2039

  LD_LIBRARY_PATH += /bgsys/drivers/ppcfloor/runtime/SPI
  LD_LIBRARY_PATH += /opt/ibmcmp/xlf/bg/11.1/bglib:/opt/ibmcmp/lib/bglib
  LD_LIBRARY_PATH += /opt/ibmcmp/xlsmp/bg/1.7/bglib:/bgsys/drivers/ppcfloor/gnu-linux/lib
  PATH += ${HOME}/gpaw/tools:${HOME}/CamposASE2/tools:${HOME}/ase3k/tools
  # to enable TAU profiling add also:
  PYTHONPATH += /soft/apps/tau/tau_latest/bgp/lib/bindings-mpi-gnu-python-pdt
  LD_LIBRARY_PATH += /soft/apps/tau/tau_latest/bgp/lib/bindings-mpi-gnu-python-pdt

and do::

  resoft

(to enable TAU profiling do also ``source /soft/apps/tau/tau.bashrc`` or ``soft add +tau``, if available),
and build GPAW (``PYTHONPATH=~/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages /bgsys/drivers/ppcfloor/gnu-linux/bin/python
setup.py build_ext``) with this :file:`customize.py` file::

  scalapack = True

  extra_compile_args += [
      '-O3'
      ]

  libraries = [
             'lapack_bgp',
             'scalapack',
             'blacsCinit_MPI-BGP-0',
             'blacs_MPI-BGP-0',
             'lapack_bgp',
             'goto',
             'xlf90_r',
             'xlopt',
             'xl',
             'xlfmath',
             'xlsmp'
             ]

  library_dirs = [
             '/soft/apps/LAPACK',
             '/soft/apps/LIBGOTO',
             '/soft/apps/BLACS',
             '/soft/apps/SCALAPACK',
             '/opt/ibmcmp/xlf/bg/11.1/bglib',
             '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
             '/bgsys/drivers/ppcfloor/gnu-linux/lib'
             ]

  gpfsdir = '/home/dulak'
  python_site = 'bgsys/drivers/ppcfloor/gnu-linux'

  include_dirs += [gpfsdir+'/Numeric-24.2-1/'+python_site+'/include/python2.5',
                   gpfsdir+'/numpy-1.0.4-1.optimized/'+python_site+'/lib/python2.5/site-packages/numpy/core/include']

  extra_compile_args += ['-std=c99']

  define_macros += [
            ('GPAW_AIX', '1'),
            ('GPAW_MKL', '1'),
            ('GPAW_BGP', '1')
            ]

  # uncomment the following lines to enable TAU profiling
  # tau_path = '/soft/apps/tau/tau_latest/bgp/'
  # tau_make = tau_path+'lib/Makefile.tau-mpi-gnu-python-pdt'
  # extra_compile_args += ['''-tau_options="-optShared -optTau='-rn Py_RETURN_NONE -i /soft/apps/tau/tau_latest/include/TAU_PYTHON_FIX.h' -optVerbose"''']
  # mpicompiler = "tau_cc.sh -tau_makefile="+tau_make
  # mpilinker = mpicompiler
  # compiler = mpicompiler

Because of missing ``popen3`` function you need to remove all the
contents of the :file:`gpaw/version.py` file after ``version =
'0.4'``.  The same holds for :file:`ase/version.py` in the ase
installation!  Suggestions how to skip the ``popen3`` testing in
:file:`gpaw/version.py` on BGP are welcome!

A gpaw script ``CH4.py`` (fetch it from ``gpaw/test``) can be submitted like this::

  qsub -n 2 -t 10 --mode vn --env \
       OMP_NUM_THREADS=1:GPAW_SETUP_PATH=$GPAW_SETUP_PATH:PYTHONPATH=$PYTHONPATH:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib:LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
       ${HOME}/gpaw/build/bin.linux-ppc64-2.5/gpaw-python ${HOME}/CH4.py

Absolute paths are important!

If you want to perform profiling with TAU submit the following wrapper instead::

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

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.
