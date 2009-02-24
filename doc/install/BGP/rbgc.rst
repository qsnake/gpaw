.. _rbgc:

=======================
bcssh.rochester.ibm.com
=======================

Instructions below are valid for ``frontend-13`` and the filesystem
:file:`/gpfs/fs2/frontend-13`.

The latest version of gpaw uses numpy
`<https://svn.fysik.dtu.dk/projects/gpaw/trunk/>`_.

To build an optimized (consider to build based on ``goto`` blas to achieve the best performance: see :ref:`surveyor`) numpy, save the :svn:`numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran <doc/install/BGP/numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran>`
patch file
(modifications required to get powerpc-bgp-linux-gfortran instead of
gfortran compiler),
the :svn:`numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg <doc/install/BGP/numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg>` patch file (lapack
section configured to use ``lapack_bgp`` and
blas section to use ``esslbg`` and ``cblas_bgp``),
and the :svn:`numpy-1.0.4-site.cfg.lapack_bgp_esslbg <doc/install/BGP/numpy-1.0.4-site.cfg.lapack_bgp_esslbg>` file (contains paths to
``lapack_bgp``, ``esslbg`` , ``cblas_bgp``, and xlf* related libraries).

**Note** that ``lapack_bgp`` and ``cblas_bgp`` are not available on ``frontend-13``, to build use instructions from `<http://www.pdc.kth.se/systems_support/computers/bluegene/LAPACK-CBLAS/LAPACK-CBLAS-build>`_. Python requires all librairies to have names like ``liblapack_bgp.a``, so please make the required links for ``lapack_bgp.a`` and ``cblas_bgp.a``. Moreover numpy requires that ``lapack_bgp``, ``esslbg``, and ``cblas_bgp`` reside in the same directory, so choose a directory and edit ``numpy-1.0.4-site.cfg.lapack_bgp_esslbg`` to reflect your installation path (in this example `/home/dulak/from_Nils_Smeds/CBLAS/lib/bgp`). Include the directory containing `cblas.h` in `include_dirs`. These instructions are valid also for `Surveyor/Intrepid` with the following locations of the libraries to be used in the makefiles: `/soft/apps/ESSL-4.4/lib` and `/opt/ibmcmp/lib/bg`.

**Warning** - if numpy built using these libraries fails
with errors of kind "R_PPC_REL24 relocation at 0xa3d664fc for symbol sqrt"
- please add ``-qpic`` to compile options for both ``lapack_bgp`` and ``cblas_bgp``. 
After bulding ``lapack_bgp`` and ``cblas_bgp``, get numpy-1.0.4 and do this::

  $ wget http://downloads.sourceforge.net/numpy/numpy-1.0.4.tar.gz
  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ mv numpy-1.0.4 numpy-1.0.4.optimized; cd numpy-1.0.4.optimized
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
  $ patch -p1 < ../numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg
  $ cp ../numpy-1.0.4-site.cfg.lapack_bgp_esslbg site.cfg
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ mkdir /gpfs/fs2/frontend-13/$USER
  $ root=/gpfs/fs2/frontend-13/$USER/numpy-1.0.4-1.optimized
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc -DNO_APPEND_FORTRAN\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

Numpy built in this way does contain the
:file:`$root/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/_dotblas.so`
, but running the following python
script (save it as :file:`/gpfs/fs2/frontend-13/$USER/dot.py`) results
in the same time as for the standard version of numpy (~329 sec)
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

Use the following command to submit this job ``cd
/gpfs/fs2/frontend-13/$USER; llsubmit numpy.llrun``, with the
following :file:`numpy.llrun` file::

  #!/bin/bash

  # @ job_type = bluegene
  # @ requirements = (Machine == "$(host)")
  # @ class = medium
  # @ job_name = $(user).$(host)
  # @ comment = "LoadLeveler llrun script"
  # @ error = $(job_name).$(jobid).err
  # @ output = $(job_name).$(jobid).out
  # @ wall_clock_limit = 00:15:00
  # @ notification = always
  # @ notify_user =
  # @ bg_connection = prefer_torus
  # @ bg_size = 32
  # @ queue

  dir="/gpfs/fs2/frontend-13/${USER}"
  home=$dir
  prog=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  args=${dir}/dot.py

  ldpath="${ldpath}:/bgsys/opt/ibmcmp/lib/bg"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/lib"
  pythonpath=":${home}/numpy-1.0.4-1.optimized/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages:"

  export LD_LIBRARY_PATH=\"$ldpath\"
  export PYTHONPATH=\"$pythonpath\"
  export OMP_NUM_THREADS=1

  mpirun=/bgsys/drivers/ppcfloor/bin/mpirun

  runargs="-np 1"
  runargs="$runargs -cwd $PWD"
  runargs="$runargs -exp_env LD_LIBRARY_PATH -exp_env PYTHONPATH -exp_env OMP_NUM_THREADS"
  runargs="$runargs -mode SMP"
  runargs="$runargs -verbose 2"

  echo "Hello. This is `hostname` at `date` `pwd`"

  echo "$mpirun $runargs $prog $args"
  /usr/bin/time $mpirun $runargs $prog $args

  echo "Program completed at `date` with exit code $?."

**Note** the colon before and after the string when setting pythonpath!

Here is how you build the standard numpy::

  $ gunzip -c numpy-1.0.4.tar.gz | tar xf -
  $ cd numpy-1.0.4
  $ patch -p1 < ../numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
  $ ldpath=/bgsys/drivers/ppcfloor/gnu-linux/lib
  $ mkdir /gpfs/fs2/frontend-13/$USER
  $ root=/gpfs/fs2/frontend-13/$USER/numpy-1.0.4-1
  $ p=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  $ c="\"/bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-gcc\""
  $ LD_LIBRARY_PATH="$ldpath" CC="$c" $p setup.py install --root="$root"

Build GPAW
(``PYTHONPATH=/gpfs/fs2/frontend-13/mdulak/numpy-1.0.4-1.optimized/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages
LD_LIBRARY_PATH="$ldpath" $p setup.py build_ext``) in
:file:`/gpfs/fs2/frontend-13/$USER/gpaw` (you need to install the ase
also somewhere below :file:`/gpfs/fs2/frontend-13/$USER`!)  with this
:file:`customize.py` file::

  scalapack = True

  extra_compile_args += [
      '-O3'
      ]

  libraries = [
             'gfortran',
             'lapack_bgp',
             'scalapack',
             'blacs',
             'lapack_bgp',
             'goto',
             'xlf90_r',
             'xlopt',
             'xl',
             'xlfmath',
             'xlsmp'
             ]

  library_dirs = [
             '/home/mdulak/blas-lapack-lib',
             '/home/mdulak/blacs-dev',
             '/home/mdulak/SCALAPACK',
             '/opt/ibmcmp/xlf/bg/11.1/bglib',
             '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
             '/bgsys/drivers/ppcfloor/gnu-linux/lib'
             ]

  gpfsdir = '/gpfs/fs2/frontend-13/mdulak'
  python_site = 'bgsys/drivers/ppcfloor/gnu-linux'

  include_dirs += [gpfsdir+'/Numeric-24.2-1/'+python_site+'/include/python2.5',
                   gpfsdir+'/numpy-1.0.4-1.optimized/'+python_site+'/lib/python2.5/site-packages/numpy/core/include']

  extra_compile_args += ['-std=c99']

  define_macros += [
            ('GPAW_AIX', '1'),
            ('GPAW_MKL', '1'),
            ('GPAW_BGP', '1')
            ]

Because of missing ``popen3`` function you need to remove all the
contents of the :file:`gpaw/version.py` file after ``version =
'0.4'``.  The same holds for :file:`ase/version.py` in the ase
installation!  Suggestions how to skip the ``popen3`` testing in
:file:`gpaw/version.py` on BGP are welcome!

Note that only files located below :file:`/gpfs/fs2/frontend-13` are
accesible to the compute nodes (even python scripts!).  A gpaw script
:file:`/gpfs/fs2/frontend-13/$USER/gpaw/test/CH4.py` can be submitted to
32 CPUs in the single mode (SMP) for 30 minutes using `LoadLeveler
<http://www.fz-juelich.de/jsc/ibm-bgl/usage/loadl/>`_ like this::

  cd /gpfs/fs2/frontend-13/$USER
  llsubmit gpaw-script.llrun

where :file:`gpaw-script.llrun` looks like this::

  #!/bin/bash

  # @ job_type = bluegene
  # @ requirements = (Machine == "$(host)")
  # @ class = medium
  # @ job_name = $(user).$(host)
  # @ comment = "LoadLeveler llrun script"
  # @ error = $(job_name).$(jobid).err
  # @ output = $(job_name).$(jobid).out
  # @ wall_clock_limit = 00:30:00
  # @ notification = always
  # @ notify_user =
  # @ bg_connection = prefer_torus
  # @ bg_size = 32
  # @ queue

  dir=/gpfs/fs2/frontend-13/$USER
  home=$dir
  prog=${home}/gpaw/build/bin.linux-ppc64-2.5/gpaw-python
  #prog=/bgsys/drivers/ppcfloor/gnu-linux/bin/python
  args="${home}/gpaw/test/CH4.py --sl_diagonalize=2,2,2,4"

  ldpath="${ldpath}:/bgsys/opt/ibmcmp/lib/bg"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib"
  ldpath="${ldpath}:/bgsys/drivers/ppcfloor/gnu-linux/lib"
  pythonpath=":${home}/Numeric-24.2-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/Numeric"
  pythonpath="${pythonpath}:${home}/numpy-1.0.4-1.optimized/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages"
  pythonpath="${pythonpath}:${home}/gpaw"
  pythonpath="${pythonpath}:${home}/ase3k:"

  export LD_LIBRARY_PATH=\"$ldpath\"
  export PYTHONPATH=\"$pythonpath\"
  export GPAW_SETUP_PATH="${home}/gpaw-setups-0.4.2039"
  export OMP_NUM_THREADS=1

  mpirun=/bgsys/drivers/ppcfloor/bin/mpirun

  runargs="-np 32"
  runargs="$runargs -cwd $PWD"
  runargs="$runargs -exp_env LD_LIBRARY_PATH -exp_env PYTHONPATH -exp_env GPAW_SETUP_PATH -exp_env OMP_NUM_THREADS"
  runargs="$runargs -mode SMP"
  runargs="$runargs -verbose 1"

  echo "Hello. This is `hostname` at `date` `pwd`"

  echo "$mpirun $runargs $prog $args"
  /usr/bin/time $mpirun $runargs $prog $args

  echo "Program completed at `date` with exit code $?."

Absolute paths are important!

It's convenient to customize as in :file:`gpaw-qsub.py` which can be
found at the :ref:`parallel_runs` page.
