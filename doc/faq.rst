.. _faq:

==========================
Frequently Asked Questions
==========================

.. contents::

General
=======

How should I cite GPAW?
-----------------------

If you find GPAW useful in your research please cite the original paper:

   | J. J. Mortensen, L. B. Hansen , and K. W. Jacobsen
   | `Real-space grid implementation of the projector augmented wave method`__
   | Physical Review B, Vol. **71**, 035109, 2005
  
   __ http://dx.doi.org/10.1103/PhysRevB.71.035109

If you are using the time-dependent DFT part of the code, please cite also:

   | M. Walter, H. Häkkinen, L. Lehtovaara, M. Puska, J. Enkovaara, C. Rostgaard and J. J. Mortensen
   | `Time-dependent density-functional theory in the projector augmented-wave method`__
   | Journal of Chemical Physics, Vol. **128**, 244101, 2008

   __ http://link.aip.org/link/?JCP/128/244101


How do you pronounce GPAW?
--------------------------

In english: "geepaw" with a long "a".

In danish: Først bogstavet "g", derefter "pav": "g-pav".

In finnish: supisuomalaisittain "kee-pav"

Download
========

Trying to checkout the code via SVN resulted::

 [~]$ svn checkout "https://svn.fysik.dtu.dk/projects/gpaw/trunk"
 svn: Unrecognized URL scheme 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'

This error is diplayed in case the library 'libsvn_ra_dav' is missing on your system. The library is used by SVN, but is not installed by default. 



Compiling the C-code
====================

For architecture dependent settings see the :ref:`platforms_and_architectures` page.

Compilation of the C part failed::

 [~]$ python2.4 setup.py build_ext
 building '_gpaw' extension
 pgcc -fno-strict-aliasing -DNDEBUG -O2 -g -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -m64 -D_GNU_SOURCE -fPIC -fPIC -I/usr/include/python2.4 -c c/localized_functions.c -o build/temp.linux-x86_64-2.4/c/localized_functions.o -Wall -std=c99
 pgcc-Warning-Unknown switch: -fno-strict-aliasing
 PGC-S-0040-Illegal use of symbol, _Complex (/usr/include/bits/cmathcalls.h: 54)

You are probably using another compiler, than it was used for compiling python. Undefine the environment variables CC, CFLAGS and LDFLAGS with::

 # sh/bash users:
 unset CC; unset CFLAGS; unset LDFLAGS
 # csh/tcsh users: 
 unsetenv CC; unsetenv CFLAGS; unsetenv LDFLAGS

and try again.

Calculation does not converge
=============================

First, you can try to get more information during the calculation by setting the ``verbose`` parameter::

  Calculator(..., verbose=True)

If your (finite) system contains nearly degenerate occupied and unoccupied states, there can be convergence problems.
You can try to occupy the states with Fermi-distribution by specifying the ``width`` parameter::

  Calculator(..., width=0.05)

However, note that this might change also the symmetry of your system

Sometimes it is possible to improve the convergence by changing the default parameters for 
`density mixing`_, try e.g.::

  mixer=Mixer(0.1, 5, metric='new', weight=100.0)
  Calculator(..., mixer=mixer)

In rare occasions the default eigensolver_ ``rmm-diis`` does not converge, and one can try either conjugate gradient or Davidson eigensolvers::

  Calculator(..., eigensolver='cg')

.. _density mixing: wiki:GPAW:Manual#density-mixing
.. _eigensolver: :ref:`gpaw_manual_eigensolver`

Poisson solver did not converge!
================================

If you are doing a spin-polarized calculation for an isolated molecule, 
then you should set the Fermi temperature to a low value: 
``width=0.001``.


Tests fail!
===========

Please report the failing test as well as information about your environment (processor architecture, C-compiler, 
BLAS and LAPACK libraries, MPI library) to the mailing-list_. 

.. _mailing-list: https://lists.berlios.de/mailman/listinfo/gridpaw-developer

Writing a restart file fails
============================

Writing a restart file results in error::

  File "/.../gridpaw/trunk/gpaw/io/netcdf.py", line 68, in fill
      self.var[indices] = array
  IOError: netcdf: Operation not allowed in define mode

NetCDF (or more specifically Scientific python's netCDF) does not support files larger than 2 GB. Use instead ``.gpw``
format or write the wave functions into separate files. See the :ref:`restart_files` page for more details.
