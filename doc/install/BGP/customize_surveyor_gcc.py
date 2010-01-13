define_macros += [('GPAW_BGP', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_BLAS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_LAPACK', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_BLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_SCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]
# define_macros += [('GPAW_HPM',1)] # FLOP rate measurements
# define_macros += [("GPAW_MPI_DEBUG",1)] # debugging
# define_macros += [("GPAW_OMP",1)] # not really working

scalapack = True

# If you are using threading, you probably
# need to change the following library:
# xlomp_ser -> xlsmp
#
# DO NOT INTERCHANGE THE ORDER OF LAPACK
# & ESSL, LAPACK SHOULD BE LINKED FIRST.
#
# Goto appears to be much faster for general
# DGEMM operations, particularly those with
# alpha != beta != 0
#
# Goto is hand-tuned assembly, it will most
# likely always be faster than ESSL-4.x.
# NAR: Goto appears to cause core dumps for
# some problems, use at your own risk.

libraries = [
           'scalapack',
           'blacsCinit_MPI-BGP-0',
           'blacs_MPI-BGP-0',
           'lapack_bgp',
           'esslbg',
#           'goto',
           'xlf90_r',
           'xlopt',
           'xl',
           'xlfmath',
           'xlomp_ser',
#           'hpm',
           ]

library_dirs = [
           '/soft/apps/SCALAPACK',
           '/soft/apps/BLACS',
           '/soft/apps/LAPACK',
           '/soft/apps/ESSL-4.4.1-0/lib',
#           '/soft/apps/LIBGOTO',
           '/opt/ibmcmp/xlf/bg/11.1/bglib',
           '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
           '/bgsys/drivers/ppcfloor/gnu-linux/lib',
#           '/soft/apps/UPC/lib',
           ]

include_dirs += [
    '/soft/apps/python/python-2.6-cnk-gcc/numpy-1.3.0/lib/python2.6/site-packages/numpy/core/include'
    ]

compiler = "bgp_gcc.py"
mpicompiler = "bgp_gcc.py"
mpilinker   = "bgp_gcc.py"
