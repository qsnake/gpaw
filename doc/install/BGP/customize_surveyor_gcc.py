define_macros += [('GPAW_BGP', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_BLAS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_LAPACK', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_BLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_SCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]
define_macros += [("GPAW_MR3",1)] # requires developmental ScaLAPACK
# define_macros += [('GPAW_HPM',1)] # FLOP rate measurements
define_macros += [("GPAW_MPI_DEBUG",1)] # debugging
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
# DGEMM operations, particularly those with:
# alpha != 1.0 and beta != 0.0
#
# Goto is hand-tuned assembly, it will most
# likely always be faster than ESSL-4.x.
# NAR: Goto appears to cause core dumps for
# some problems, use at your own risk.
# Disabling the stackground seems to make
# the problem go away, but this is not 
# recommended.
# --env BG_STACKGUARDENABLE=0

libraries = [
           'scalapackmr3',
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

#          make sure XL library_dirs below match XL compiler version
#          (e.g. aug2010, jan2011) used in mpilinker variable

library_dirs = [
           '/soft/apps/SCALAPACK',
           '/soft/apps/BLACS',
           '/soft/apps/LAPACK',
           '/soft/apps/ESSL-4.4.1-0/lib',
#           '/soft/apps/LIBGOTO', 
           '/soft/apps/ibmcmp-aug2010/xlf/bg/11.1/bglib',
           '/soft/apps/ibmcmp-aug2010/xlsmp/bg/1.7/bglib',
           '/bgsys/drivers/ppcfloor/gnu-linux/lib',
#           '/soft/apps/UPC/lib',
           ]

include_dirs += [
    '/soft/apps/python/python-2.6-cnk-gcc/numpy-1.3.0/lib/python2.6/site-packages/numpy/core/include'
    ]

# TAU library below needed for automatic instrumentation only

mpi_libraries = [
#   'TAU',
    'mpich.cnk',
    'opa',
    'dcmf.cnk',
    'dcmfcoll.cnk',
    'SPI.cna',
    ]

mpi_library_dirs = [
    '/soft/apps/tau/tau-2.19.2/bgp/lib/bindings-bgptimers-gnu-mpi-python-pdt',
    '/bgsys/drivers/ppcfloor/comm/default/lib',
    '/bgsys/drivers/ppcfloor/comm/sys/lib',
    '/bgsys/drivers/ppcfloor/runtime/SPI',
    ]

   
compiler = "bgp_gcc.py"
mpicompiler = "bgp_gcc.py"
mpilinker   = "bgp_gcc_linker.py"
