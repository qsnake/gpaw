define_macros += [("GPAW_AIX",1)]
define_macros += [("GPAW_MKL",1)]
define_macros += [("GPAW_BGP",1)]
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
#
# It is also possible to use Goto BLAS instead
# of ESSL. The performance is similar though,
# and IBM updates ESSL frequently.

libraries = [
           'scalapack',
           'blacsCinit_MPI-BGP-0',
           'blacs_MPI-BGP-0',
           'lapack_bgp',
           'esslbg',
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
           '/opt/ibmcmp/xlf/bg/11.1/bglib',
           '/opt/ibmcmp/xlsmp/bg/1.7/bglib',
           '/bgsys/drivers/ppcfloor/gnu-linux/lib',
#           '/soft/apps/UPC/lib',
           ]

include_dirs += [
    '/home/dulak/numpy-1.0.4-1/bgsys/drivers/ppcfloor/gnu-linux/lib/python2.5/site-packages/numpy/core/include'
    ]

compiler = "bgp_gcc.py"
mpicompiler = "bgp_gcc.py"
mpilinker   = "bgp_gcc.py"
