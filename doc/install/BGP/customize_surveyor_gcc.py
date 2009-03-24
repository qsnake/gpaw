scalapack = True

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

define_macros += [
          ('GPAW_AIX', '1'),
          ('GPAW_MKL', '1'),
          ('GPAW_BGP', '1')
          ('GPAW_ASYNC', '1')
          ]

mpicompiler = "bgp_gcc.py"
mpilinker = "bgp_gcc.py"
compiler = "bgp_gcc.py"
