scalapack = False

extra_compile_args = ['-O3', '-std=c99', '-fpic']

#compiler = '/usr/lam-7.1.4_intelv10/bin/mpicc'
compiler = 'gcc'

libraries = ['mkl_def', 'mkl_lapack', 'mkl_core', 'mkl_sequential', 'mkl_gf_lp64', 'iomp5']

mkl_lib_path = '/opt/intel/mkl/10.2.1.017/lib/em64t'
#ompi_lib_path = '/usr/lam-7.1.4_intelv10/lib'

#library_dirs = [mkl_lib_path, ompi_lib_path]
library_dirs = [mkl_lib_path]

#extra_link_args =['-Wl,-rpath='+mkl_lib_path+',-rpath='+ompi_lib_path]
extra_link_args =['-Wl,-rpath='+mkl_lib_path]

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]

