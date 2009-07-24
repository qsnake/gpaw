scalapack = True
compiler = 'gcc43'
libraries = [
    'gfortran', 'blas', 'blas', 'blas', 'blas', 'lapack', 'lapack',
    'lapack', 'scalapack', 'mpiblacsF77init', 'mpiblacs', 'scalapack',
    'mpi', 'mpi_f77']
library_dirs = [
    '/opt/openmpi/1.3.2-1.gfortran43/lib64',
    '/usr/lib64', '/usr/lib64','/usr/lib64',
    '/opt/blacs/1.1/24.el5.fys.gfortran43.openmpi/lib64',
    '/opt/scalapack/1.8.0/1.el5.fys.gfortran43.openmpi.blas.3.0.37.el5.lapack/lib64']
include_dirs += ['/opt/openmpi/1.3.2-1.gfortran43/include']
extra_link_args = [
    '-Wl,-rpath=/opt/openmpi/1.3.2-1.gfortran43/lib64,'
    '-rpath=/usr/lib64,'
    '-rpath=/usr/lib64,-rpath=/usr/lib64,'
    '-rpath=/opt/blacs/1.1/24.el5.fys.gfortran43.openmpi/lib64,'
    '-rpath=/opt/scalapack/1.8.0/1.el5.fys.gfortran43.openmpi.blas.3.0.37.el5.lapack/lib64']
extra_compile_args = ['-O3', '-std=c99', '-funroll-all-loops', '-fPIC']
define_macros += [('GPAW_MKL', '1')]
mpicompiler = '/opt/openmpi/1.3.2-1.gfortran43/bin/mpicc'
mpilinker = mpicompiler
platform_id = 'intel'
