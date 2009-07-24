scalapack = True
compiler = 'pathcc'
libraries = [
    'pathfortran', 'goto','goto','goto','goto', 'acml', 'acml', 'acml',
    'scalapack', 'mpiblacsF77init', 'mpiblacs', 'scalapack', 'mpi', 'mpi_f77']
library_dirs = [
    '/opt/openmpi/1.3.2-1.pathscale/lib64',
    '/opt/goto/1.26/1.el5.fys.pathscale.smp/lib64',
    '/opt/acml/4.2.0/pathscale64/lib',
    '/opt/pathscale/lib/3.2',
    '/opt/blacs/1.1/24.el5.fys.pathscale.openmpi/lib64',
    '/opt/scalapack/1.8.0/1.el5.fys.pathscale.openmpi.goto.1.26.acml/lib64']
include_dirs += ['/opt/openmpi/1.3.2-1.pathscale/include']
extra_link_args = [
    '-Wl,-rpath=/opt/openmpi/1.3.2-1.pathscale/lib64,'
    '-rpath=/opt/goto/1.26/1.el5.fys.pathscale.smp/lib64,'
    '-rpath=/opt/acml/4.2.0/pathscale64/lib,'
    '-rpath=/opt/pathscale/lib/3.2,'
    '-rpath=/opt/blacs/1.1/24.el5.fys.pathscale.openmpi/lib64,'
    '-rpath=/opt/scalapack/1.8.0/1.el5.fys.pathscale.openmpi.goto.1.26.acml/lib64']
extra_compile_args = ['-O3', '-OPT:Ofast', '-ffast-math', '-std=c99', '-fPIC']
define_macros += [('GPAW_MKL', '1'), ('SL_SECOND_UNDERSCORE', '1')]
mpicompiler = '/opt/openmpi/1.3.2-1.pathscale/bin/mpicc'
mpilinker = mpicompiler
platform_id = 'amd'
