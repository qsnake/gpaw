scalapack = True

extra_compile_args = ['-fast', '-std=c99', '-fPIC']

compiler = 'icc'

libraries = ['iomp5']

mkl_lib_path = '/software/intel/mkl/10.0.4.023/lib/em64t/'

library_dirs = [mkl_lib_path]

extra_link_args = [
    mkl_lib_path+'libmkl_lapack.a',
    mkl_lib_path+'libmkl_intel_lp64.a',
    mkl_lib_path+'libmkl_intel_thread.a',
    mkl_lib_path+'libmkl_core.a',
    mkl_lib_path+'libmkl_blacs_openmpi_lp64.a',
    mkl_lib_path+'libmkl_scalapack_lp64.a',
    mkl_lib_path+'libmkl_blacs_openmpi_lp64.a',
    mkl_lib_path+'libmkl_lapack.a',
    mkl_lib_path+'libmkl_intel_lp64.a',
    mkl_lib_path+'libmkl_intel_thread.a',
    mkl_lib_path+'libmkl_core.a',
    ]

define_macros = [('GPAW_MKL', '1')]
