#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own 
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']


#compiler = 'mpcc'
#libraries = []
#libraries += ['gomp']

#library_dirs = []
#library_dirs += []

#include_dirs = []
#include_dirs += []

#extra_link_args = []
extra_link_args += ['-fopenmp']

#extra_compile_args = []
extra_compile_args += ['-fopenmp']

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

#define_macros = []
define_macros += [('BLUEGENE', '1'), ('NUM_OF_THREADS', '2')]

#mpicompiler = None
#mpi_libraries = []
#mpi_libraries += ['gomp']

#mpi_library_dirs = []
#mpi_library_dirs += []

#mpi_include_dirs = []
#mpi_include_dirs += []

#mpi_runtime_library_dirs = []
#mpi_runtime_library_dirs += []

#mpi_define_macros = []
mpi_define_macros += [('BLUEGENE', '1'), ('NUM_OF_THREADS', '2')]
