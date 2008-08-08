#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own 
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']


#compiler = 'mpcc'
#libraries = []
#libraries += []

#library_dirs = []
#library_dirs += []

#include_dirs = []
#include_dirs += []

#extra_link_args = []
extra_link_args += ['-fopenmp']

#extra_compile_args = []
extra_compile_args += ['-fopenmp', '-O3']

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

#define_macros = []
define_macros += [('BLUEGENE', '1')]

#mpicompiler = None
#mpi_libraries = []
#mpi_libraries += []

#mpi_library_dirs = []
#mpi_library_dirs += []

#mpi_include_dirs = []
#mpi_include_dirs += []

#mpi_runtime_library_dirs = []
#mpi_runtime_library_dirs += []

#mpi_define_macros = []
#mpi_define_macros += []
