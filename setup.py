#!/usr/bin/env python

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
import re
import distutils
from distutils.core import setup, Extension
from distutils.util import get_platform
from distutils.sysconfig import get_config_var
from glob import glob
from os.path import join
from stat import ST_MTIME

from config import *

# Get the current version number:
execfile('gridpaw/version.py')

long_description = """\
A grid-based real-space Projector Augmented Wave (PAW) method Density
Functional Theory (DFT) code featuring: Flexible boundary conditions,
k-points and gradient corrected exchange-correlation functionals."""

msg = [' ']

libraries = []
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = []
runtime_library_dirs = []
extra_objects = []
define_macros = []

mpi_libraries = []
mpi_library_dirs = []
mpi_include_dirs = []
mpi_runtime_library_dirs = []
mpi_define_macros = []


packages = ['gridpaw',
            'gridpaw.io',
            'gridpaw.atom',
            'gridpaw.utilities',
            'gridpaw.setuptests']

check_packages(packages, msg)

get_system_config(define_macros, include_dirs, libraries, library_dirs,
                  extra_link_args, extra_compile_args,
                  runtime_library_dirs, extra_objects, msg)

mpicompiler, custom_interpreter = get_parallel_config(mpi_libraries,
                                                      mpi_library_dirs,
                                                      mpi_include_dirs,
                                                      mpi_runtime_library_dirs,
                                                      mpi_define_macros)



#User provided customizations
if os.path.isfile('customize.py'):
    execfile('customize.py')

if not custom_interpreter and mpicompiler:
    msg += ['* Compiling gpaw with %s' % mpicompiler]
    #A sort of hack to change the used compiler
    cc = get_config_var('CC')
    oldcompiler=cc.split()[0]
    os.environ['CC']=cc.replace(oldcompiler,mpicompiler)
    ld = get_config_var('LDSHARED')
    oldlinker=ld.split()[0]
    os.environ['LDSHARED']=ld.replace(oldlinker,mpicompiler)
    define_macros.append(('PARALLEL', '1'))
elif not custom_interpreter:
    libraries += mpi_libraries
    library_dirs += mpi_library_dirs
    include_dirs += mpi_include_dirs
    runtime_library_dirs += mpi_runtime_library_dirs
    define_macros += mpi_define_macros
    

# Check the command line so that custom interpreter is build only with "build"
# or "build_ext":
if 'build' not in sys.argv and 'build_ext' not in sys.argv:
    custom_interpreter = False

# distutils clean does not remove the _gridpaw.so library so do it here:
plat = get_platform() + '-' + sys.version[0:3]
gpawso = 'build/lib.%s/' % plat + '_gridpaw.so'
if "clean" in sys.argv and os.path.isfile(gpawso):
    print 'removing ', gpawso
    os.remove(gpawso)

include_dirs += [os.environ['HOME'] + '/include/python']

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
check_dependencies(sources)

extension = Extension('_gridpaw',
                      sources,
                      libraries=libraries,
                      library_dirs=library_dirs,
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args,
                      runtime_library_dirs=runtime_library_dirs,
                      extra_objects=extra_objects)

scripts = glob(join('tools', 'gpaw-*[a-z]')) + ['tools/gpaw']
if custom_interpreter:
    scripts.append('build/bin.%s/' % plat + 'gridpaw-python')

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs,extra_objects)

setup(name = 'gridpaw',
      version=version,
      description='A grid-based real-space PAW method DFT code',
      author='J. J. Mortensen',
      author_email='jensj@fysik.dtu.dk',
      url='http://www.fysik.dtu.dk',
      license='GPL',
      platforms=['unix'],
      packages=packages,
      ext_modules=[extension],
      scripts=scripts,
      long_description=long_description,
      )


if custom_interpreter:
    msg += build_interpreter(define_macros, include_dirs, libraries, library_dirs,
                      extra_link_args, extra_compile_args,mpicompiler)

if ('PARALLEL', '1') not in define_macros:
    msg += ['* A serial version of gridpaw was build!']

# Messages make sense only when building
if "build" in sys.argv or "build_ext" in sys.argv:
    for line in msg:
        print line
