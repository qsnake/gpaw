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
execfile('gpaw/version.py')

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
undef_macros = []

mpi_libraries = []
mpi_library_dirs = []
mpi_include_dirs = []
mpi_runtime_library_dirs = []
mpi_define_macros = []


packages = ['gpaw',
            'gpaw.io',
            'gpaw.gui',
            'gpaw.mpi',
            'gpaw.atom',
            'gpaw.lrtddft',
            'gpaw.tddft',
            'gpaw.testing',
            'gpaw.utilities',
            'gpaw.eigensolvers',
            'gpaw.gui.languages']

force_inclusion_of_ase = False
if '--force-inclusion-of-ase' in sys.argv:
    force_inclusion_of_ase = True
    sys.argv.remove('--force-inclusion-of-ase')
    
check_packages(packages, msg, force_inclusion_of_ase)

get_system_config(define_macros, undef_macros,
                  include_dirs, libraries, library_dirs,
                  extra_link_args, extra_compile_args,
                  runtime_library_dirs, extra_objects, msg)

mpicompiler = get_parallel_config(mpi_libraries,
                                  mpi_library_dirs,
                                  mpi_include_dirs,
                                  mpi_runtime_library_dirs,
                                  mpi_define_macros)


compiler = None

#User provided customizations
if os.path.isfile('customize.py'):
    execfile('customize.py')

if compiler is not None:
    msg += ['* Compiling gpaw with %s' % compiler]
    # A hack to change the used compiler and linker:
    vars = get_config_vars()
    for key in ['CC', 'LDSHARED']:
        value = vars[key].split()
        # first argument is the compiler/linker.  Replace with mpicompiler:
        value[0] = compiler
        vars[key] = ' '.join(value)
    
custom_interpreter = False
# Check the command line so that custom interpreter is build only with 
# "build", "build_ext", or "install":
if mpicompiler is not None:
    for cmd in ['build', 'build_ext', 'install']:
        if cmd in sys.argv:
            custom_interpreter = True
            break

# distutils clean does not remove the _gpaw.so library and gpaw-python
# binary so do it here:
plat = get_platform() + '-' + sys.version[0:3]
gpawso = 'build/lib.%s/' % plat + '_gpaw.so'
gpawbin = 'build/bin.%s/' % plat + 'gpaw-python'
if "clean" in sys.argv:
    if os.path.isfile(gpawso):
        print 'removing ', gpawso
        os.remove(gpawso)
    if os.path.isfile(gpawbin):
        print 'removing ', gpawbin
        os.remove(gpawbin)

include_dirs += [os.environ['HOME'] + '/include/python']

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
check_dependencies(sources)

extension = Extension('_gpaw',
                      sources,
                      libraries=libraries,
                      library_dirs=library_dirs,
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      undef_macros=undef_macros,
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args,
                      runtime_library_dirs=runtime_library_dirs,
                      extra_objects=extra_objects)

scripts = [join('tools', script)
           for script in ('gpaw-setup', 'gpaw', 'g2')]

write_configuration(define_macros, include_dirs, libraries, library_dirs,
                    extra_link_args, extra_compile_args,
                    runtime_library_dirs,extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros)

setup(name = 'gpaw',
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
    scripts.append('build/bin.%s/' % plat + 'gpaw-python')
    error, par_msg = build_interpreter(define_macros, include_dirs, libraries,
                             library_dirs, extra_link_args, extra_compile_args,
                             runtime_library_dirs, extra_objects,
                             mpicompiler, mpi_libraries, mpi_library_dirs,
                             mpi_include_dirs,
                             mpi_runtime_library_dirs, mpi_define_macros)
    msg += par_msg
    # install also gpaw-python
    if "install" in sys.argv and error == 0:
        setup(name = 'gpaw',
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

else:
    msg += ['* Only a serial version of gpaw was build!']

# Messages make sense only when building
if "build" in sys.argv or "build_ext" in sys.argv:
    for line in msg:
        print line
