#!/usr/bin/env python

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
import re
from distutils.core import setup, Extension
from distutils.util import get_platform
from glob import glob
from os.path import join
from stat import ST_MTIME


# Get the current version number:
execfile('gridpaw/version.py')


long_description = """\
A grid-based real-space Projector Augmented Wave (PAW) method Density
Functional Theory (DFT) code featuring: Flexible boundary conditions,
k-points and gradient corrected exchange-corellation functionals."""


if sys.version_info < (2, 3, 0, 'final', 0):
    raise SystemExit, 'Python 2.3.1 or later is required!'


# ???? Use a setup.cfg file and the config command:
#  check_func
#  check_lib
#  check_header




scripts = glob(join('tools', 'gridpaw-*[a-z]'))

libraries = []
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = []
runtime_library_dirs = []
extra_objects = []
define_macros = []

machine = os.uname()[4]
if machine == 'sun4u':

    #  _
    # |_ | ||\ |
    #  _||_|| \|
    #
    
    include_dirs += ['/opt/SUNWhpc/include']
    extra_compile_args += ['-KPIC']
    libraries += ['mpi']
    library_dirs += ['/opt/SUNWspro/lib',
                     '/opt/SUNWhpc/lib']
    runtime_library_dirs = ['/opt/SUNWspro/lib',
                            '/opt/SUNWhpc/lib']
    extra_objects = ['/opt/SUNWhpc/lib/shmpm.so.2',
                     '/opt/SUNWhpc/lib/rsmpm.so.2',
                     '/opt/SUNWhpc/lib/tcppm.so.2']

    # We need the -Bstatic before the -lsunperf and -lfsu:
    extra_link_args = ['-Bstatic', '-lsunperf', '-lfsu', '-lmtsk']

    define_macros.append(('NO_C99_COMPLEX', '1'))
    define_macros.append(('PARALLEL', '1'))

elif machine == 'x86_64':

    #    _ 
    # \/|_||_    |_ |_|
    # /\|_||_| _ |_|  |
    #
    
    extra_compile_args += ['-Wall', '-std=c99']

    libraries += ['acml', 'g2c']
    library_dirs += ['/opt/acml/gnu64/lib']
    extra_link_args += ['-Wl,-rpath=/opt/acml/gnu64/lib']
    print 'Using ACML library'

    output = os.popen('mpicc -showme').read()

    if output == '':
        print 'SERIAL version'
    else:
        define_macros.append(('PARALLEL', '1'))
        libraries += re.findall(' -l(\S+)', output)
        while 'aio' in libraries:
            libraries.remove('aio')
        while 'lamf77mpi' in libraries:
            libraries.remove('lamf77mpi')
        library_dirs += [dir.replace('pgi', 'gcc')
                         for dir in re.findall('-L(\S+)', output)]
        include_dirs += [dir.replace('pgi', 'gcc')
                         for dir in re.findall('-I(\S+)', output)]

elif sys.platform == 'aix5':

    #
    # o|_  _ _
    # ||_)| | |
    #

    extra_compile_args += ['-qlanglvl=stdc99']
    extra_link_args += ['-bmaxdata:0x80000000', '-bmaxstack:0x80000000']


    libraries += ['f', 'essl', 'lapack']
    define_macros.append(('GRIDPAW_AIX', '1'))
    include_dirs += ['/usr/lpp/ppe.poe/include']
    libraries += ['mpi']
    library_dirs += ['/usr/lpp/ppe.poe/lib']
#    runtime_library_dirs = ['/opt/SUNWspro/lib', '/opt/SUNWhpc/lib']
#    extra_link_args = ['-Bstatic', '-lsunperf', '-lfsu', '-lmtsk']

    define_macros.append(('NO_C99_COMPLEX', '1'))
    define_macros.append(('PARALLEL', '1'))
#    extra_link_args += ['-Wl,-rpath=/opt/acml/gnu64/lib']
#    print 'Using ACML library'


else:

    #      _
    # o|_ |_||_
    # ||_||_||_|
    #
    
    extra_compile_args += ['-Wall', '-std=c99']

    libs = glob('/opt/intel/mkl*cluster/lib/32/libmkl_ia32.a')
    if libs != []:
        libs.sort()
        libraries += ['mkl_lapack',
                      'mkl_ia32', 'guide', 'pthread', 'mkl', 'mkl_def']
        library_dirs += [os.path.dirname(libs[-1])]
        print 'Using MKL library:', library_dirs[-1]
        extra_link_args += ['-Wl,-rpath=' + library_dirs[-1]]
    else:
        atlas = False
        for dir in ['/usr/lib', 'usr/local/lib']:
            if glob(join(dir, 'libatlas.a')) != []:
                atlas = True
                break
        if atlas:
##            libraries += ['lapack', 'cblas', 'blas']
            libraries += ['lapack', 'atlas', 'blas']
            library_dirs += [dir]
            print 'Using ATLAS library'
        else:
            libraries += ['blas', 'lapack']

    output = os.popen('mpicc -showme').read()

    if output == '':
        print 'SERIAL version'
    else:
        define_macros.append(('PARALLEL', '1'))
        libraries += re.findall(' -l(\S+)', output)
        while 'aio' in libraries:
            libraries.remove('aio')
        while 'lamf77mpi' in libraries:
            libraries.remove('lamf77mpi')
        library_dirs += [dir.replace('pgi', 'gcc')
                         for dir in re.findall('-L(\S+)', output)]
        include_dirs += [dir.replace('pgi', 'gcc')
                         for dir in re.findall('-I(\S+)', output)]

include_dirs += [os.environ['HOME'] + '/include/python']

sources = glob('c/*.c') + ['c/bmgs/bmgs.c']
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

# Distutils does not do deep dependencies correctly.  We take care of
# that here so that "python setup.py build_ext" always does the right
# thing!

mtimes = {}  # modification times
include = re.compile('^#\s*include "(\S+)"', re.MULTILINE)
def mtime(path, name):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned."""

    global mtimes
    if mtimes.has_key(name):
        return mtimes[name]
    t = os.stat(path + name)[ST_MTIME]
    for name2 in include.findall(open(path + name).read()):
        if name2 != name:
            t = max(t, mtime(path, name2))
    mtimes[name] = t
    return t

# Remove object files if any dependencies have changed:
plat = get_platform() + '-' + sys.version[0:3]
remove = False
for source in sources:
    path, name = os.path.split(source)
    t = mtime(path + '/', name)
    o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
    if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
        print 'removing', o
        os.remove(o)
        remove = True

so = 'build/lib.%s/_gridpaw.so' % plat
if os.path.exists(so) and remove:
    # Remove shared object C-extension:
    print 'removing', so
    os.remove(so)


setup(name = 'gridpaw',
      version=version,
      description='A grid-based real-space PAW method DFT code',
      author='J. J. Mortensen',
      author_email='jensj@fysik.dtu.dk',
      url='http://www.fysik.dtu.dk',
      license='GPL',
      platforms=['unix'],
      packages=['gridpaw',
                'gridpaw.atom',
                'gridpaw.utilities',
                'gridpaw.tests'],
      ext_modules=[extension],
      scripts=scripts,
      long_description=long_description,
##      data_files=[('doc', ['doc/index.txt'])],
      )

if sys.platform == 'aix5':

    #
    # o|_  _ _
    # ||_)| | |
    #

    # Normally nothing needs to be changed below
    import distutils
    import distutils.sysconfig
    import os  

    cfgDict = distutils.sysconfig.get_config_vars()

    # Name of the MPI compilation script.
    mpicompiler = 'mpcc_r'
    sources='c/_gridpaw.c'


    cmd = '%s -DGRIDPAW_INTERPRETER=1 %s %s -o tools/gridpaw-python -I%s %s %s -L%s -lpython%s %s %s %s %s %s' % \
          (mpicompiler,
           ' '.join(['-D%s=%s' % x for x in define_macros]),
           cfgDict['LINKFORSHARED'].replace('Modules', cfgDict['LIBPL']), 
           cfgDict['INCLUDEPY'],
           sources,
           ' '.join(['build/temp.aix-5.2-2.3/' + x[:-1] + 'o'
                     for x in glob('c/[a-z]*.c') + ['c/bmgs/bmgs.c']]),
           cfgDict['LIBPL'],
           cfgDict['VERSION'], 
           cfgDict['LIBS'], 
           cfgDict['LIBM'],
           ' '.join(['-l' + lib for lib in libraries]),
           ' '.join(extra_compile_args),
           ' '.join(extra_link_args))
    
    print 'cmd = ', cmd 
    os.system(cmd)
    """
    extra_compile_args += ['-qlanglvl=stdc99']

    libraries += ['f', 'essl', 'lapack']
    define_macros.append(('GRIDPAW_AIX', '1'))
    include_dirs += ['/usr/lpp/ppe.poe/include']
    libraries += ['mpi']
    library_dirs += ['/usr/lpp/ppe.poe/lib']
#    runtime_library_dirs = ['/opt/SUNWspro/lib', '/opt/SUNWhpc/lib']
#    extra_link_args = ['-Bstatic', '-lsunperf', '-lfsu', '-lmtsk']

    define_macros.append(('NO_C99_COMPLEX', '1'))
    define_macros.append(('PARALLEL', '1'))
#    extra_link_args += ['-Wl,-rpath=/opt/acml/gnu64/lib']
#    print 'Using ACML library'
"""
