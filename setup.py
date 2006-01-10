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
    raise SystemExit('Python 2.3.1 or later is required!')
    
try:
    import Numeric
except ImportError:
    raise SystemExit('Numeric is not installed!')

msg = []

try:
    import ASE
except ImportError:
    msg += ['* ASE is not installed!  You may be able to install gridpaw, but',
            "  you can't use it without ASE!"]

try:
    import Scientific.IO.NetCDF
except ImportError:
    try:
        import Scientific
    except ImportError:
        msg = ['* Scientific is not installed.']
    else:
        msg = ['* Scientific.IO.NetCDF is not installed (the NetCDF C-library',
               '  is probably missing).']
    msg += ['  You will not be able to write and read wavefunctions!']
        
## import numpy
## numpy.get_numpy_include()

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
    extra_compile_args += ['-KPIC', '-fast']

    # Suppress warning from -fast (-xarch=native):
    f = open('cc-test.c', 'w')
    f.write('int main(){}\n')
    f.close()
    stderr = os.popen3('cc cc-test.c -fast')[2].read()
    arch = re.findall('-xarch=(\S+)', stderr)
    os.remove('cc-test.c')
    if len(arch) > 0:
        extra_compile_args += ['-xarch=%s' % arch[-1]]
        
    libraries += ['mpi']
    library_dirs += ['/opt/SUNWspro/lib',
                     '/opt/SUNWhpc/lib']
    runtime_library_dirs = ['/opt/SUNWspro/lib',
                            '/opt/SUNWhpc/lib']
    extra_objects = ['/opt/SUNWhpc/lib/shmpm.so.2',
                     '/opt/SUNWhpc/lib/rsmpm.so.2',
                     '/opt/SUNWhpc/lib/tcppm.so.2']

    # We need the -Bstatic before the -lsunperf and -lfsu:
    extra_link_args = ['-Bstatic', '-lsunperf', '-lfsu']
    cc_version = os.popen3('cc -V')[2].readline().split()[3]
    if cc_version > '5.6':
        libraries.append('mtsk')
    else:
        extra_link_args.append('-lmtsk')
        define_macros.append(('NO_C99_COMPLEX', '1'))

    define_macros.append(('PARALLEL', '1'))

elif machine == 'x86_64':

    #    _ 
    # \/|_||_    |_ |_|
    # /\|_||_| _ |_|  |
    #
    
    extra_compile_args += ['-Wall', '-std=c99']

    libraries += ['acml', 'g2c']
    acml = glob('/opt/acml*/gnu64/lib')[-1]
    library_dirs += acml
    extra_link_args += ['-Wl,-rpath=' + acml]
    print 'Using ACML library'

    output = os.popen('mpicc -showme 2>/dev/null').read()
    if output == '':
        output = os.popen('mpicc -show 2>/dev/null').read()

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

    define_macros.append(('NO_C99_COMPLEX', '1'))
    define_macros.append(('PARALLEL', '1'))

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
            libraries += ['lapack', 'atlas', 'blas']
            library_dirs += [dir]
            print 'Using ATLAS library'
        else:
            libraries += ['blas', 'lapack']

    output = os.popen('mpicc -showme 2>/dev/null').read()
    if output == '':
        output = os.popen('mpicc -show 2>/dev/null').read()

    if output != '':
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


scripts = glob(join('tools', 'gridpaw-*[a-z]'))

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
      )

if sys.platform == 'aix5':

    #
    # o|_  _ _
    # ||_)| | |
    #
    
    # On the IBM, we must make a special Python interpreter with our
    # MPI code built in.  This sucks, but there is no other way!
    
    import distutils
    import distutils.sysconfig
    import os  

    cfgDict = distutils.sysconfig.get_config_vars()

    mpicompiler = 'mpcc_r'
    sources = 'c/_gridpaw.c'

    cmd = ('%s -DGRIDPAW_INTERPRETER=1 %s %s -o tools/gridpaw-python' +
           ' -I%s %s %s -L%s -lpython%s %s %s %s %s %s') % \
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



if ('PARALLEL', '1') not in define_macros:
    msg += ['* Only a serial version of gridpaw was build!']
    

for line in msg:
    print line

