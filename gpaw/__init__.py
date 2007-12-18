# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gpaw module.

Use like this::

  from gpaw import Calculator

"""

import os
import sys
from distutils.util import get_platform
from glob import glob
from os.path import join, isfile

__all__ = ['Calculator', 'Mixer', 'MixerSum', 'PoissonSolver']


class ConvergenceError(Exception):
    pass

# Check for special command line arguments:
debug = False
trace = False
dry_run = False
parsize = None
arg = None
setup_paths = []
i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--gpaw-'):
        # Found old-style gpaw command line argument:
        arg = '--' + arg[7:]
        print 'Warning: %s is prefered instead of %s' % (arg, sys.argv[i])
    if arg == '--trace':
        trace = True
    elif arg == '--debug':
        debug = True
        print >> sys.stderr, 'gpaw-DEBUG mode'
    elif arg == '--dry-run':
        dry_run = True
    elif arg.startswith('--setups='):
        setup_paths = arg.split('=')[1].split(':')
    elif arg.startswith('--domain-decomposition='):
        parsize = [int(n) for n in arg.split('=')[1].split(',')]
    else:
        i += 1
        continue
    # Delete used command line argument:
    del sys.argv[i]

if debug:
    import Numeric
    oldempty = Numeric.empty
    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        if a.typecode() == Numeric.Int:
            a[:] = -100000000
        else:
            a[:] = 1e400
        return a
    Numeric.empty = empty

build_path = join(__path__[0], '..', 'build')
arch = '%s-%s' % (get_platform(), sys.version[0:3])

# If we are running the code from the source directory, then we will
# want to use the extension from the distutils build directory:
sys.path.insert(0, join(build_path, 'lib.' + arch))

def get_gpaw_python_path():
    paths = os.environ['PATH'].split(os.pathsep)
    paths.insert(0, join(build_path, 'bin.' + arch))
    for path in paths:
        if isfile(join(path, 'gpaw-python')):
            return path
    raise RuntimeError('Could not find gpaw-python!')

import Numeric
from gpaw.utilities.blas import dotc
Numeric.vdot = dotc



paths = os.environ.get('GPAW_SETUP_PATH', '')
if paths != '':
    setup_paths += paths.split(':')

from gpaw.ase import Calculator
from gpaw.mixer import Mixer, MixerSum
from gpaw.poisson import PoissonSolver

if trace:
    indent = '    '
    path = __path__[0]
    from gpaw.mpi import parallel, rank
    if parallel:
        indent = 'CPU%d    ' % rank
    def profile(frame, event, arg):
        global indent
        f = frame.f_code.co_filename
        if not f.startswith(path):
            return
        
        if event == 'call':
            print '%s%s:%d(%s)' % (indent, f[len(path):], frame.f_lineno,
                                   frame.f_code.co_name)
            indent += '| '
        elif event == 'return':
            indent = indent[:-2]
        
    sys.setprofile(profile)
