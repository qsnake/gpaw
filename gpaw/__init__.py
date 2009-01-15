# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.



# XXX
# XXX
# XXX Use random number generator objects
# XXX
# XXX

"""Main gpaw module.

Use like this::

  from gpaw import Calculator

"""

import os
import sys
from distutils.util import get_platform
from glob import glob
from os.path import join, isfile

__all__ = ['GPAW', 'Calculator',
           'Mixer', 'MixerSum', 'MixerDif',
           'PoissonSolver',
           'restart']


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class PoissonConvergenceError(ConvergenceError):
    pass


# Check for special command line arguments:
debug = False
trace = False
dry_run = False
dry_run_size = 1
parsize = None
parsize_bands = None
sl_diagonalize = False
sl_inverse_cholesky = False
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
    elif arg.startswith('--dry-run'):
        dry_run = True
        if len(arg.split('=')) == 2:
            dry_run_size = int(arg.split('=')[1])
    elif arg.startswith('--setups='):
        setup_paths = arg.split('=')[1].split(':')
    elif arg.startswith('--domain-decomposition='):
        parsize = [int(n) for n in arg.split('=')[1].split(',')]
    elif arg.startswith('--state-parallelization='):
        parsize_bands = int(arg.split('=')[1])
    elif arg.startswith('--sl_diagonalize='):
        # --sl_diagonalize=nprow,npcol,mb,cpus_per_node # see c/scalapack.c
        # use 'd' for the default of one or more of the parameters
        # --sl_diagonalize=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_diagonalize = ['d']*4
        else:
            sl_diagonalize = []
            assert len(sl_args) == 4
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_diagonalize.append(int(sl_args[sl_args_index]))
                else:
                    sl_diagonalize.append(sl_args[sl_args_index])
    elif arg.startswith('--sl_inverse_cholesky='):
        # --sl_inverse_cholesky=nprow,npcol,mb,cpus_per_node # see c/sl_inverse_cholesky.c
        # use 'd' for the default of one or more of the parameters
        # --sl_inverse_cholesky=default to use all default values
        sl_args = [n for n in arg.split('=')[1].split(',')]
        if len(sl_args) == 1:
            assert sl_args[0] == 'default'
            sl_inverse_cholesky = ['d']*4
        else:
            sl_inverse_cholesky = []
            assert len(sl_args) == 4
            for sl_args_index in range(len(sl_args)):
                assert sl_args[sl_args_index] is not None
                if sl_args[sl_args_index] is not 'd':
                    assert int(sl_args[sl_args_index]) > 0
                    sl_inverse_cholesky.append(int(sl_args[sl_args_index]))
                else:
                    sl_inverse_cholesky.append(sl_args[sl_args_index])
    else:
        i += 1
        continue
    # Delete used command line argument:
    del sys.argv[i]

if 0:
    import numpy
    oldsum = numpy.sum
    def zum(*args, **kwargs):
        if numpy.asarray(args[0]).ndim != 1 and 'axis' not in kwargs:
            raise RuntimeError
        return oldsum(*args, **kwargs)
    numpy.sum = zum

if debug:
    import numpy
    oldempty = numpy.empty
    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        if a.dtype == int:
            a[:] = -100000000
        else:
            a[:] = numpy.inf
        return a
    numpy.empty = empty

if debug:
    import numpy
    olddot = numpy.dot
    def dot(a, b):
        a = numpy.asarray(a)
        b = numpy.asarray(b)
        if (a.ndim == 1 and b.ndim == 1 and
            (a.dtype == complex or b.dtype == complex)):
            if 1:
                #print 'Warning: Bad use of dot!'
                from numpy.core.multiarray import dot
                return dot(a, b)
            else:
                raise RuntimeError('Bad use of dot!')
        else:
            return olddot(a, b)
    numpy.dot = dot

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


paths = os.environ.get('GPAW_SETUP_PATH', '')
if paths != '':
    setup_paths += paths.split(':')

from gpaw.aseinterface import Calculator
from gpaw.mixer import Mixer, MixerSum, MixerDif
from gpaw.poisson import PoissonSolver

GPAW = Calculator

def restart(filename, Class=Calculator, **kwargs):
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc


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
