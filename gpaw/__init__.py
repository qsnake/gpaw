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
setup_paths = []
dry_run = 0
parsize = None
parsize_bands = None
sl_diagonalize = False
sl_inverse_cholesky = False
extra_parameters = {}
profile = False
i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--gpaw-'):
        # Found old-style gpaw command line argument:
        arg = '--' + arg[7:]
        raise RuntimeError('Warning: Use %s instead of %s.' %
                           (arg, sys.argv[i]))
    if arg == '--trace':
        trace = True
    elif arg == '--debug':
        debug = True
        sys.stderr.write('gpaw-DEBUG mode\n')
    elif arg.startswith('--setups='):
        setup_paths = arg.split('=')[1].split(':')
    elif arg.startswith('--dry-run'):
        dry_run = 1
        if len(arg.split('=')) == 2:
            dry_run = int(arg.split('=')[1])
    elif arg.startswith('--domain-decomposition='):
        parsize = [int(n) for n in arg.split('=')[1].split(',')]
        if len(parsize) == 1:
            parsize = parsize[0]
        else:
            assert len(parsize) == 3
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
    elif arg.startswith('--gpaw='):
        extra_parameters = eval('dict(%s)' % arg[7:])
    elif arg == '--gpaw':
        extra_parameters = eval('dict(%s)' % sys.argv.pop(i + 1))
    elif arg.startswith('--profile='):
        profile = arg.split('=')[1]
    else:
        i += 1
        continue
    # Delete used command line argument:
    del sys.argv[i]

if debug:
    import numpy
    numpy.seterr(over='raise', divide='raise', invalid='raise', under='ignore')

    oldempty = numpy.empty
    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        try:
            a[:] = numpy.nan
        except:
            a[:] = -100000000
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

from gpaw.aseinterface import GPAW
from gpaw.mixer import Mixer, MixerSum, MixerDif
from gpaw.poisson import PoissonSolver

class Calculator(GPAW):
    def __init__(self, *args, **kwargs):
        sys.stderr.write('Please start using GPAW instead of Calculator!\n')
        GPAW.__init__(self, *args, **kwargs)

def restart(filename, Class=GPAW, **kwargs):
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc

if trace:
    indent = '    '
    path = __path__[0]
    from gpaw.mpi import parallel, rank
    if parallel:
        indent = 'CPU%d    ' % rank
    def f(frame, event, arg):
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

    sys.setprofile(f)

if profile:
    from cProfile import Profile
    import atexit
    prof = Profile()
    def f(prof, filename):
        prof.disable()
        from gpaw.mpi import rank
        if filename == '-':
            prof.print_stats('time')
        else:
            prof.dump_stats(filename + '.%04d' % rank)
    atexit.register(f, prof, profile)
    prof.enable()


# Shut down all MPI tasks if one of them crash
from gpaw.mpi import parallel
if 0:#parallel:
    import ase.parallel
    ase.parallel.register_parallel_cleanup_function()
