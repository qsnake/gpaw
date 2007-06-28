# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gpaw module.

Use like this::

  from gpaw import Calculator

"""

import os
import sys
import signal
from distutils.util import get_platform
from glob import glob
from os.path import join


class ConvergenceError(Exception):
    pass

home = os.environ['HOME']

# Check for special command line arguments:
debug = False
trace = False
parallel = False
hosts = None
parsize = None
dry_run = False
x = ''
arg = None
setup_paths = []
i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--gpaw-'):
        # We have found a gpaw command line argument:
        if arg == '--gpaw-trace':
            trace = True
        elif arg == '--gpaw-debug':
            debug = True
            print >> sys.stderr, 'gpaw-DEBUG mode'
        elif arg.startswith('--gpaw-setups='):
            setup_paths = arg.split('=')[1].split(':')
        elif arg == '--gpaw-poly':
            GAUSS = False
        elif arg == '--gpaw-parallel':
            parallel = True
        elif arg == '--gpaw-dry-run':
            dry_run = True
        elif arg.startswith('--gpaw-x='):
            x = arg.split('=')[1]
        elif arg.startswith('--gpaw-hosts='):
            hosts = arg.split('=')[1].split(',')
            if len(hosts) == 1 and hosts[0].isdigit():
                hosts = int(hosts[0])
        elif arg.startswith('--gpaw-hostfile='):
            hosts = arg.split('=')[1]
        elif arg.startswith('--gpaw-domain-decomposition='):
            parsize = [int(s) for s in arg.split('=')[1].split(',')]
        else:
            raise RuntimeError, 'Unknown command line argument: ' + arg
        # Delete used command line argument:
        del sys.argv[i]
    else:
        i += 1

if debug:
    import Numeric
    oldempty = Numeric.empty
    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        a[:] = 117
        return a
    Numeric.empty = empty

# If we are running the code from the source directory, then we will
# want to use the extension from the distutils build directory:
sys.path.insert(0, join(__path__[0], '..', 'build',
                        'lib.%s-%s' % (get_platform(), sys.version[0:3])))

import Numeric
from gpaw.utilities.blas import dotc
Numeric.vdot = dotc

# Install call-back handler for USR1 signal:
# (use "kill -s USR1 <pid>" to stop a calculation)
sigusr1 = [False]
def cb_handler(number, frame):
    """Call-back handler for USR1 signal."""
    sigusr1[0] = True
#mpich ch_p4 uses SIGUSR1, so in general case we cannot use
#our own signal handler 
#signal.signal(signal.SIGUSR1, cb_handler)


paths = os.environ.get('GPAW_SETUP_PATH', '')
if paths != '':
    setup_paths += paths.split(':')
if setup_paths == []:
    path = join(home, '.gridpaw/setups')
    if os.path.isdir(path):
        setup_paths = [path]
        print 'Using setups from:', path
        print 'Please put that path in your GPAW_SETUP_PATH environment',
        print 'variable!'

from gpaw.calculator import Calculator


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
    
# Clean up:
del os, sys, get_platform, i, arg
