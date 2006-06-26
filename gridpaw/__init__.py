# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gridpaw module.

Use like this::

  from gridpaw import Calculator

"""

import os
import sys
import signal
from distutils.util import get_platform
from glob import glob
from os.path import join

import Numeric as num


home = os.environ['HOME']

# Check for special command line arguments:
debug = False
parallel = False
hosts = None
parsize = None
x = ''
arg = None
setup_paths = []
i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--gridpaw-'):
        # We have found a gridpaw command line argument:
        if arg == '--gridpaw-debug':
            debug = True
            print >> sys.stderr, 'gridpaw-DEBUG mode'
        elif arg.startswith('--gridpaw-setups='):
            setup_paths = arg.split('=')[1].split(':')
        elif arg == '--gridpaw-poly':
            GAUSS = False
        elif arg == '--gridpaw-parallel':
            parallel = True
        elif arg.startswith('--gridpaw-x='):
            x = arg.split('=')[1]
        elif arg.startswith('--gridpaw-hosts='):
            hosts = arg.split('=')[1].split(',')
            if len(hosts) == 1 and hosts[0].isdigit():
                hosts = int(hosts[0])
        elif arg.startswith('--gridpaw-hostfile='):
            hosts = arg.split('=')[1]
        elif arg.startswith('--gridpaw-domain-decomposition='):
            parsize = [int(s) for s in arg.split('=')[1].split(',')]
        else:
            raise RuntimeError, 'Unknown command line argument: ' + arg
        # Delete used command line argument:
        del sys.argv[i]
    else:
        i += 1

# If we are running the code from the source directory, then we will
# want to use the extension from the distutils build directory:
sys.path.insert(0, join(__path__[0], '..', 'build',
                        'lib.%s-%s' % (get_platform(), sys.version[0:3])))


# Install call-back handler for USR1 signal:
# (use "kill -s USR1 <pid>" to stop a calculation)
sigusr1 = [False]
def cb_handler(number, frame):
    """Call-back handler for USR1 signal."""
    sigusr1[0] = True
#mpich ch_p4 uses SIGUSR1, so in general case we cannot use
#our own signal handler 
#signal.signal(signal.SIGUSR1, cb_handler)


paths = os.environ.get('GRIDPAWSETUPPATH', '')
if paths != '':
    setup_paths += paths.split(':')
if setup_paths == []:
    path = join(home, '.gridpaw/setups')
    if os.path.isdir(path):
        setup_paths = [path]
        print 'Using setups from:', path
        print 'Please put that path in your GRIDPAWSETUPPATH environment',
        print 'variable!'


from gridpaw.calculator import Calculator


# Clean up:
del os, sys, get_platform, i, arg
