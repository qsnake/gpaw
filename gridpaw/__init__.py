# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
import signal
from distutils.util import get_platform
from glob import glob
from os.path import join

import Numeric as num


class GPError(Exception):
    pass


home = os.environ['HOME']

# Check for special command line arguments:
debug = False
hosts = None
parsize = None
setup_home = os.path.join(home, '.gridpaw/setups')
gpspecial = None
arg = None
i = 1
while len(sys.argv) > i:
    arg = sys.argv[i]
    if arg.startswith('--gridpaw-'):
        # We have found a gridpaw command line argument:
        if arg == '--gridpaw-debug':
            debug = True
            print >> sys.stderr, 'gridpaw-DEBUG mode'
        elif arg.startswith('--gridpaw-setups='):
            setup_home = arg.split('=')[1]
        elif arg == '--gridpaw-poly':
            GAUSS = False
        elif arg.startswith('--gridpaw-special='):
            gpspecial = arg.split('=')[1]
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

try:
    import _gridpaw
except ImportError:
    # gridpaw has not been installed correctly.  Maybe we can find the
    # extension in the distutils build directory:
    sys.path.insert(0, join(__path__[0], '..', 'build',
                            'lib.%s-%s' % (get_platform(), sys.version[0:3])))


# Install call-back handler for USR1 signal:
# (use "kill -s USR1 <pid>" to stop calculation)
sigusr1 = [False]
def cb_handler(number, frame):
    sigusr1[0] = True
signal.signal(signal.SIGUSR1, cb_handler)


from gridpaw.calculator import Calculator
## def Calculator(*args, **kwargs):
##     # Delayed import:
##     from gridpaw.calculator import Calculator
##     return Calculator(*args, **kwargs)

# Clean up:
del os, sys, get_platform, i, arg
