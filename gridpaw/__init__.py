# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import generators
import os
import sys
import signal
from distutils.util import get_platform
from glob import glob
from os.path import join

import Numeric as num


try:
    enumerate
except NameError:
    def enumerate(collection):
        """Generates an indexed series:  (0,coll[0]), (1,coll[1]) ..."""
        x = 0
        it = iter(collection)
        while 1:
            yield (x, it.next())
            x += 1
else:
    enumerate = enumerate

 
home = os.environ['HOME']


class GPError(Exception):
    pass


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
