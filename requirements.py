#!/usr/bin/env python

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import sys
import os
from glob import glob


print 'Checking requirements ...'
print

ok = True

if sys.version_info < (2, 2, 1, 'final', 0):
    print 'Python 2.2.2 or later is required!'
    print 'Try "python2 requirements.py".  If that works, you will need'
    print 'to substitute "python2" for "python" in all further commands'
    print 'you use in relation to gridpaw.'
    print
    ok = False
    
try:
    import Numeric
except ImportError:
    print 'Numeric is not installed'
    print
    ok = False

try:
    import Scientific.IO.NetCDF
except ImportError:
    # find netcdf.h
    try:
        import Scientific
    except ImportError:
        print 'Scientific is not installed'
        print
    else:
        print 'Scientific.IO.NetCDF is not installed.'
        print 'The NetCDF C-library is probably not installed.'
        print 
    ok = False

# find arrayobject.h

try:
    import ASE
except ImportError:
    print 'ASE is not installed!  You may be able to install gridpaw, but'
    print "you can't use it without ASE!"
    print
    ok = False


mpi = (os.popen('mpicc -showme').read() != '') # XXX

# BLAS and LAPACK???

"""
for prefix in ['/usr/local', '/usr']:
    if os.path.exists(os.path.join(netcdf_include, 'netcdf.h')):
                break
find arrayobject.h (in setup.py ??)
/usr/include/python2.2/Numeric/arrayobject.h
if netcdf_prefix is None:
    try:
        netcdf_prefix=os.environ['NETCDF_PREFIX']
    except KeyError:
        for netcdf_prefix in ['/usr/local', '/usr']:
            netcdf_include = os.path.join(netcdf_prefix, 'include')
            netcdf_lib = os.path.join(netcdf_prefix, 'lib')
            if os.path.exists(os.path.join(netcdf_include, 'netcdf.h')):
                break
        else:
            netcdf_prefix = None
"""

if ok:
    print 'You should be ready to install and use gridpaw.  If you'
    print 'experience any problems, please let us know!'
else:
    print 'You need to install the missing pieces before you can'
    print 'install and use gridpaw!'

if not mpi:
    print
    print 'Could not find the "mpi.h" include file:  only a serial version'
    print 'of gridpaw can be build!'
    
print



