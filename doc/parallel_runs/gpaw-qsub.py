#!/usr/bin/env python

from sys import argv
from os import popen

for i in range(1, len(argv)):
    if argv[i].endswith('.py'):
        break
    
options = ' '.join(argv[1:i])

gpaw = ''
if argv[-1].startswith('--gpaw='):
    gpaw = argv.pop().split('=')[-1]

job = ' '.join(argv[i:])

qsub = '#!/usr/bin/env python\n'
qsub += '#PBS -N %s\n' % argv[i].split('/')[-1] # set default job name

for line in open(argv[i], 'r'):
    if line.startswith('#PBS'):
        qsub += line

qsub += """\
import os
import sys
gpaw_path = 'GPAW'
if gpaw_path != '':
    sys.path.insert(0, gpaw_path)
from gpaw import get_gpaw_python_path

nodename = os.uname()[1]
c = nodename[0]
assert c in 'nmpqtu'
if c in 'nmq':   # OpenMPI 1.2.3:
    mpirun = ('. /usr/local/openmpi-1.2.3-pathf90/bin/mpivars-1.2.3.sh && ' +
              'mpirun')
elif c == 'p':   # Infiniband:
    mpirun = 'mpirun -np %d -machinefile $PBS_NODEFILE'
else:            # OpenMPI 1.2.5:
    mpirun = ('. /usr/local/openmpi-1.2.5-gfortran/bin/mpivars-1.2.5.sh && ' +
              'mpirun')

if c == 'p':
    np = len(open(os.environ['PBS_NODEFILE']).readlines())
    mpirun = mpirun % np

path = get_gpaw_python_path()
print mpirun, sys.path, path
os.system('PYTHONPATH=GPAW:$PYTHONPATH %s %s/gpaw-python JOB' % (mpirun, path))
""".replace('JOB', job).replace('GPAW', gpaw)

popen('qsub ' + options, 'w').write(qsub)
