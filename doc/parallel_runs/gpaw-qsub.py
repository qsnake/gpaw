#!/usr/bin/env python

from sys import argv
import os

for i in range(1, len(argv)):
    if argv[i].endswith('.py'):
        script = argv[i]
        break
    
options = ' '.join(argv[1:i])

job = ' '.join(argv[i:])

qsub = '#!/usr/bin/env python\n'
qsub += '#PBS -N %s\n' % script.split('/')[-1] # set default job name

for line in open(script, 'r'):
    if line.startswith('#PBS'):
        qsub += line

# Lock script until it starts running:
os.chmod(script, 0444)

qsub += """\
import os
import sys
from gpaw import get_gpaw_python_path

nodename = os.uname()[1]
c = nodename[0]
assert c in 'nmpqtu'

if c in 'nmq':
    # Normal Niflheim node:
    mpirun = "export LD_LIBRARY_PATH=/usr/lib:/usr/lib/openmpi/1.2.5-gcc/lib&&export PATH=/usr/lib/openmpi/1.2.5-gcc/bin:${PATH}&&mpirun"

elif c == 'p':
    # Infiniband:
    np = len(open(os.environ['PBS_NODEFILE']).readlines())
    mpirun = "export LD_LIBRARY_PATH=/usr/lib:/usr/lib/openmpi/1.2.5-gcc/lib&&export PATH=/usr/lib/openmpi/1.2.5-gcc/bin:${PATH}&&mpirun -np %d -machinefile $PBS_NODEFILE" % np
    
elif c in 'tu':
    # s50:
    mpirun = "export LD_LIBRARY_PATH=/usr/lib:/usr/lib/openmpi/1.2.5-gcc/lib&&export PATH=/usr/lib/openmpi/1.2.5-gcc/bin:${PATH}&&mpirun"
"""
# Release script:
qsub += "os.chmod('%s', 0644)\n" % script

# Start script:
qsub += """
path = get_gpaw_python_path()
os.system('%s %s/gpaw-python JOB' % (mpirun, path))
""".replace('JOB', job)

os.popen('qsub ' + options, 'w').write(qsub)
