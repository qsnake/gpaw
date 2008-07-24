#!/usr/bin/env python
from sys import argv
import os
options = ' '.join(argv[1:-1])
job = argv[-1]
dir = os.getcwd()
f = open('script.sh', 'w')
f.write("""\
NP=`wc -l < $PBS_NODEFILE`
cd %s
mpirun -np $NP -machinefile $PBS_NODEFILE gpaw-python %s
""" % (dir, job))
f.close()
os.system('qsub ' + options + ' script.sh')
