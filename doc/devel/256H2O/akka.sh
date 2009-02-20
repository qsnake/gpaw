#!/bin/bash
### SNAC project number, enter if applicable.
### NOTE! No spaces or slashes allowed
#PBS -A HPC2N-2008-005
### Requesting 2 nodes with 2 VP:s on each node
#PBS -l nodes=10:ppn=6
### Requesting time - 10 minutes - defaults to
#PBS -l walltime=00:20:00

# Change to Working Directory
cd $PBS_O_WORKDIR

module add openmpi/1.2.6/gcc

gpawhome=${HOME}/gpaw
export PYTHONPATH=${gpawhome}:${PYTHONPATH}
mpiexec ${gpawhome}/build/bin.linux-x86_64-2.4/gpaw-python b256H2O.py --sl_diagonalize=4,4,64,8
