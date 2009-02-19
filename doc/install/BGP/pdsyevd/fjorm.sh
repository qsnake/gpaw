#!/bin/bash
#PBS -l nodes=1:ppn=4
. /usr/local/packages/openmpi-1.3-1.gfortran/bin/mpivars-1.3.sh
mpirun -np 4 test.exe
