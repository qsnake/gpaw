#!/bin/sh

if test -z $GPAW_HOME;
    then
    echo "Error: \$GPAW_HOME variable not set"
    exit 1
fi

rm -rf $GPAW_HOME/build/
#echo "cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/slid-ethernet.py build_ext 2>&1 | tee compile-slid-ethernet.log" | ssh slid bash
#echo "cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/slid-infiniband.py build_ext 2>&1 | tee compile-slid-infiniband.log" | ssh slid bash
echo "source /home/camp/modulefiles.sh&& module load open64/4.2.3-0&& module load NUMPY&& cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el5-xeon-open64-goto2-1.13-acml-4.4.0.py build_ext 2>&1 | tee compile-el5-xeon-open64-goto2-1.13-acml-4.4.0.log" | ssh thul bash
echo "source /home/camp/modulefiles.sh&& module load open64/4.2.3-0&& module load NUMPY&& cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el5-opteron-open64-goto2-1.13-acml-4.4.0.py build_ext 2>&1 | tee compile-el5-opteron-open64-goto2-1.13-acml-4.4.0.log" | ssh fjorm bash
echo "source /home/camp/modulefiles.sh&& module load open64/4.2.3-0&& module load NUMPY&& cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el5-opteron-infiniband-open64-goto2-1.13-acml-4.4.0.py build_ext 2>&1 | tee compile-el5-opteron-infiniband-open64-goto2-1.13-acml-4.4.0.log" | ssh fjorm bash
# TAU
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& module load TAU&&cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el5-opteron-gcc43-goto2-1.13-acml-4.4.0-TAU.py build_ext 2>&1 | tee compile-el5-opteron-gcc43-goto2-1.13-acml-4.4.0-TAU.log" | ssh fjorm bash
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& module load TAU&&cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/el5-xeon-gcc43-goto2-1.13-acml-4.4.0-TAU.py build_ext 2>&1 | tee compile-el5-xeon-gcc43-goto2-1.13-acml-4.4.0-TAU.log" | ssh thul bash
