#!/bin/sh

if test -z $GPAW_HOME;
    then
    echo "Error: \$GPAW_HOME variable not set"
    exit 1
fi

rm -rf $GPAW_HOME/build/
echo "cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-slid-ethernet.py build_ext 2>&1 | tee compile-slid-ethernet.log" | ssh slid bash
echo "cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-slid-infiniband.py build_ext 2>&1 | tee compile-slid-infiniband.log" | ssh slid bash
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-fjorm.py build_ext 2>&1 | tee compile-fjorm.log" | ssh fjorm bash
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-thul-acml.py build_ext 2>&1 | tee compile-thul-acml.log" | ssh thul bash
# TAU
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& module load TAU&&cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-fjorm-TAU.py build_ext 2>&1 | tee compile-fjorm-TAU.log" | ssh fjorm bash
echo "source /home/camp/modulefiles.sh&& module load NUMPY&& module load TAU&&cd $GPAW_HOME&& python setup.py --remove-default-flags --customize=./doc/install/Linux/Niflheim/customize-thul-acml-TAU.py build_ext 2>&1 | tee compile-thul-acml-TAU.log" | ssh thul bash
