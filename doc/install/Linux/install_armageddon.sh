export APPS="/home/firegam/CAMd"
export MODULEFILES="${APPS}/modulefiles"

# build packages

openmpi_version=1.4.3
tar jxf openmpi-${openmpi_version}.tar.bz2
cd openmpi-${openmpi_version}
./configure --prefix=${APPS}/openmpi-${openmpi_version}-1
make && make install
cd ..

nose_version=0.11.3
tar zxf nose-${nose_version}.tar.gz
cd nose-${nose_version}
python setup.py install --root=${APPS}/nose-${nose_version}-1
cd ..

numpy_version=1.5.0
tar zxf numpy-${numpy_version}.tar.gz
cd  numpy-${numpy_version}
# disable compiling with atlas
sed -i "s/_lib_atlas =.*/_lib_atlas = ['ignore_atlas']/g" numpy/distutils/system_info.py
python setup.py install --root=${APPS}/numpy-${numpy_version}-1
cd ..

ase_version=3.4.1.1765
tar zxf python-ase-${ase_version}.tar.gz

gpaw_version=0.7.2.6974
tar zxf gpaw-${gpaw_version}.tar.gz

gpaw_setups_version=0.6.6300
tar zxf gpaw-setups-${gpaw_setups_version}.tar.gz

. set_env_armageddon.sh

# test numpy
python -c "import numpy; numpy.test()"
# test ase
mkdir -p testase
cd testase
testase.py --no-display 2>&1 | tee testase.log
cd ..
# build gpaw
cd gpaw-${gpaw_version}
python setup.py build_ext --customize=../customize_armageddon.py --remove-default-flags
cd ..
mkdir -p testgpaw
cd testgpaw
mpiexec -np 4 gpaw-python `which gpaw-test` 2>&1 | tee testgpaw.log
