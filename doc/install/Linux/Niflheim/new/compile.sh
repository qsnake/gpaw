rm -rf $GPAW/build/
echo "cd $GPAW;python setup.py --customize=customize-slid-ethernet.py build_ext > compile-slid-ethernet.log" | ssh slid tcsh
echo "cd $GPAW;python setup.py --customize=customize-slid-infiniband.py build_ext > compile-slid-infiniband.log" | ssh slid tcsh
echo "cd $GPAW;python setup.py --remove-default-flags --customize=customize-fjorm.py build_ext > compile-fjorm.log" | ssh fjorm tcsh
echo "cd $GPAW;python setup.py --remove-default-flags --customize=customize-thul.py build_ext > compile-thul.log" | ssh thul tcsh
