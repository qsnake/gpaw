.. _updating_libxc:

==============
Updating libxc
==============

GPAW uses functionals from `libxc
<http://www.tddft.org/programs/octopus/wiki/index.php/Libxc>`_ - the
source code of which is in the trunk of GPAW under :file:`c/libxc`.
Here is how to update gpaw to use the latest version of libxc:

 - backup the current version of libxc (you will need it later!)::

    cd ~/gpaw/c; mv libxc libxc.old

 - download libxc from svn `<http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`_::
 
    cd ~/gpaw/c
    svn co http://www.tddft.org/svn/octopus/trunk/libxc

   Note the version number!

 - remove libxc svn directories::

    cd libxc; rm -rf .svn; rm -rf */.svn

 - copy gpaw's svn directories (this is where you need :file:`libxc.old` created in the first step), e.g.::

    cp -rp ../libxc.old/.svn .
    cp -rp ../libxc.old/src/.svn src
    ...

 - check the svn status::

    svn status

   and (svn add)/(svn remove) necessary files.

 - create libxc header files (at the time of writing :file:`config.h` and :file:`src/xc_funcs.h`),
   and the automatically generated c-code (at the time of writing :file:`src/work_*.c`)::

    autoreconf -i
    ./configure --disable-fortran
    make
    
 - check svn status again (files could have changed names compared to the previous release of libxc)::

    svn status

   **Remember**: do **not** add automatically generated files (like :file:`src/Makefile` and many others to gpaw's svn)!

If you have made changes (e.g. added new functionals) to the libxc
included in gpaw or just updated to the latest libxc, change the
version number :file:`self.version` in :file:`gpaw/libxc.py` and make
sure to run from the top level (important!) directory of gpaw::

  [~]$ cd gpaw
  [gpaw]$ python gpaw/libxc.py

This will generate :file:`gpaw/libxc_functionals.py` python-dictionary
file of available functionals, based on the :file:`c/libxc/src/xc.h` file.

Now, build :ref:`c_extension` according to :ref:`developer_installation`,
and :ref:`running_tests`.
