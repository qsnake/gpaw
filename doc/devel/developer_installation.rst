.. _developer_installation:

======================
Developer installation
======================

The `Standard installation`_ will copy all the Python files to the
standard place for Python modules (something like
``/usr/lib/python2.5/site-packages/gpaw``) or to
``<my-directory>/lib/python/gpaw`` if you used the
``--home=<my-directory>`` option.  As a developer, you will want
Python to use the files from the SVN checkout that you are hacking
on.  Do this::

  [~]$ svn checkout https://USER@svn.fysik.dtu.dk/projects/gpaw/trunk gpaw

**On slid and thul machines only**: if you want to use a parallel version modify `customize.py`::

 libraries += ['gfortran']

and apply the openmpi environment settings::

  [~]$ source /usr/local/openmpi-1.2.5-gfortran/bin/mpivars-1.2.5.csh

To make it the default setting add the line to your ~/.tcshrc. See `<https://wiki.fysik.dtu.dk/niflheim/Parallelization>`_ for details
(note however, that it contains instructions for openmpi fortran codes).

Then::

  [~]$ cd gpaw
  [gpaw]$ python setup.py build_ext

This will build two things:

* ``_gpaw.so``:  A shared library for serial calculations containing
  GPAW's C-extension module.  The module will be in
  ``~/gpaw/build/lib.<platform>-2.5/``.
* ``gpaw-python``: A special Python interpreter for parallel
  calculations.  The interpreter has GPAW's C-code build in.  The
  executable is in ``~/gpaw/build/bin.<platform>-2.5/``.

The ``gpaw-python`` interpreter will only be made if ``setup.py`` found an ``mpicc`` compiler.

Put ``~/gpaw`` in your ``$PYTHONPATH`` and ``~/gpaw/tools:~/gpaw/build/bin.<platform>-2.5`` in your
``$PATH``, e.g. put into ~/.tcshrc::

 setenv PYTHONPATH ${HOME}/gpaw
 setenv PATH ${HOME}/gpaw/build/bin.<platform>-2.5:${PATH}

or if you use bash, put these lines into ~/.bashrc::

 export PYTHONPATH=${HOME}/gpaw
 export PATH=${HOME}/gpaw/build/bin.<platform>-2.5:${PATH}

GPAW uses functionals from `libxc <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc>`_
- the source code of which is in the trunk of GPAW under ``c/libxc``.
Here is how to update gpaw to use the latest version of libxc:

 - backup the current version of libxc (you will need it later!)::

    cd gpaw/c; mv libxc libxc.old   

 - download libxc from svn `<http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`_::
 
    cd gpaw/c
    svn co http://www.tddft.org/svn/octopus/trunk/libxc

   Note the version number!

 - remove libxc svn directories::

    cd libxc; rm -rf .svn; rm -rf */.svn

 - copy gpaw's svn directories (this is where you need libxc.old created in the first step), e.g.::

    cp -rp ../libxc.old/.svn .
    cp -rp ../libxc.old/src/.svn src
    ...

 - check the svn status::

    svn status

   and (svn add)/(svn remove) necessary files.

 - create libxc header files (at the time of writing `config.h` and `src/xc_funcs.h`)
   and the automatically generated c-code (at the time of writing `src/work_*.c`)::

    autoreconf -i
    ./configure --disable-fortran
    make
    
 - check svn status again (files could have changed names compared to the previous release of libxc)::

    svn status

   **Remember**: do not add automatically generated files (like src/Makefile and many others to gpaw's svn)!

If you have made changes (e.g. added new functionals) to the libxc included in gpaw
or just updated to the latest libxc, change the version number ``self.version`` in ``gpaw/libxc.py``
and make sure to run from the top level (important!) directory of gpaw::

  [gpaw]$ python gpaw/libxc.py

This will generate ``gpaw/libxc_functionals.py`` python-dictionary file of available functionals,
based on the ``c/libxc/src/xc.h`` file.

Now, (after the developer installation), test the serial code::

  [gpaw]$ cd test
  [test]$ python test.py

If that works, you can go on and test the parallel code::

  [test]$ cd ..
  [gpaw]$ mpirun -np 2 gpaw-python -c "import gpaw.mpi as mpi; print mpi.rank"
  1
  0

Try also::

  [gpaw]$ cd examples
  [demo]$ mpirun -np 2 gpaw-python H.py

This will do a calculation for a single hydrogen atom parallelized with spin up on one processor and spin down on the other.  If you run the example on 4 processors, you should get parallelization over both spins and the domain.


.. _standard installation:  InstallationGuide_#standard-installation
