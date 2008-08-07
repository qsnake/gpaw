.. _installationguide:

==================
Installation guide
==================

Requirements
============

1) Python 2.3 or later is required.  Python is available from http://www.python.org.

2) Atomic Simulation Environment (:ase:`ASE <>`).

3) NumPy_.

4) BLAS and LAPACK libraries.

5) An MPI library is required for parallel calculations.


.. _NumPy: http://www.scipy.org/NumPy

See the :ref:`platforms_and_architectures` page for information on how to
install GPAW on a specific architecture.


Standard installation
=====================

1) :ref:`download` the code.

2) Go to the :file:`gpaw` directory (:file:`gpaw-0.3` if you use the tar-ball)::

     [~]$ cd gpaw

3) or, alternatively, install with the standard::

     [gpaw]$ python setup.py install

   This step requires root permissions - if you don't have that, just do a::

     [gpaw]$ python setup.py install --home=<my-directory>

   and put :file:`{<my-directory>}/lib/python` (or
   :file:`{<my-directory>}/lib64/python`) in your :envvar:`$PYTHONPATH` 
   environment variable.  Usually :envvar:`$HOME` is a good coice for
   :file:`{<my-directory>}`.

4) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz` from the 
   :ref:`setups` page
   and unpack it somewhere, preferably in :envvar:`${HOME}`
   (``cd; tar zxf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw/`.  There will
   now be a directory :file:`gpaw-setups-{<version>}/` containing all the
   atomic data needed for doing LDA and PBE calculations.  Set the
   environment variable :envvar:`GPAW_SETUP_PATH` to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. put into :file:`~/.tcshrc`::

    setenv GPAW_SETUP_PATH ${HOME}/gpaw-setups-<version>

   or if you use bash, put these lines into :file:`~/.bashrc`::

    export GPAW_SETUP_PATH=${HOME}/gpaw-setups-<version>

5) Make sure that everything works by running the test suite::

     [gpaw]$ cd test
     [test]$ python test.py

   This will take around 20 minutes.  Please report errors to the `GPAW 
   developer mailing list`_

  .. _GPAW developer mailing list: gridpaw-developer@lists.berlios.de

If you are a developer, you will want to install the code in a
different way to allow code updates via SVN update.  See
:ref:`developer_installation`.



Custom installation
===================

The install script does its best when trying to guess proper libraries
and commands to build gpaw. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :svn:`customize.py` file which is located in the gpaw base
directory. As an example, :svn:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, gpaw would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :svn:`customize.py`
itself for more possible options. After editing :svn:`customize.py`,
follow the instructions for the standard installation from step 3 on.



Parallel installation
=====================

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :svn:`customize.py` file.

For the parallel calculations, a special :program:`gpaw-python`
python-interpreter is created. If GPAW was installed without root
permissions, i.e.::

  python setup.py install --home=<my-directory>

:file:`{<my-directory>}/bin` should be added to
:envvar:`PATH`. Alternatively, the full pathname
:file:`{<my-directory}>/bin/gpaw-python` can be used when executing
parallel runs.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.
