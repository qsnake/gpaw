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

.. note::

   In order to use the code, you need also the setups for all your atoms (:ref:`setups`).

.. _NumPy: http://www.scipy.org/NumPy

See the :ref:`platforms_and_architectures` page for information on how to
install GPAW on a specific architecture.

Standard installation
=====================

0) assuming bash

.. note::

   **CAMd users** installing on ``Niflheim``: please follow instructions for :ref:`Niflheim`.

1) :ref:`download` the code.

2) Go to the :file:`gpaw` directory::

     [~]$ cd gpaw

3) install with the standard::

     [gpaw]$ python setup.py install --home=<my-directory>  2>&1 | tee install.log

   and put :file:`{<my-directory>}/lib/python` (or
   :file:`{<my-directory>}/lib64/python`) in your :envvar:`$PYTHONPATH` 
   environment variable. Moreover, if parallel environment is found on your system,
   a special :program:`gpaw-python` python-interpreter is created under
   :file:`{<my-directory>}/bin`. Please add
   :file:`{<my-directory>}/bin` to :envvar:`PATH`. Alternatively, the full pathname
   :file:`{<my-directory}>/bin/gpaw-python` can be used when executing
   parallel runs. See :ref:`parallel_installation` for details.

   .. note::

     Usually :envvar:`$HOME` is a good choice for :file:`{<my-directory>}`.

   Alternatively, if you have root-permissions, you can install GPAW system-wide::

     [gpaw]$ python setup.py install 2>&1 | tee install.log

   .. note::

    The installation described here is suitable only as a first try:

     - if you install on a cluster, please follow :ref:`install_custom_installation`,

     - if you are a developer, please follow :ref:`developer_installation`.

4) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz` from the 
   :ref:`setups` page
   and unpack it somewhere, preferably in :envvar:`${HOME}`
   (``cd; tar zxf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw/`.  There will
   now be a directory :file:`gpaw-setups-{<version>}/` containing all the
   atomic data needed for doing LDA, PBE, and RPBE calculations.  Set the
   environment variable :envvar:`GPAW_SETUP_PATH` to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. put into :file:`~/.tcshrc`::

    setenv GPAW_SETUP_PATH ${HOME}/gpaw-setups-<version>

   or if you use bash, put these lines into :file:`~/.bashrc`::

    export GPAW_SETUP_PATH=${HOME}/gpaw-setups-<version>

5) Make sure that everything works by running the test suite::

     [gpaw]$ cd test
     [test]$ python test.py 2>&1 | tee test.log

   This will take around 20 minutes.  Please report errors to the
   `GPAW developer mailing list`_
   (send us :file:`test.log`, and (only when requested) :file:`install.log`).

  .. _GPAW developer mailing list: gridpaw-developer@lists.berlios.de

.. _install_custom_installation:

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
itself for more possible options.
:ref:`platforms_and_architectures` provides examples of :file:`customize.py` for different platforms.
After editing :svn:`customize.py`,
follow the instructions for the :ref:`installationguide` from step 3 on.

.. _parallel_installation:

Parallel installation
=====================

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :svn:`customize.py` file.

Additionally a user may want to enable scalapack, setting in :file:`customize.py`::

 scalapack = True

and, if needed, providing blacs/scalapack `libraries` and `library_dirs`
as described in :ref:`install_custom_installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.
