.. _installationguide:

==================
Installation guide
==================

Requirements
============

1) Python 2.3 or later is required.  Python is available from http://www.python.org.

2) Atomic Simulation Environment (ASE_).

3) NumPy_.

4) BLAS and LAPACK libraries.

5) An MPI library is required for parallel calculations.


.. _ASE: https://wiki.fysik.dtu.dk/ase
.. _NumPy: http://www.scipy.org/NumPy

See the :ref:`platforms_and_architectures` page for information on how to
install GPAW on a specific architecture.


Standard installation
=====================

1) :ref:`download` the code.

2) Go to the ``gpaw`` directory (``gpaw-0.3`` if you use the tar-ball)::

     [~]$ cd gpaw

3) See :ref:`developer_installation` for the recommended way of installing gpaw,
4) or, alternatively, install with the standard::

     [gpaw]$ python setup.py install

   This step requires root permissions - if you don't have that, just do a::

     [gpaw]$ python setup.py install --home=<my-directory>

   and put ``<my-directory>/lib/python`` (or
   ``<my-directory>/lib64/python``) in your ``PYTHONPATH`` environment
   variable.  Usually ``$HOME`` is a good coice for
   ``<my-directory>``.

4) Get the tar file ``gpaw-setups-<version>.tar.gz`` from the :ref:`setups` page
   and unpack it somewhere, preferably in ``${HOME}`` 
   (``cd; tar zxf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in ``/usr/share/gpaw/``.  There will
   now be a directory ``gpaw-setups-<version>/`` containing all the
   atomic data needed for doing LDA and PBE calculations.  Set the
   environment variable ``GPAW_SETUP_PATH`` to point to the directory
   ``gpaw-setups-<version>/``, e.g. put into ``~/.tcshrc``::

    setenv GPAW_SETUP_PATH ${HOME}/gpaw-setups-<version>

   or if you use bash, put these lines into ``~/.bashrc``::

    export GPAW_SETUP_PATH=${HOME}/gpaw-setups-<version>

5) Make sure that everything works by running the test suite::

     [gpaw]$ cd test
     [test]$ python test.py

   This will take around 20 minutes.  Please report errors to:

   .. macro:: [[MailTo(gridpaw-developer@lists.berlios.de)]]


If you are a developer, you will want to install the code in a
different way to allow code updates via SVN checkout.  See
:ref:`developer_installation`.



Custom installation
===================

The install script does its best when trying to guess proper libraries
and commands to build gpaw. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with ``customize.py`` file which is located in the gpaw base
directory. As an example, ``customize.py`` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, gpaw would be built with "``-Lpath_to_myblas -lmyblas -lmylapack``" linker flags. Look at the file ``customize.py`` itself for more possible options. After editing ``customize.py``, follow the instructions for the standard installation from step 3 on.

Parallel installation
=====================

By default, setup looks if mpicc is available, and if setup finds one, a parallel version is build. If the setup does not find mpicc, a user can specify one in the ``customize.py`` file. 

For the parallel calculations, a special ``gpaw-python`` python-interpreter is created. If gpaw was installed without root permissions, i.e.::

  python setup.py install --home=<my-directory>

``<my-directory>/bin`` should be added to PATH. Alternatively, the full pathname ``<my-directory>/bin/gpaw-python`` can be
used when executing parallel runs.

Instructions for running parallel calculations can be found in the :ref:`user manual <manual#parallel-calculations>`.
