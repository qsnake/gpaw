.. _Ubuntu:

======
Ubuntu
======

Here you find information about the the system
`<http://www.ubuntu.com/>`_.

Install the required packages, listed below, then :ref:`download
<download>` GPAW trunk or stable and modify :file:`.bashrc` as detailed
in the :ref:`installationguide`.

Version 8.10 or newer
---------------------

Required packages:

* python-dev
* python-numpy
* liblapack-dev

Recommended:

* python-scientific
* python-matplotlib

Parallelization with OpenMPI:

* openmpi-bin
* libopenmpi-dev

Parallelization with MPI-LAM and BLACS/ScaLAPACK:

* lam-runtime
* scalapack1-lam
* scalapack-lam-dev
* blacs1gf-lam
* blacsgf-lam-dev

This also requires building GPAW with the customize-file
:svn:`~doc/install/Linux/customize-ubuntu-sl-blacs-lam.py`:

.. literalinclude:: customize-ubuntu-sl-blacs-lam.py

Building documentation:

* python-sphinx
* povray

Sphinx and povray are necessary only to build the documentation.

For your pasting convenience::
  
  sudo apt-get install python-dev python-numpy liblapack-dev python-scientific python-matplotlib python-sphinx povray

Version 8.04 or earlier
-----------------------

Install these packages:

* python-dev
* lapack3
* lapack3-dev
* refblas3
* refblas3-dev
* build-essential
* python-numpy
* python-numpy-ext

Optional:

* atlas3-base
* atlas3-base-dev
* atlas3-headers
* python-scientific

GPAW will use atlas3 if available, which should increase performance. 
Python-scientific is not strictly necessary, but some tests require it. 
Some packages in build-essential are likewise not necessary.
