.. _Ubuntu:

======
Ubuntu
======

Here you find information about the the system
`<http://www.ubuntu.com/>`_.

Install the required packages, listed below, then :ref:`download
<download>` GPAW trunk or 0.3 and modify :file:`.bashrc` as detailed
in the :ref:`installationguide`.

Version 8.10
------------

Install these packages:

* python-dev
* python-numpy
* liblapack-dev

Optional:

* python-scientific
* python-matplotlib
* python-sphinx
* povray

Sphinx and povray are necessary only to build the documentation.

For your pasting convenience::
  
  sudo apt-get install python-dev python-numpy liblapack-dev python-scientific python-matplotlib python-sphinx povray

If using GPAW 0.3, Numeric is required rather than numpy (as below).

Version 8.04 or earlier
-----------------------

Install these packages:

* python-dev
* lapack3
* lapack3-dev
* refblas3
* refblas3-dev
* build-essential

If using GPAW 0.3, then:

* python-numeric
* python-numeric-ext

Else, if using trunk:

* python-numpy
* python-numpy-ext

Optional:

* atlas3-base
* atlas3-base-dev
* atlas3-headers
* python-scientific

GPAW will use atlas3 if available, which should increase performance. Python-scientific is not strictly necessary, but some tests require it. Some packages in build-essential are likewise not necessary.
