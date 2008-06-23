Setting up your LINUX environment
=================================

:svn:`examples/H.py`

.. contents::
.. section-numbering::

.. _setups: http://wiki.fysik.dtu.dk/stuff/gpaw-setups-course.tar.gz

First, find out what type of shell you are using.  Type:

.. highlight:: bash
 
::

  echo $SHELL

If you are running a ``bash`` shell, include the following lines lines
in your ``.bashrc`` file:

::

  $ esm_home=~v40082
  $ export PYTHONPATH=${esm_home}/gpaw:${esm_home}/ase3000:/usr/local/gbar/lib/pythonmodules
  $ export GPAW_SETUP_PATH=${esm_home}/gpaw-setups-0.4/setups:${GPAW_SETUP_PATH}
  $ export PATH=${esm_home}/gpaw/tools:${esm_home}/ase3000/tools:/appl/htools/vmd/bin:${PATH}
  $ export PYTHONSTARTUP=${HOME}/.pythonrc
  $ alias python="fysikpython"
