CAMd Summer school 2008
=======================

...
================================
Setting up your UNIX environment
================================

This page describes how to make GPAW, ASE, PYTHON and VMD run on your account at ``bohr.gbar.dtu.dk``.

First, find out what type of shell you are using.  Type::

  echo $SHELL

If you are running a ``bash`` shell, include the lines in your ``.bashrc``
file::

     esm_home=~mdul
     export PYTHONPATH=${esm_home}/gpaw:${esm_home}/ase3k:/usr/local/gbar/lib/pythonmodules
     export GPAW_SETUP_PATH=${esm_home}/gpaw-setups:${GPAW_SETUP_PATH}
     export PATH=${esm_home}/gpaw/tools:${esm_home}/ase3k/tools:/appl/htools/vmd/bin:${PATH}
     export PYTHONSTARTUP=${HOME}/.pythonrc
     alias python="fysikpython"

If you are running a ``tcsh`` or ``csh`` shell include this line in your
``.tcshrc`` or ``.cshrc`` file::

    set esm_home=~mdul
    setenv PYTHONPATH ${esm_home}/gpaw:${esm_home}/ase3k:/usr/local/gbar/lib/pythonmodules
    setenv GPAW_SETUP_PATH ${esm_home}/gpaw-setups:${GPAW_SETUP_PATH}
    setenv PATH ${esm_home}/gpaw/tools:${esm_home}/ase3k/tools:/appl/htools/vmd/bin:${PATH}
    setenv PYTHONSTARTUP ${HOME}/.pythonrc
    alias python fysikpython


Create a python startup file
============================
Put this in your $HOME/.pythonrc file::

    import rlcompleter
    import readline
    readline.parse_and_bind("tab: complete")

This will enable tab-completion when running python interactively

Create a VMD startup file
============================
Put this in your $HOME/.vmdrc file::
  
  menu main on
  display projection orthographic
  mol delrep 0 top
  mol representation CPK 1.0 0.1 10.0 10.0
  mol addrep top

This will enable CPK representation of your atoms, and make the main window appear.
You can also show the main window manually by typing the following in the *VMD Console* ::

  menu main on


Parallel runs
=============

Follow instructions from :ref:`platforms_and_architectures`
