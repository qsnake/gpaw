.. _summerschool:

=======================
CAMd Summer school 2008
=======================

...

Setting up your UNIX environment
================================

This page describes how to make GPAW, ASE, PYTHON and VMD run on your
account at ``bohr.gbar.dtu.dk``.

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

This will enable CPK representation of your atoms, and make the main
window appear.  You can also show the main window manually by typing
the following in the *VMD Console* ::

  menu main on


Parallel runs
=============

Follow instructions from :ref:`bohr_gbar_dtu_dk`

Notes XXX
==========

*   Useful links: Userguides_ FAQ_ Unix_ USB-sticks_

*   Octopus_ tutorial_

*   Editors: emacs, vim, nedit (MS Windows/Macintosh-like environment). Python syntax

*   Printer: gps1-308. Terminal: lp -d gps1-308 filename

*   E-mail client:
    Thunderbird is the default mail client in the databar and configured  
    with your summer school e-mail (camd0??@student.dtu.dk).

*   To open a pdf-file: acroread filename

*   Log in to niflheim: ssh school1.fysik.dtu.dk or ssh school2.fysik.dtu.dk.
    Same password as handed out for the databar. Please use school1 if the number in your 
    userid is odd and school2 if it is even.

*   gpaw-qsub...

*   How to copy from gbar to niflheim:
    scp hald.gbar.dtu.dk:path/filename .
    scp school1.fysik.dtu.dk:path/filename .

.. _Userguides: http://www.gbar.dtu.dk/index.php/Category:User_Guides
.. _FAQ: http://www.gbar.dtu.dk/index.php/General_use_FAQ
.. _Unix: http://www.gbar.dtu.dk/index.php/UNIX
.. _USB-sticks: http://www.gbar.dtu.dk/index.php/USBsticks
.. _Octopus: http://www.tddft.org/programs/octopus/wiki/index.php/
.. _tutorial: http://www.tddft.org/programs/octopus/wiki/index.php/Tutorial
