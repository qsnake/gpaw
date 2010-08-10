.. _summerschool10:

=======================
CAMd Summer school 2010
=======================

Setting up your UNIX environment
--------------------------------

The first time you use the databar computers, you must configure your
environment.  Open the ``.bashrc`` file in your favourite editor:

.. highlight:: bash

::

  $ emacs ~/.bashrc

Scroll down and append this line at the end of the file::

  source ~ashj/summerschool/gbar-gpaw.rc

Run this command to apply the changes:

.. highlight:: bash

::

  $ source ~/.bashrc

That will set up the environment for you so that you can use ASE,
GPAW, VMD and matplotlib.

Running GPAW calculations
-------------------------

GPAW calculations are written as Python scripts, which can be run with
the command::

  $ python filename.py

If the calculation lasts more than a few seconds, submit it to the
queue instead of running it directly::

  $ gpaw-qsub 1 filename.py

This will allow the script to be executed on a different host, so the
jobs will be distributed efficiently even if many users logged on to
the same computer.  You can run jobs in parallel, using more CPUs for
increased speed, by specifying e.g. 4 CPUs like this::

  $ gpaw-qsub 4 filename.py

The ``qstat`` or :samp:`qstat -u {USERNAME}` commands can be used to
monitor running jobs, and :samp:`qdel {JOB_ID}` to delete jobs if
necessary.


Notes
-----

* Useful links: Userguides_ FAQ_ Unix_ USB-sticks_ (is it still valid???)

* Editors: emacs, vim, nedit (MS Windows/Macintosh-like environment) support python syntax highlighting (is it still valid???)

* Printer: gps1-308. Terminal: :samp:`lp -d gps1-308 {filename}`

* E-mail client:
  Thunderbird is the default mail client in the databar and configured  
  with your summer school e-mail (camd0??@student.dtu.dk) (is it still valid???)

* To open a pdf-file: acroread filename
