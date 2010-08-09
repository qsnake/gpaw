.. _summerschool10:

=======================
CAMd Summer school 2010
=======================

Databar
=======

Setting up your UNIX environment
--------------------------------

The first time you use the databar computers, you must do this:

.. highlight:: bash

::

  $ ~jjmo/summerschool/setup.sh
  $ source ~/.bashrc

That will set up the environment for you so that you can use ASE, GPAW, VMD and matlpotlib.

**Warning** runnig :command:`~jjmo/summerschool/setup.sh` owervrites
users ~/.bashrc ~/.emacs ~/.pythonrc ~/.vmdrc and ~/.matplotlib directory.

Notes
-----

* Useful links: Userguides_ FAQ_ Unix_ USB-sticks_ (is it still valid???)

* Editors: emacs, vim, nedit (MS Windows/Macintosh-like environment) support python syntax highlighting (is it still valid???)

* Printer: gps1-308. Terminal: lp -d gps1-308 filename

* E-mail client:
  Thunderbird is the default mail client in the databar and configured  
  with your summer school e-mail (camd0??@student.dtu.dk) (is it still valid???)

* To open a pdf-file: acroread filename

Niflheim
========

Frontend nodes
--------------

Log in to niflheim::

  ssh thul.fysik.dtu.dk

Same password as handed out for the databar.

Copying files from gbar to niflheim
-----------------------------------

You can copy files from the Gbar to niflheim with ``scp``. If you are on 
niflheim::

    scp glint00.gbar.dtu.dk:path/filename .

will copy ``filename`` to your present location. Likewise::

    scp thul.fysik.dtu.dk:path/filename .

will copy ``filename`` from Niflheim to your present location at the Gbar.

GPAW
----

Use the :command:`gpaw-qsub` command to submit GPAW jobs to the queue.

.. _Userguides: http://www.gbar.dtu.dk/index.php/Category:User_Guides
.. _FAQ: http://www.gbar.dtu.dk/index.php/General_use_FAQ
.. _Unix: http://www.gbar.dtu.dk/index.php/UNIX
.. _USB-sticks: http://www.gbar.dtu.dk/index.php/USBsticks
.. _Octopus: http://www.tddft.org/programs/octopus/wiki/index.php/
.. _tutorial: http://www.tddft.org/programs/octopus/wiki/index.php/Tutorial

