.. _summerschool:

=======================
CAMd Summer school 2008
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

* Useful links: Userguides_ FAQ_ Unix_ USB-sticks_

* Editors: emacs, vim, nedit (MS Windows/Macintosh-like environment). Python syntax

* Printer: gps1-308. Terminal: lp -d gps1-308 filename

* E-mail client:
  Thunderbird is the default mail client in the databar and configured  
  with your summer school e-mail (camd0??@student.dtu.dk).

* To open a pdf-file: acroread filename

Niflheim
========

Frontend nodes
--------------

Log in to niflheim::

  ssh school1.fysik.dtu.dk

or::

  ssh school2.fysik.dtu.dk.

Same password as handed out for the databar. Please use school1 if the
number in your userid is odd and school2 if it is even.

Copying files from gbar to niflheim
-----------------------------------

You can copy files from the Gbar to niflheim with ``scp``. If you are on 
niflheim::

    scp hald.gbar.dtu.dk:path/filename .

will copy ``filename`` to your present location. Likewise::

    scp school1.fysik.dtu.dk:path/filename .

will copy ``filename`` from Niflheim to your present location at the Gbar.

GPAW
----

Use the :command:`gpaw-qsub.py` command to submit GPAW jobs to the queue.


SIESTA
------

Siesta is installed on Niflheim, so you need to log in to the Niflheim
front-end nodes as described above in the Niflheim section.
Furthermore you have to set two environment variables by adding the
following two lines to your ~/.bashrc file::

  export SIESTA_PP_PATH=~mvanin/asesiesta
  export SIESTA_SCRIPT=~mvanin/asesiesta/run_siesta.py  

and source it by typing::

  $ source ~/.bashrc

To submit a job to Niflheim, use the ``qsub`` command::

  $ qsub -l nodes=1:ppn=1:switch5 filename.py


Octopus
-------

Octopus_ is installed on the 'q' opteron nodes on Niflheim. The way to
run jobs is the following: Create inp file in the working directory as
described in the tutorial_, and then run
:svn:`~doc/run.py?format=raw`. To use various octopus utilities such
as ``oct-cross-section`` and ``oct-broad`` you need to do::

  source /usr/local/openmpi-1.2.5-pathf90/bin/mpivars-1.2.5.sh

first. Submitting jobs to the queue is done by::

  qsub -l nodes=2:ppn=4:switch5 run.py


.. _Userguides: http://www.gbar.dtu.dk/index.php/Category:User_Guides
.. _FAQ: http://www.gbar.dtu.dk/index.php/General_use_FAQ
.. _Unix: http://www.gbar.dtu.dk/index.php/UNIX
.. _USB-sticks: http://www.gbar.dtu.dk/index.php/USBsticks
.. _Octopus: http://www.tddft.org/programs/octopus/wiki/index.php/
.. _tutorial: http://www.tddft.org/programs/octopus/wiki/index.php/Tutorial

i386 RPM based systems (fc8, fc9, el4, el5)
===========================================

System wide installation
------------------------

The steps described below require root access and assume bash shell:

 - install external packages required by gpaw and ase::

    cd
    REPO="https://wiki.fysik.dtu.dk/stuff/school08/RPMS"
    yum -y install wget
    wget --no-check-certificate https://svn.fysik.dtu.dk/projects/rpmbuild/trunk/SOURCES/RPM-GPG-KEY-fys
    rpm --import RPM-GPG-KEY-fys

   - el4 (CentOS 4)::

      wget http://packages.sw.be/rpmforge-release/rpmforge-release-0.3.6-1.el4.rf.i386.rpm
      rpm -ivh rpmforge-release-0.3.6-1.el4.rf.i386.rpm
      wget ftp://ftp.scientificlinux.org/linux/scientific/4x/i386/SL/RPMS/numpy-1.0.4-1.i386.rpm
      wget --no-check-certificate https://www.scientificlinux.org/documentation/gpg/RPM-GPG-KEY-dawson
      rpm --import RPM-GPG-KEY-dawson
      yum -y localinstall numpy-1.0.4-1.i386.rpm
      wget --no-check-certificate $REPO/i386/python-matplotlib-0.91.2-3.el4.fys.i386.rpm
      wget --no-check-certificate $REPO/i386/pytz-2006p-1.el4.fys.i386.rpm
      yum -y localinstall python-matplotlib-0.91.2-3.el4.fys.i386.rpm pytz-2006p-1.el4.fys.i386.rpm

   - el5 (CentOS 5)::

      wget http://packages.sw.be/rpmforge-release/rpmforge-release-0.3.6-1.el5.rf.i386.rpm
      yum -ivh rpmforge-release-0.3.6-1.el5.rf.i386.rpm
      rpm --import http://download.fedora.redhat.com/pub/epel/RPM-GPG-KEY-EPEL
      yum -y install blas-devel lapack-devel
      wget http://download.fedora.redhat.com/pub/epel/5/i386/numpy-1.0.4-1.el5.i386.rpm
      yum -y localinstall numpy-1.0.4-1.el5.i386.rpm
      yum -y update numpy
      wget http://download.fedora.redhat.com/pub/epel/5/i386/python-matplotlib-0.90.1-1.el5.i386.rpm
      wget http://download.fedora.redhat.com/pub/epel/5/i386/pytz-2006p-1.el5.noarch.rpm
      yum -y localinstall python-matplotlib-0.90.1-1.el5.i386.rpm pytz-2006p-1.el5.noarch.rpm
      yum -y update python-matplotlib

   - fc8 and fc9 (Fedora Core 8/9)::

      yum -y install blas-devel lapack-devel
      yum -y install numpy
      yum -y install python-matplotlib

 - install gpaw and ase (**Note**! replace xxx with one of fc8, fc9, el4, el5)::

    yum -y remove campos-gpaw-setups campos-gpaw campos-ase3
    wget --no-check-certificate $REPO/i386/campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    wget --no-check-certificate $REPO/i386/campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm
    wget --no-check-certificate $REPO/noarch/campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm
    yum -y localinstall campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    yum -y localinstall campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm

