.. _nanolab:

=======================
nanolab.cnf.cornell.edu
=======================

Here you find information about the the system
`<http://www.cnf.cornell.edu/cnf5_tool.taf?_function=detail&eq_id=111>`_.

The installation of user's packages on nanolab EL4 described below uses
`modules <http://modules.sourceforge.net/>`_, and assumes `bash` shell:

- packages are installed under ``~/CAMd``::

   mkdir ~/CAMd
   cd ~/CAMd

- module files are located under ``~/CAMd/modulefiles``

- download the :svn:`~doc/install/Linux/customize_nanolab_EL4_serial.py` file::

   wget https://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/Linux/customize_nanolab_EL4_serial.py

  .. literalinclude:: customize_nanolab_EL4_serial.py

- download packages with :svn:`~doc/install/Linux/download_nanolab.sh`,
  buy running ``sh download_nanolab.sh``:

  .. literalinclude:: download_nanolab.sh

- from `nanolab.cnf.cornell.edu` login to one of `c`-nodes (Red Hat 4)::

    ssh c4.cnf.cornell.edu

- install packages, deploy modules and test with :svn:`~doc/install/Linux/install_nanolab.sh`,
  buy running ``sh install_nanolab.sh``:

  .. literalinclude:: install_nanolab.sh

  **Note** that every time you wish to install a new version of a package,
  and deploy new module file, better keep the old module file.

- submit the test job::

   qsub submit.sh

  using the following :file:`submit.sh`::

   TODO
