.. _download:

========
Download
========

Latest stable release
=====================

The latest stable release can be obtained from ``svn`` or as a ``tarball``.

When using svn please set the following variable:

- bash::

   export GPAW_TAGS=https://svn.fysik.dtu.dk/projects/gpaw/tags/

- csh/tcsh::

   setenv GPAW_TAGS https://svn.fysik.dtu.dk/projects/gpaw/tags/

======= =========== ================================== ===================== ===========
Release Date        Retrieve as svn checkout           Retrieve as tarball   ASE release
======= =========== ================================== ===================== ===========
    0.4 Nov 16 2008 ``svn co $GPAW_TAGS/0.4 gpaw-0.4`` gpaw-0.4.2734.tar.gz_ ase-3.0.0_
======= =========== ================================== ===================== ===========

.. _gpaw-0.4.2734.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.4.2734.tar.gz

.. _ase-3.0.0:
    https://svn.fysik.dtu.dk/projects/ase/tags/3.0.0

.. note::

   GPAW should be retrieved together with compatible ASE release
   (see :ase:`Download and install ASE <download.html>`).

Latest development release
==========================

The latest revision can be obtained like this::

  $ svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw

or from the daily snapshot: `<gpaw-snapshot.tar.gz>`_.

See :ref:`faq` in case of problems.

Installation
============

After downloading create the link to the requested version, e.g.:

- if retrieved from ``svn``::

   $ cd $HOME
   $ ln -s gpaw-0.4 gpaw

- if retrieved as ``tarball``::

   $ cd $HOME
   $ tar xtzf gpaw-0.4.2734.tar.gz
   $ ln -s gpaw-0.4.2734 gpaw

When you have the code, go to the :ref:`installationguide`.
