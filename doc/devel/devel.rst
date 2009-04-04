.. _devel:

===========
Development
===========

GPAW development can be done by anyone!

Just take a look at the
:ref:`todo` list and find something that suits your talents!

Read about :ref:`Theory <features_and_algorithms>` and
:ref:`technology` behind the code. Follow with :ref:`introduction_to_paw`
and :ref:`code_overview`, and print :ref:`the_big_picture`.

As a developer, you should subscribe to all GPAW related :ref:`mailing_lists`.

Now you are ready to to perfom a :ref:`developer_installation`
and start development!

.. toctree::
   :maxdepth: 1

   developer_installation

.. note --- below toctrees are defined in separate files to make sure that the line spacing doesn't get very large (which is of course a bad hack)

Development topics
==================

When committing significant changes to the code, remember to add a
note in the :ref:`releasenotes` at the top (current svn) - the version
to become the next release.

.. toctree::
   :maxdepth: 1

   toc-general

* A report_ from PyLint.

.. _report: http://wiki.fysik.dtu.dk/stuff/pylint_global.html

Specialized information
=======================

.. toctree::
   :maxdepth: 1

   toc-special

* Supported :ref:`platforms_and_architectures`.

.. _code_overview:

Code Overview
=============

The developer guide provides an overview of the PAW quantities and how
the corresponding objects are defined in the code:

.. toctree::
   :maxdepth: 2

   overview
   developersguide
   paw
   wavefunctions
   setups
   density_and_hamiltonian
   others

.. _the_big_picture:

The big picture
===============

Keep this under your pillow (work in progress):

.. image:: overview.png
   :target: overview.pdf_
   :width: 550 px

Created with OpenOffice.org-Draw_: overview.odg_.


.. _OpenOffice.org-Draw: http://www.openoffice.org/product/draw.html
.. _overview.pdf: ../_static/overview.pdf
.. _overview.odg: ../_static/overview.odg

.. image:: bslogo.png
   :target: http://developer.berlios.de/projects/gridpaw

The source for the GPAW-logo can be found here: gpaw-logo.odg_

.. _gpaw-logo.odg: ../_static/gpaw-logo.odg


Old pages
---------

These pages may contain inaccurate or outdated information

.. toctree::
   :maxdepth: 1

   bugs
   testsuite
   statistics
