.. _using_sphinx:


.. _reStructuredText: http://docutils.sf.net/rst.html
.. _Sphinx: http://sphinx.pocoo.org
.. _PDF: ../GPAW.pdf

Using Sphinx
============

.. highlight:: bash

We use the Sphinx_ tool to generate the GPAW documentation (both HTML
and PDF_).

First, you should take a look at the documentation for Sphinx_ and
reStructuredText_.

If you don't already have your own copy of the GPAW package, then get
that first::

  $ svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw
  $ cd gpaw

Then :command:`cd` to the :file:`doc` directory and build the html-pages::

  $ cd doc
  $ sphinx-build . _build

Make your changes to the ``.rst`` files, run the
:command:`sphinx-build` command again, check the results and if things
looks ok, commit::

  $ emacs index.rst
  $ sphinx-build . _build
  $ firefox _build/index.html
  $ svn ci -m "..." index.rst

To build a pdf-file, you do this::

  $ sphinx-build -b latex . _build
  $ cd _build
  $ make GPAW.pdf

More tricks to be found at ASE `Writing documentation <https://wiki.fysik.dtu.dk/ase/development/writing_documentation_ase.html>`_ page.
