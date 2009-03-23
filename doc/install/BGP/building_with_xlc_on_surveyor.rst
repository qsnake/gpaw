.. _building_with_xlc_on_surveyor:

==========================
Building with xlc compiler
==========================

NumPy
=======

We currently do not know how to build NumPy with xlc on BG/P.

GPAW
====

Proceed as in the gcc case, but use `bg_compiler.py <https://svn.fysik.dtu.dk/projects/gpaw/trunk/bg_compiler.py>`_
instead of the ``bg_gcc.py``; change the lines in ``customize.py`` accordingly::

  mpicompiler = "bg_gcc.py"
  mpilinker = "bg_gcc.py"
  compiler = "bg_gcc.py"

Everything else should be the same.

