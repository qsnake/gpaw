.. _building_with_xlc_on_surveyor:

==========================
Building with xlc compiler
==========================

NumPy
=======

We currently do not know how to build NumPy with xlc on BG/P.

GPAW
====

Proceed as in the :ref:`building_with_gcc_on_surveyor`,
but use the following :svn:`~doc/install/BGP/bgp_xlc.py` file:

.. literalinclude:: bgp_xlc.py

Finally, change the lines in :svn:`~doc/install/BGP/customize_surveyor_gcc.py` accordingly::

  mpicompiler = "bgp_xlc.py"
  mpilinker = "bgp_xlc.py"
  compiler = "bgp_xlc.py"

Everything else should be the same.

