.. _faeq:

===================================
Frequently asked exercise questions
===================================

Python
======

Square root
-----------

Square roots are calculated like this: ``2**0.5`` or ``sqrt(2)`` (the
``sqrt`` function must first be imported: ``from math import sqrt`` or
``from ase import *``).


Integer division
----------------

In python, ``/`` is used for both integer- and float
divisions. Integer division is only performed if both sides of the
operator are integers (you can always force an integer division by
using ``//``)::

  >>> 1 / 3
  0
  >>> 1 / 3.0
  0.33333333333333331
