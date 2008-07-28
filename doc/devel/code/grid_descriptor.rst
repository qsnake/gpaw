.. _grid_descriptor:

===============
Grid descriptor
===============

A grid-descriptor is used to create grids for functions like wave functions and electron densities (3D arrays of values at grid points).

Attributes:
 ========== ========================================================
 ``domain`` Domain_ object.
 ``dv``     Volume per grid point.
 ``h_i``    Array of the grid spacing along the three axes.
 ``n_i``    Array of the number of grid points along the three axes.
 ========== ========================================================

Parallel stuff: ``comm``, ``N_i``, ``beg0_i``, ``beg_i`` and ``end_i``.

.. |times|  unicode:: U+000D7 .. MULTIPLICATION SIGN

This is how a 2 |times| 2 |times| 2 3D array is layed out in memory::

    3-----7
    |\    |\
    | \   | \
    |  1-----5      z
    2--|--6  |   y  |
     \ |   \ |    \ |
      \|    \|     \|
       0-----4      +-----x

Example::

  >>> a = num.zeros((2, 2, 2))
  >>> a.flat[:] = range(8)
  >>> a
  array([[[0, 1],
          [2, 3]],
         [[4, 5],
          [6, 7]]])
    
