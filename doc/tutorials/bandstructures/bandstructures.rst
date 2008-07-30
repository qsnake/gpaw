.. _bandstructures:

=========================================
Calculation of electronic band structures
=========================================

The script bandstructure.py_ will calculate the band structure of Na along the Gamma-X direction.
The resulting band structure is shown below.

.. figure:: ../../_static/sodium_bands.png

(This plot was made using plot_bands.py_)

.. _bandstructure.py: wiki:SVN:tutorials/bandstructures/bandstructure.py
.. _plot_bands.py: wiki:SVN:tutorials/bandstructures/plot_bands.py

One should note that as GPAW only works with orthorhombic cell, the unit cell here thus larger than
the primitive BCC cell. Accordingly, the Brillouin zone is smaller, and bands are folded back from
the larger primitive Brillouin zone. For a description of the symmetry labels of the Brillouin zone;
see the figure below.

.. figure:: ../../_static/bz-all.png
   :width: 600 px
   :align: left
