.. _all_electron_density:

================================
Getting the all-electron density
================================

The variational quantity of the PAW formalism is the pseudo-density ñ. This is also the density returned by the ``GetDensityArray`` method of the GPAW calculator. Sometimes it is desirable to work with the true all-electron density.
The PAW formalism offers a recipe for reconstructing the all-electron density from the pseudo-density, and in GPAW, this can be reached by the method ``get_all_electron_density`` of the ``Calculator`` class.

This tutorial describes how to get and use the all-electron density.

The all electron density is reached by calling the ``get_all_electron_density`` method in the same way as you would normally use the ``GetDensityArray`` method, i.e.

>>> from gpaw.utilities.singleatom import SingleAtom
>>> Si = SingleAtom('Si', a=5.5, h=.2)
>>> calc = Si.atom.get_calculator()
>>> e  = Si.energy()
>>> nt = calc.get_pseudo_valence_density()
>>> n  = calc.get_all_electron_density()

would give you the pseudo-density in ``nt`` and the all-electron density in ``n``.

As the all-electron density has more structure than the pseudo-density, it is necessary to refine the density grid used to represent the pseudo-density. This can be done using the ``gridrefinement`` keyword of the ``get_all_electron_density`` method

>>> n = calc.get_all_electron_density(gridrefinement=2)

The plot below shows a line-section of the Si density using a grid refinement factor of 1, 2, and 4 respectively

.. image:: gridrefinement.png

The all-electron density will always integrate to the total number of electrons of the considered system (independent of the grid resolution), while the pseudo density will integrate to some more or less arbitrary number. This fact is illustrated in the following example.

---------------
Example 1: NaCl
---------------

As an example of application, consider the three systems Na, Cl, and NaCl. The pseudo- and all-electron densities of these three systems are shown on the graph below.  

.. image:: ae_density_NaCl.png

The results of this example have been made with the script NaCl.py:

.. literalinclude:: NaCl.py

The pseudo- and all-electron densities of the three systems integrate to:

==== ==== =====
\    ñ    n
Na   1.60 11.00
Cl   7.50 17.00
NaCl 9.07 28.00
==== ==== =====

From which we see that the all-electron densities integrate to the total number of electrons in the system, as expected.

-------------------------------------------
Example 2: Bader analysis of H\ :sub:`2`\ O
-------------------------------------------

To do a bader anaysis of the all-electron density, you should first save the density to a .cube file:

.. literalinclude: H2O-bader.py

Then run the bader program::

  $ bader H2O.cube 2

The bader program will then produce a range of files, the most usefull of which I think is:

* ``ACF.dat``, which contains a list of the total charge of each bader volume (in the column *BADER*), and
* ``bader_rho.dat``, which is a grid of the same size as the density, and with each value indicating the bader volume to which the respective density coordinate belongs.

In the case of the water molecule above, the partial charges of the bader volumes is: 9.13, 0.42, and 0.45, indicating a charge transfer of approximately 0.55 electrons from each hydrogen atom to the oxygen atom.

To access the ``bader_rho.dat`` file in python, you can do the following:

>>> import numpy as npy
>>> f = open('bader_rho.dat', 'r')
>>> d = []
>>> for row in f.readlines():
>>>     for col in row.split():
>>>         d.append(eval(col))
>>> f.close()
>>> bader = npy.array(d)
>>> bader.shape = gridrefinement * calc.get_number_of_grid_points()[::-1]
>>> bader = npy.transpose(bader)

This will put the array ``bader`` in the same format as the density arrays.

The plot below shows a crossection of the ae-density and the bader partitions.

.. image:: ae_density_H2O.png

The plots have been made using the script H2O-plot.py:

.. literalinclude:: H2O-plot.py

For more information on the Bader method, see the tutorial_ on the ASE wiki.

.. _tutorial: wiki:ASE:Analysis#bader-analysis
