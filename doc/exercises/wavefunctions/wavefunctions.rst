============================================================
Kohn-Sham wavefunctions of the oxygen atom and CO molecule
============================================================

In this section we will look at the Kohn-Sham wavefunctions of the O
atom and CO molecule and compare them to results from molecular orbital theory.

* The first script :svn:`~doc/exercises/wavefunctions/O.py` sets up an oxygen
  atom in a cubic supercell with non-periodic boundary conditions and calculates
  the total energy. A couple of unoccupied bands are included in the calculation:

.. literalinclude:: O.py

.. highlight:: bash


* Towards the end, a :file:`.gpw` file is written with the Kohn-Sham wavefunctions
  by `calc.write('O.gpw', mode='all')`. At the very end we write the Kohn-Sham
  wavefunctions to :file:`.cube` files for
  handling with the :program:`VMD` program.

* Run the script and check the output file. What is the occupation numbers
  for the oxygen atom free in vacuum?

* The orbitals can be visualized in :program:`VMD`. 
  Load all of the wavefunctions into :program:`VMD`
  simultaneously, by running :samp:`vmd O{?}.cube`. In :program:`VMD` choose
  :menuselection:`Graphics --> Representations`, click
  :guilabel:`Create Rep`, then choose
  :menuselection:`Drawing Method --> isosurface`.  In the
  :guilabel:`Data Set` field, you can then
  choose between all the saved wavefunctions.

  Can you identify the highest occupied state and the lowest unoccupied state?

  How does your wave functions compare to a molecular orbital picture?

  
* Make a script, where a CO molecule is placed in the center of a cubic
  unit cell with non-periodic boundary conditions, e.g. of 6 Å. For
  more accurate calculations, the cell should definitely be bigger,
  but for reasons of speed, we use this cell here. A grid spacing of 
  around 0.20 Å will suffice. Include a couple of unoccupied bands in the
  calculation (what is the number of valence electrons in CO?).
  You can quickly create the Atoms object with the CO molecule by::
  
    $ from ase.data.molecules import molecule
    $ CO = molecule('CO')
  
  This will create a CO molecule with an approximately correct bond length
  and the correct magnetic moments on each atom.

  Then relax the CO molecule to its minimum energy position. 
  Write the relaxation to a trajectory file and
  the final results to a :file:`.gpw` file. The wavefunctions
  are not written to the :file:`.gpw` file by default, but can again be saved by
  writing :samp:`{calc}.write('CO.gpw', mode='all')`, where :samp:`{calc}` is
  the calculator object. The trajectory can be viewed by::

    $ ag CO.traj

  Mark the two atoms to see the bond length.

* As this is a calculation of a molecule, one should get integer
  occupation numbers - check this in the text output.  What electronic
  temperature was used and what is the significance of this?

* Plot the Kohn-Sham wavefunctions of the different wave functions of the CO
  molecule by after writing :file:`.cube` files for handling with :program:`VMD`.

* Can you identify the highest occupied state and the lowest unoccupied state?

  How does your wavefunctions compare to a molecular orbital picture?
  Try to Identify :math:`\sigma` and :math:`\pi` orbitals. Which
  wavefunctions are bonding and which are antibonding?

.. hint::

  You might find it usefull to look at the molecular orbital diagram
  below, taken from `The Chemogenesis Web Book`_.

  .. figure:: co_bonding.jpg
     :align: center

.. _The Chemogenesis Web Book: http://www.meta-synthesis.com/webbook/39_diatomics/diatomics.html#CO
