.. |angst|  unicode:: U+0212B .. ANGSTROM SIGN
.. |sigma|  unicode:: U+003C3 .. GREEK SMALL LETTER SIGMA
.. |pi|     unicode:: U+003C0 .. GREEK SMALL LETTER PI


========================================
Kohn-Sham wavefunctions of a CO molecule
========================================

In this section, we will look at the Kohn-Sham wavefunctions of the CO
molecule and compare them to results from molecular orbital theory.

* Make a script, where a CO molecule is placed in the center of a qubic 
  unit cell with non-periodic boundary conditions, e.g. of 6 |angst|. For 
  more accurate calculations, the cell should definitely be bigger, 
  but for reasons of speed, we use  this cell here. Which value for the 
  grid spacing would you use? Include a couple of unoccupied bands in the 
  calculation (what is the number of valence electrons in CO). 
  Guess reasonable positions from
  the covalent radii of C and O. Then relax the CO molecule to its
  minimum energy position. Write the relaxation to a trajectory file and 
  the final results to a .gpw file. The wavefunctions 
  are not written to the .gpw file by default, but can be saved by 
  writing ``calc.write('CO.gpw', mode='all')``, where ``calc`` is 
  your calculator object. The trajectory can be viewed by::

    $ ag CO.traj

  Mark the two atoms to see the bond length.

* As this is a calculation of a molecule, one should get integer
  occupation numbers - check tis in the text output.  What electronic
  temperature was used?

* Plot the Kohn-Sham wavefunctions of the different wave functions of the CO
  molecule. The wavefunctions should be written to .cube files for handling 
  with VMD. The following lines could be included in a script or written 
  directly from the python promt::

    >>> from ase import * 
    >>> from gpaw import Calculator
    >>> calc = Calculator('CO.gpw')
    >>> CO = calc.get_atoms()
    >>> for n in range(calc.get_number_of_bands()):
    ...     wf = calc.get_pseudo_wave_function(band=n)
    ...     write('CO%d.cube' % n, CO, data=wf)
  
  What is the highest occupied state and the lowest unoccupied state?
  Which wavefunctions are bonding and which are antibonding? Identify 
  |sigma| and |pi| orbitals.
