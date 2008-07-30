.. |sigma|  unicode:: U+003C3 .. GREEK SMALL LETTER SIGMA
.. |pi|     unicode:: U+003C0 .. GREEK SMALL LETTER PI


Wannier Functions
=================

In order to get a feel for chemical bonds in molecules and solids, 
we can transform the Kohn-Sham orbitals 
into a set of maximally localized Wannier functions.
We have cheated a little bit and
prepared a file for bulk silicon and a Benzene molecule so that you
only have to concentrate on the wannier analysis of the molecules.

Start by running si.py_ and make sure you agree with the way the
diamond structure is set up. The resulting .gpw file is used as input
to wannier-si.py_ which transforms the Kohn-Sham orbitals to maximally
localized wannier functions and plot the atoms along with the centers
of the wannier functions.  Note that the wannier centers are treated
as "X" atoms which are plotted as small red spheres.  How many
covalent bonds do you expect in a unit cell with 8 tetravalent Silicon
atoms?

The script benzene.py_ produces a .gpw that can be used as input to
create wannier functions. Convince yourself that the chosen number of
bands matches the number of occupied orbitals in the molecule.  How
many covalent bonds do you expect in Benzene?  Look at
wannier-benzene.py_ and figure out what it does. Run it and look at
the graphical representation.  Note in particular the alternating
single/double bonds between the carbon atoms.  What happens if also
you include one or two unoccupied bands?  The script also produces two
.cube files. One contains the wavefunction of the Highest Occupied
Molecular Orbital (HOMO) and the other contains a wannier function
centered between a Carbon and a Hydrogen atom. Study these with vmd
and determine which type of orbitals they represent (|sigma| or |pi|).

Now repeat the wannier function analysis on the following molecules

* H2O : use your own files from the vibrational exercise, but make
  sure the number of bands is equal to the number of occupied orbitals.

* CO : use your own .gpw file from the wavefunction exercise. Is it a
  single, double or triple bond?

or study your own favorite molecule.

.. hint::
  
  To be able to see the Wannier centers, it might be necessary to
  decrease the atomic radii, so the spheres don't overlap.
  In ``ag`` this can be done by choosing ``View -> Settings...``, and
  then decrease the scaling factor of the covalent radii.

.. _benzene.py : wiki:SVN:examples/wannier/benzene.py
.. _wannier-benzene.py : wiki:SVN:examples/wannier/wannier-benzene.py
.. _si.py : wiki:SVN:examples/wannier/si.py
.. _wannier-si.py : wiki:SVN:examples/wannier/wannier-si.py
