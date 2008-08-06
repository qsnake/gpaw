================================
Aluminium with adsorbed hydrogen
================================

One of the most prominent applications of density functional theory is
making predictions about alternative atomic structures, when
adsorbates are present on the surface.  In this exercise, we study the
adsorption of H onto Al(100). For simplicity, we use a small and thin
Al(100) slab: A 1 x 1 surface unit cell of a two layer Al(100) surface and
deposit a monolayer of H onto the Al(100) slab.  The thermodynamics of
the H/Al(100) system is governed by the adsorption potential energy
surface (PES), i.e. the energy of H/Al(100) as function of the H atom
coordinates; in this section, we try to map out and understand the
adsorption PES.

* Why are high symmetric sites on the Al(100) surface good
  candidates for stable adsorption sites?

* How many inequivalent high symmetry adsorption sites are present on
  a Al(100) surface.  What is the coordination number 
  *Z* (i.e. number of nearest Aluminum neighbors of an H atom) in each
  adsorption site?

* Find the high symmetry adsorption sites on a Al(111) surface. There
  are two inequivalent hollow sites. What is the difference?

* We'll adsorb a monolayer of H in all high symmetry sites of the
  Al(100), i.e. one H atom per Al(100) 1 x 1 surface unit cell.

  - Use a unit cell with a height of 10.0 Å.

  - We relax ionic positions.  For simplicity, we'll freeze the
    Aluminum slab atoms, and only relax the H adsorbate. This is done
    in the script :svn:`~doc/exercises/adsorbate/relax.py?format=raw`.

  - If you want the relaxation to converge quickly, it is necessary to
    make a qualified guess on the equilibrium position of the H
    adsorbate.  A simple way is to assume that equilibrium bond
    lengths are the sum of the respective covalent radii. In our case *r*\
    :sub:`Al` = 1.18 Å and *r*\ :sub:`H` = 0.37 Å.  You may get these
    kinds of data at http://www.webelements.com.  Select the initial
    adsorbate elevation over the surface for each adsorption site, so
    that Al-H bond lengths are 1.55 Å.

* Run your script and calculate the total energies of the
  different adsorption sites, thereby determining their relative
  energetic stability.

* Suppose now that an H atom diffuses from one hollow to a neighboring
  hollow site at the surface. What is the activation energy
  for this process? Assuming a prefactor of 10\ :sup:`13`/sec, how many
  times does the diffusion take place at *T* = 100 K, 200 K, 300 K and
  500 K.

* For biological catalytic processes, a popular rule of thumb is
  that the rate doubles for every temperature increase of 10 K around
  room temperature.  What activation energy does this correspond to?

* If one would want to investigate the diffusion process properly, how would
  you do this? What would have to be changed from the present setup?


* Look at the relaxed configurations with the :command:`ag`
  command::

    $ ag -r 3,3,1 ontop.traj

  or::

    $ ag -g 'd(1,2),F[2,2]' ontop.traj

  to plot the force in the *z*-direction on the H atom as a function of the Al-H
  distance.  Try also *terminal-only-mode*::
 
    $ ag -t -g 'd(1,2),F[2,2]' ontop.traj



Making nice plots with :program:`VMD`
=====================================

One functionality in ASE is that you can make nice plots of the atomic
configurations, the Kohn-Sham wave functions and the electron
density. Apart from that these plots can be made to look very nice,
they can also visualize things which otherwise are hard to analyze or
explain. ASE supports visualization tools like :program:`gOpenMol`,
:program:`Rasmol` and :program:`VMD`. We will focus on :program:`VMD`.



Plotting the atoms
------------------


:program:`VMD` uses :file:`.cube` files as input. The construction of
these can be integrated in a basic script or written afterwards from a
:file:`.gpw` file. In the example above one can use

  >>> from ase import * 
  >>> from gpaw import *
  >>> atoms, calc = restart('ontop.gpw')
  >>> n = calc.get_pseudo_density()
  >>> write('relax.cube', atoms, data=n)

The resulting :file:`relax.cube` file contains the atoms and density and is
opened in :program:`VMD` by ``vmd relax.cube``.

Three windows pop up, an OpenGL display where the atoms are visible, a
vmd console, and :program:`VMD` main. The :program:`VMD` main window
have different menues, open the :menuselection:`Graphics -->
Representations` menu and change the drawing method to CPK.
:program:`VMD` can do many things but you should try to use the Render
option to make a ray tracing figure of your slab, change the colors of
the atoms using different representations, remove the axis indicator
and change the background color. Now add a representation that shows a
density isosurface (it is best visualized with mesh or solid
surface). When you have made a povray plot you can use your favorite
graphics program (:program:`gimp` is a good one), to edit your plot
and save it as an :file:`.eps` file, which you can include in latex.



Using :program:`VMD` to plot density differences
------------------------------------------------

It is sometimes useful to look at density changes when studying for
instance adsorption reactions. Copy the script
:svn:`~doc/exercises/adsorbate/densitydiff.py?format=raw` to your area.

Read it and try to understand what is does. Change the necessary lines
to look at one of your slabs with H adsorbed. There is one major
assumption in the script if this is used for the H adsorbed on a metal
surface, try to identify it. When you have written the density
difference to a :file:`.cube` file, open this file in :program:`VMD`
and use it to investigate what is happening.


Using :program:`VMD` to make input files
----------------------------------------

:program:`VMD` is very useful for setting up input files to your
calculations. Use :menuselection:`Mouse --> Move --> Atom` to move H
to another position and save the coordinates as an :file:`xyz` file.
:file:`xyz` files can be read from your Python script like this::

  >>> atoms = read('abc.xyz')

The :file:`xyz` format does not have a unit cell, so you must set that
yourself::

  >>> atoms.set_cell((Lx,Ly,Lz), scale_atoms=False)
