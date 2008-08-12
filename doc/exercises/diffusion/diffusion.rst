.. _diffusion_exercise:

=========================================
Diffusion of gold atom on Al(100) surface
=========================================

In this ASE-tutorial:

* :ase:`Constraint <tutorials/constraints/diffusion.html>`

the energy barrier for diffusion of a gold atom on an Al(100) surface
is calculated using a semi-empirical EMT potential.  In this
exercise, we will try to use DFT and GPAW.

* Run the script from the ASE-tutorial above and use the graphical representation 
  to get good initial guesses for the height of the gold atom in the initial and
  transition states (hollow and bridge sites).

The PAW setups for both Al and Au are quite smooth (see
:ref:`Aluminium` and :ref:`Gold`), so we can try with a coarse
real-space grid with grid-spacing 0.25 Å.  For a quick'n'dirty
calculation we can do with just a :math:`2 \times 2` sampling of the
surface Brillouin zone.  Use these parameters for the DFT
calculations::

  calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE')

In order to speed up the calculation, use just a single frozen Al(100) layer.

* Calculate the energy of the initial and final states.  Start from
  this script: :svn:`~doc/exercises/diffusion/initial.py?format=raw`.
  Do we need to apply any constraint to the gold atom?

* What is the PBE energy barrier? (Do not repeat the ASE-tutorial with GPAW, 
  but simply relax the gold atom at the transition state and use the total energy 
  differences)

* Can both initial and transition state calculations be done with only
  one **k**-point in the irreducible part of the Brillouin zone?

* Try to repeat the EMT calculations with a single frozen Al(100) layer.



Making Python Tool Boxes
========================

A science project (like the one you are going to make), will often
contain some repeated and similar sub tasks like loops over different
kind of atoms, structures, parameters etc.  As an alternative to a
plethora of similar Python scripts, made by *copy+paste*, it is
advantageous to put the repeated code into tool boxes.

Python supports such tool boxes (in Python called modules): put any
Python code into a file :file:`stuff.py` then it may be used as a tool box
in other scripts, using the Python command: ``from stuff import
thing``, where ``thing`` can be almost anything.  When Python sees
this line, it runs the file :file:`stuff.py` (only the first time) and
makes ``thing`` available.  Lets try an example:

* In file :file:`stuff.py`, put::

    constant = 17
    def function(x):
        return x - 5

* and in file :file:`program.py`, put::

    from stuff import constant, function
    print 'result =', function(constant)

* Now run the script :file:`program.py` and watch the output.

You can think of ASE and GPAW as big collections of modules, that we
use in our scripts.



Writing an adsorption script
============================

As a non-trivial example of a Python module, try to write a function:

.. function:: aual100(site, height)

The *site* argument should be one of the strings that the
:ase:`fcc100() <ase/lattice.html#lattice.surface.fcc100>` function
accepts: ``'ontop'``, ``'hollow'`` or ``'bridge'``.  The *height*
argument is the height above the Al layer.  The function must return
the energy and write ``<site>.txt``, ``<site>.traj``, and
``<site>.gpw`` files.

* You could have used this functions to calculate the energy barrier
  above.  Use it to calculate the energy in the ontop site::

    e_ontop = aual100('ontop')

* What seems to determine the relative energetic ordering of the three sites?

* Suppose now that an Au atom diffuses from one hollow to a
  neighboring hollow site at the surface.  Assuming a prefactor of 10\
  :sup:`13`/sec, how often does the diffusion take place at *T* = 100
  K, 200 K, 300 K and 500 K.

* For biological catalytic processes, a popular rule of thumb is
  that the rate doubles for every temperature increase of 10 K around
  room temperature.  What activation energy does this correspond to?

* Look at the relaxed configurations with the :command:`ag`
  command::

    $ ag -r 3,3,2 ontop.traj

  or::

    $ ag -g 'd(0,4),F[4,2]' ontop.traj

  to plot the force in the *z*-direction on the gold atom as a
  function of the Au-Al distance.  Try also *terminal-only-mode*::
 
    $ ag -t -g 'd(0,-1),F[2,2]' ontop.traj



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
:svn:`~doc/exercises/diffusion/densitydiff.py?format=raw` to your area.

Read it and try to understand what is does. Change the necessary lines
to look at one of your slabs with Au adsorbed. When you have written the 
density difference to a :file:`.cube` file, open this file in :program:`VMD`
and use it to investigate what is happening.


Using :program:`VMD` to make input files
----------------------------------------

:program:`VMD` is very useful for setting up input files to your
calculations. Use :menuselection:`Mouse --> Move --> Atom` to move Au
to another position and save the coordinates as an :file:`xyz` file.
:file:`xyz` files can be read from your Python script like this::

  >>> atoms = read('abc.xyz')

The :file:`xyz` format does not have a unit cell, so you must set that
yourself::

  >>> atoms.set_cell((Lx,Ly,Lz), scale_atoms=False)


