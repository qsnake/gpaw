==============================
Initialization and I/O changes
==============================

This is a proposal for some changes that will solve various issues with 
the maintainability and stability of the I/O code amongst other things.

.. contents::

Rationale
=========

Presently the gpw I/O is handled centrally by the module 
gpaw/io/__init__.py.  If someone makes changes in setup.py or 
density.py, the I/O may break due to these "non-local correlations" (we 
in particular, being physicists, should know to appreciate locality), or 
it may misbehave in subtle ways for certain cases 
(TDDFT/LCAO/non-gamma-point/etc.).

Most of this trouble can be avoided entirely by requiring that objects 
should know how to read and write themselves.  Thus, responsibility for 
what to write (and how to read it back!) is delegated to various objects 
as per the usual 'object oriented' way.

A different but sort of related issue: The output.py module writes lots 
of things to the log file, and those things would be better off 'writing 
themselves'.  There are several bugs here waiting to be fixed: if the 
Poisson solver was created by hand with a non-standard stencil, then the 
wrong stencil is written.  Scalapack/BLACS information is sometimes 
wrong (depending on the way it was specified).  No information on the 
stridedness of the band descriptor is written (and thus parallelization 
info is incomplete).  There are probably other issues.


Object hierarki
===============

.. image:: ../bigpicture.png
   :target: ../../bigpicture.pdf

So all the objects above may implement functions to read and write their 
own parameters.  They could also implement functions to read/write 
human-readable information to log files (which is highly desirable).

On a somewhat deeper level, we could formalize the tree hierarchy 
between the major GPAW objects and define a mandatory interface for all 
major objects to implement (read/write, memory estimate, 
initialize/set_positions/allocate -- these procesure *all* involve 
traversing the same tree hierarchy).  This might make it easier for new 
programmers to learn the structure of GPAW (a well-known problem).

Example of what an object could look like:

.. literalinclude:: density.py


The PAW calculator object
=========================

The following base class (rough sketch) should propably be moved to
ASE, so that all ASE-calculators can use it:

.. literalinclude:: ase.py

It should be possible to create the PAW calculator without knowing the
atoms and also from a restart file - and both ways must be cheap.

.. literalinclude:: paw.py

Open questions
==============

The above pseudo code is not the final answer - it is an attempt to
make the further discussions more concrete.  There are several things
to think about:

* What should the ASECalculator look like?

* How should reading and writing work?

  + Should it be like pickle where the class name is written (example:
    MixerSum).  The reader will then create that object?

  + What methods should a reader have?  ``get_object('name of
    object')``, ``get_array('name of array')``, ``get_atoms()``,
    ``get_grid_descriptor()``, ...

* How much should/can ``__init__()`` for the different objects do?


Other thoughts
==============

What should the restart file format look like:

* Should we use our tar-file format and put folders in it - one for
  each object (density, hamiltonian, wave functions, ...).
* It should be easy to have several backends (pickle, directory, hdf5).
* We need backwards compatibility.
* Use ``__str__()`` and ``__repr__()``.
