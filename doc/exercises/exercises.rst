.. _exercises:

=========
Exercises
=========

XXX link to setting up unix environment (:ref:`summerschool`)

XXX link to ASE, Python tips and tricks, ...

.. hint::

   Square roots are calculated like this: ``2**0.5`` or
   ``sqrt(2)`` (the ``sqrt`` function must first be imported: ``from
   math import sqrt`` or ``from ase import *``).

.. note::

   In python, ``/`` is used for both integer- and float
   divisions. Integer division is only performed if both sides of the
   operator are integers (you can always force an integer division by
   using ``//``)::

     >>> 1 / 3
     0
     >>> 1 / 3.0
     0.33333333333333331

XXX add more links from the exercises to the corresponding ASE-documentation.

.. toctree::
   :maxdepth: 1

   aluminium/aluminium
   diffusion/diffusion
   surface/surface
   vibrations/vibrations
   wavefunctions/wavefunctions
   wannier/wannier
   iron/iron
   dos/dos
   band_structure/bands
   stm/stm
   neb/neb
   tst/tst
   transport/transport
   lrtddft/lrtddft
   
See also :ref:`timepropagation`.

These exercises are used in the course Electronic structure methods in
materials physics, chemistry and biology. For comments and questions,
write to the :ref:`gpaw-devel-ml`.
