.. _optimizer_tests:

===============
Optimizer tests
===============
This page shows benchmarks of optimizations done with our different optimizers.
Note that the iteration number (steps) is not the same as the number of force
evaluations. This is because some of the optimizers uses internal line searches
or similar.

N2Cu
====
Relaxation of Cu surface.

Calculator used: EMT

.. csv-table::
   :file: N2Cu-surf.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

N2 adsorption on relaxed Ru surface

Calculator used: EMT

.. csv-table::
   :file: N2Cu-N2.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

Cu_bulk
=======
Bulk relaxation of Cu where atoms has been rattled.

Calculator used: EMT

.. csv-table::
   :file: Cu_bulk.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

CO_Au111
========
CO adsorption on Au

Calculator used: EMT

.. csv-table::
   :file: CO_Au111.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

H2
==
Geometry optimization of gas-phase molecule.

Calculator used: EMT

.. csv-table::
   :file: H2-emt.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

Calculator used: GPAW

.. csv-table::
   :file: H2-gpaw.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

C5H12
=====
Geometry optimization of gas-phase molecule.

Calculator used: GPAW (lcao)

.. csv-table::
   :file: C5H12-gpaw.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

nanoparticle
============
Adsorption of a NH on a Pd nanoparticle.

Calculator used: GPAW (lcao)

.. csv-table::
   :file: nanoparticle.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

NEB
=======
Diffusion of gold atom on Al(100) surface.

Calculator used: EMT

.. csv-table::
   :file: neb-emt.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note

Calculator used: GPAW (lcao)

.. csv-table::
   :file: neb-gpaw.csv       
   :header: Optimizer, Steps, Force evaluations, Energy, Note
