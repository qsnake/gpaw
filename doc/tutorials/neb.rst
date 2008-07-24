.. _neb:

=========================================
NEB calculations parallelized over images
=========================================

The `Gold atom diffusion on Al(100)`_ example can be used with GPAW like this:

par_

.. _par: inline:neb.py

If we run the job on 12 cpu's::

  $ mpirun -np 12 gpaw-python neb.py

then each of the three internal images will be parallelized over 4 cpu's.
The energy barrier is found to be 0.29 eV.


.. _Gold atom diffusion on Al(100): http://web2.fysik.dtu.dk/ase/tutorials/neb/diffusion.html
