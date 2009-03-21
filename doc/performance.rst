===========
Performance
===========

Accuracy
========

The PAW method gives us the accuracy of all-electron calculations, if:

* the frozen core approximation is a good approximation,
* we have enough projector functions for all atoms.

We have tested this for the atomization energy of 20 molecules
[Mor05]_.  The average
and maximum differences between our PAW atomization energies and the
all-electron calculations of Kurth *et al.* [Kur99]_ are
0.05 eV and 0.15 eV respectively, and comparing with the all-electron
calculations of Zhang *et al.* [Zha98]_ we get an average difference of
0.05 eV and a maximum difference of 0.13 eV (the two sets of
all-electron calculations differ by 0.05 eV in average and 0.17 as
maximum).

We also compared calculated PAW Cohesive energies, lattice constants,
and bulk moduli with accurate all-electron energies, and got good
agreement.  The solids were: Na, Li, Si, C, SiC, AlP, MgO, NaCl, LiF, Cu
and Pt (see the paper [Mor05b]_ for more details).


The setups used for the test calculations can be found on the
:ref:`setups` page.


Efficiency
==========

Hopefully the CPU-timings for the examples in this section will
improve as we find time for optimization.


Molybdenumdisulfide slab
------------------------

Here we have calculated the energy difference between a perfect
:math:`\rm{MoS}_2` slab and one with a defect.  The following plots
show the convergence of the defect energy as a function of grid
spacing (for the gridcode calculation) and the planewave cutoff (for
the Dacapo calculation):

Defect energy [eV]:

|grid|             |dacapo|

.. |grid| image:: _static/gridperf.png
.. |dacapo| image:: _static/dacapoperf.png


The time it takes to do the calculations with the two codes, will
depend on the choice for grid spacing and planewave cutoff.  If we
choose *h* = 0.25 Å and a planewave cutoff of 420 eV, we should get
comparable accuracy.  With these choices the grid code finishes in 31
iteration after 785 seconds, and Dacapo will finish after 20
iterations (of a different kind )in 470 seconds.


Au wire between Au(100) surfaces
--------------------------------

For this system we used 136x44x44 grid points and 477 bands (924
electrons).  On two nodes, the ground state was found in 30224
seconds.  A similar ultra-soft plane-wave Dacapo calculation could be
done in 40989 seconds on one node.  Dacapo needs 25 iterations to find
the ground state, whereas the grid code uses 71 iterations (of a
different kind).  At the moment work is being done in order to reduce
the number of iterations necessary in the grid code.

.. image:: _static/goldwire.png


.. [Kur99] S. Kurth and J. P. Perdew and P. Blaha,
   Int. J. Quantum Chem. 75, 889 (1999)
.. [Zha98] Y. Zhang and W. Yang,
   Phys. Rev. Lett. 80, 890 (1998)
