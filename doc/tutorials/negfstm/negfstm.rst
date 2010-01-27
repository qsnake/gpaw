.. _negfstm:
.. default-role:: math

=======================================
STM-simulations using Green's functions
=======================================

This tutorial describes the simulation of STM-images using non equilibrium 
Green's functions methods to calculate the tunneling current.
The method is based on J. Bardeen's theory\ [#Bardeen]_, where
the Bardeen equation is expressed in terms of Green's functions.

Method
------

The method is based on the assumption that the tip and the surface region
can be treated as independent subsystems which both consist of a
non-periodic part coupled to a periodic and semi-infinite bulk crystal.

With the choice of a localized basis,
the Hamiltonian matrix of the combined tunneling junction can be 
decomposed into 
three terms:

.. math::

    H = H_S + H_T + V .

Here `H_S` and `H_T` describe the isolated surface region and tip
region, respectively, and `V` is the coupling in between. For a
particular basis function `|\phi_k\rangle` in the tip region and basis
function `|\psi_n\rangle` in the surface region the matrix elements
are calculated according to:

.. math::

    V_{kn} = \langle \phi_k | -\frac{1}{2}\nabla^2 + V^T_{KS} +
             V^S_{KS} |\psi_n \rangle,

where `V^\alpha_{KS}` is the Kohn-Sham potential of the isolated
region `\alpha`.  For large tip to surface distances and low
temperatures the tunneling current is calculated as

.. math::

  I = \frac{2e^2}{h}\int_{\varepsilon_f}^{\varepsilon_f+eU} d\omega
      \textrm{Tr}\big[V_{ST}A_{TT}(\omega-eU)V_{TS}A_{SS}(\omega)\big],

where the integration is over the bias window of width `eU` and

.. math::

    A_{\alpha\alpha} = i\big[G^a_{\alpha\alpha}-G^r_{\alpha\alpha}\big],

is the spectral function for the isolated region `\alpha` expressed in
terms of the retarded Green's function `G^r` and the advanced Green's
function `G^a`.  The retarded Green's function as a function of
energy, `\omega`, is given by

.. math::

    G_{\alpha\alpha}^r(\omega) = \big[(\omega + i\eta)S_\alpha - H_\alpha - \Sigma_\alpha \big].

Here `\eta` is a positive infinitesimal, `S_\alpha` and `H_\alpha`
denote the overlap matrix and the Hamiltonian matrix, respectively, of
the nonperiodic part of region `\alpha` and `\Sigma_\alpha` is the
self-energy that takes into account the presence of the semi-infinite
bulk crystal.

The basis set that is used to describe the electronic structure is a
set of strictly localized atomic orbitals.  For a discussion of
problems associated with this particular choice see\ [#Lorente]_.

Setting up surface and tip
--------------------------

Primarily, the Hamiltonian matrices for the tip region and the surface
region have to be calculated.  For the calculation of the selfenergy
the Hamiltonian of one principle layer and the coupling between
neighbouring layers suffices.

The following script will perform the necessary calculations for a
simple Al(100) surface with a hydrogen atom adsorbed in the on-top
position.  The tip will be simulated by a finite chain of hydrogen
atoms.

Since the surface and tip slabs interact with vacuum on both sides, a
couple of convergence layers have to be included in the calculations
to assure a smooth matching of the potential at the surface-bulk
interface.  These convergence layers have to be 'cut' away in the
subsequent process.

.. literalinclude:: dumphs.py


STM simulation
--------------

The following script shows the calculation of the STM-image at constant height

.. literalinclude:: scan.py

The result should look like this:

.. image:: fullscan.png
.. image:: linescan.png

The calculation can also be performed in parallel, however with the
restriction that the number of processors should not exceed the number
of energy points on the energy grid.

Constant current images
-----------------------

To calculate a constant current image, the current `I(x, y, d)` is mapped 
as a function of both `x` and `y` position and the minimal tip height `d`. 
The constant current image associated with a particular current value 
is then calculated by interpolating between constant height images.
To map the current, the function :meth:`scan3d()` can be used in
e.g. the following way::

  stm.scan3d(dmin=3.0, dmax=7.0, filename='scan3d')

This calculates all current values for tip heights between `3\text{Å}`
and `7\text{Å}` and dumps the result to a pickle file in the local
directory.  Most efficiently, the calculation can be done in parallel,
where the number of processors should not exceed the number of energy
points in the energy grid.

Subsequently the simulated images can be analyzed and plotted in the
following way:

>>> from gpaw.transport.jstm import *
>>> stm = STM()
>>> stm.read_scans_from_file('scan3d')
>>> stm.get_constant_current_image(0.001)
>>> stm.plot()

Also different constant height scans can be plottet using:

>>> index=0
>>> stm.get_constant_height_image(index)
>>> stm.plot()

this returns the constant height scan with the smallest tip height.
Performing a line scan is also possible: 

>>> stm.linescan([[0, 0], [27, 27]])
>>> stm.plot()

.. [#Bardeen] J. Bardeen,
              Tunneling from a many-body point of view.
              *Phys. Rev. Lett* **6**, 57-59 (1961)

.. [#Lorente] S. Garcia, A. Garcia, N. Lorente and P. Ordejon,
              Optimal strictly localised basis sets for noble metal
              surfaces.
              *Phys. Rev. B* **79**, 0754411-0754419 (2009)

