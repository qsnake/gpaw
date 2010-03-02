.. _transport:

.. default-role:: math

=========
Transport
=========

The Transport object in GPAW has been written as a calculator, 
different for other calculators, it supports open boundary
condition, it needs the information of the electrodes and 
scattering region, and can calculate the density, hamiltonian, 
total energy, forces and current as well.

Quick Overview
--------------

This is a small script to get an i-v curve for a sodium 
atomic chain, which includes some main concepts in transport.

.. literalinclude:: transport.py


Model Picture
-------------

::

        ______ ______ ____________ ______ ______ 
       |      |      |     __     |      |      |
   ....|______|______|___ /  \____|______|______|....
       |      |      |    \__/    |      |      |
       |______|______|____________|______|______|
       |      |                          |      |
       | Lead |---->   Scattering   <----| Lead |
       |      |          region          |      |



How to describe an open system
------------------------------

The total open system includes some semi-infinite electrodes
or leads connected to the scattering region we focus on. 
When it is far away enough from the scattering region, 
the lead parts are quite close to the periodical case.
So here we divide the total system into two parts: one is 
the lead parts, there is neither reconstruct nor charge 
transfer, all the information of it can be got from the 
periodical calculation, the other is the scattering region, 
we need a self-consistent procedure to get the properites
here.

Leads
-----

The influence of leads to the scattering region is absorbed
in a item named surface Green's function

.. math::

  g(E) = (E*S_l - H_l)^{-1}

`S_l` and `H_l` are the overlap and hamiltonian matrices of
leads respectively, and since they are inifinite, we need to
do some handling to get it.

We import the concept of priciple layer, the unit cell when 
solving the surface Green's function. We assum the interaction
Hamiltonian only exsits between two adjacent principle layers.
That means the Hamiltonian matrix is tridiagonal by the size
of a principle layer, which is necessary to get the surface
Green's function. 

The selfenergy of lead can be calculated like this

.. math::

  \Sigma _l(E) = \tau _l g(E) \tau _l^{+}

.. math::

  \tau _l = E * S_{lc} - H_{lc}

`S_{lc}` and `H_{lc}` are the coupling overlap and Hamiltonian
matrices. 

Scattering region
-----------------

For the reason mentioned above we need to choose a relatively
big scattering region, some atoms in the leads, or generally
speaking, at least one principle layer should be included in
the scattering region.

The retarded Green's function of scattering region is written as

.. math::

  G^r(E) = ((E + i\eta)*S - H -\Sigma) ^ {-1}

`\Sigma` is the sum of the leads selfeneries.

The lesser Green's function is from the Keldysh formalism

.. math::

  G^<(E) = G^r\Sigma^<G^a

With these two Green's functions, we can get the electron
density in non-equilirbium case which will be introduced
later.

Keywords for Setup
------------------

For leads(necessary):

=================  =========    ============================
keyword            type         description
=================  =========    ============================
``pl_atoms``       list         :ref:`manual_pl_atoms`  
``pl_cells``       list         :ref:`manual_pl_cells`
``pl_kpts``        list         :ref:`manual_pl_kpts`
=================  =========    ============================

Keywords for gpaw is inherited, which is used to descibe
scattering region. Something special for Transport is:

* ``mode`` should be ``'lcao'`` always.

* if use fixed_boundary_condition, ``poissonsolver`` should be set
  like ``PoissonSolver(nn=x)``.

* ``usesymm`` does not act, Transport set a value for it automatically.
 
Other keywords:


====================  =====  =============  ==============================
keyword               type   default value  description
====================  =====  =============  ==============================
``bias``              list   [0, 0]         :ref:`manual_bias`  
``gate``              float  0              :ref:`manual_gate`
``fixed_boundary``    bool   True           :ref:`manual_fixed_boundary`
``lead_restart``      bool   False          :ref:`manual_lead_restart`
``scat_restart``      bool   False          :ref:`manual_scat_restart`
``cal_loc``           bool   False          :ref:`manual_cal_loc`
``recal_path``        bool   False          :ref:`manual_recal_path`
``use_buffer``        bool   False          :ref:`manual_use_buffer`
``buffer_atoms``      list   []             :ref:`manual_buffer_atoms`
``edge_atoms``        list   []             :ref:`manual_edge_atoms`
``use_qzk_boundary``  bool   False          :ref:`manual_use_qzk_boundary`
``identical_leads``   bool   False          :ref:`manual_identical_leads`
``non_sc``            bool   False          :ref:`manual_non_sc`
====================  =====  =============  ==============================

Usage:
Get an iv curve:

>>> from gpaw.transport.calculator import Transport
>>> atoms = Atoms(...)
>>> t = Transport(...)
>>> atoms.set_calculator(t)
>>> t.calculate_iv(3., 16)
  
Optimize:

>>> from gpaw.transport.calculator import Transport  
>>> atoms = Atoms(...)
>>> t = Transport(...)
>>> atoms.set_calculator(t)
>>> dyn = QuasiNewton(atoms, trajectory='xxx.traj')
>>> dyn.run(fmax=0.05)

Analysis:

>>> from gpaw.transport.analysor import Transport_Plotter
>>> plotter = Transport_Plotter()
>>> data = plotter.get_info(XXX, bias_step, ion_step) 

Transport_Plotter now just get the data, users need to plot the data themselves.
XXX can be one in the list ['tc', 'dos', 'force', 'lead_fermi', 'bias', 'gate', 'nt', 'vt'].
The analysis functionality only works after a transport calculation
is done successfully and the directory analysis_data and some files in it are generated. 
 
.. _manual_pl_atoms:


Principle Layer Atoms
---------------------

``pl_atoms`` is the index of lead atoms, whose length is the 
number of leads. For example, [[0,1,2,3],[7,8,9,10]] means there
are two leads, [0,1,2,3] is the principle layer of the first
lead and [7,8,9,10] for the second. The sequence is arbitary.

.. _manual_pl_cells:

Principle Layer Cells
---------------------

``pl_cells`` is a list of leads' cells, also has the same length
with the leads number. [[10., 10., 30], [10., 10., 30.]] for example.

.. _manual_pl_kpts:

Principle Layer K-points
------------------------

``pl_kpts`` is k-points sampling for leads, it is a 1*3 int sequence.
We just let all the leads have the same K number. Attention here that
the k number in the transport direction should bigger than 3, 
in principle we should have enough k points in this direction, an
experenced rule is nK * L(Å) ~ 50. L is the length of unit cell
in this direction.

.. _manual_bias:

Bias
----

``bias`` is a list of bias value for all the leads. For example,
[1.0, -1.0] means we have two leads with the bias shift 1.0 and -1.0
respectively.

.. _manual_gate:

Gate
----

``gate`` is a float number that should only make some sense with 
the fixed boundary condition. The atoms on what a constant gate
is applied is the total scattering region minus the lead's
principle layers.

.. _manual_fixed_boundary:

Fixed Boundary Condition
------------------------

``fixed_boundary`` is a bool option. If set True, we solve the
Poisson equation for the scattering region with fixed boundary
condition. It workes when ``pbc`` in the transport direction
for the scattering region is False and ``poissonsolver=PoissonSolver(nn=X)``. 
If set False, Transport object will deal with a 
regular gpaw option which depends on ``pbc``.

.. _manual_lead_restart:

Lead Calculation Restart
------------------------

``lead_restart`` is a bool option decides if restart from
some previous calculation or not. It is for leads especially.

.. _manual_scat_restart:

Scattering Region Calculation Restart
-------------------------------------

``scat_restart`` same with ``lead_restart``, this is just
for scattering region.

.. _manual_cal_loc:

Double Path Integral
--------------------

``cal_loc`` If set True, Transport will adopt a more complicate
schem for the Green's function integral, which has better
precision and costs more cpu time at the same time.

.. _manual_recal_path:

Recalculate Integral Path
-------------------------

``recal_path`` When doing the Green's function integral,
the energy points on the integral path depends on the 
hamiltonian. The default option is we get the path info with
a hamiltonian guess and then fix it, it often works fine,
saving much time for calculation the leads selfenergy, but when
the guess is not good enough, the integral result differs 
from the real value. This keyword will force to refine the 
energy points on path in each SCF iteration, it should be
a option when can not get a convergence result.

.. _manual_use_buffer:

Use Buffer Atoms
----------------

``use_buffer`` Buffer atoms are needed somtime, this part of
atoms are included when calculating hamiltonian, but will be 
neglected when calculating the Green's function, that means
the density of this part is fixed. If set True, you should
provide the information in ``buffer_atoms``.

.. _manual_buffer_atoms:

Buffer Atoms Indices
--------------------

``buffer_atoms`` is the list of buffer atoms index, just like
``pl_atoms``.

.. _manual_edge_atoms:

Edge Atoms
----------

``edge_atoms`` If ``align`` is True, you need to point which atom
is used to align the energy levels, that means in ground
state calculation, the hamiltonian matrix should have the same 
number in the orbitals of that atom. It is a list includes 
two sub lists. For example, [[0,3],[0,9]] means the atom 0 in
lead1, atom 3 in lead2 are equal to atom 0, atom 9 in scattering 
region respectively.

.. _manual_use_qzk_boundary:

Use Qzk Boundary Condition
--------------------------

``use_qzk_boundary`` is a particular keyword corresponding to
a method introduced in the paper J.Chem.Phys 194710, 127(2007)

.. _manual_identical_leads:

Identical Leads
---------------

When the two electrodes are exactly the same, including cell, atom positions,
and also stacking, set ``identical_leads`` as True can help to save
some time for electrode calculation.

.. _manual_non_sc:

NonSelfConsistent
-----------------

When ``non_sc`` is True, the scattering region is calculated by a normal
DFT using a periodical slab. 





