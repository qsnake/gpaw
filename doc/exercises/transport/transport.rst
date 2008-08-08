.. _transport_exercise:

==============================
Exercise on electron transport
==============================

Recent experiments suggests that a hydrogen molecule trapped between metal 
electrodes has a conductance close to the quantum unit of conductance. 
In this exercise we will study the electrical proerties of hydrogen molecule
between Pt leads. The leads will be approximated by semi-infinite 
one-dimensional Pt atomic chains. Setup the list of atoms (``Atoms``) for the 
scattering region corresponding to the structure shown in the figure below.

.. image:: pt_h2.png

You can use the following bond lengths: d(Pt-Pt)=2.41, d(Pt-H)=1.7 and
d(H-H)=0.9.

.. literalinclude:: transport.py

Run this :svn:`script <doc/exercises/transport/transport.py?format=txt>`.

What is the conductance?
    
We will now investigate which molecular orbital is responsible
for the rather high conductance. The molecular subspace can be
diagonalized by running::
    
    eps_n, psi_jn = calc_tran.sub_diagonalize([n1,n2]),

where ``n1`` and ``n2`` corresponds to the index of the hydrogen atoms
in your atoms list.

Argue that ``psi_n[:,0]`` and ``psi_n[:,1]`` corresponds to the bonding and 
anti-bonding molecular hydrogen orbitals, respectively. 

What is the calculated band-gap of the hydrogen-molecule?

Try to plot the molecular orbital projected density of states which
can be obtained like this::

    pdos_n = calc_tran.get_pdos(psi_jn)

Which orbital do you think is responsible for the high conductance?

A direct way of visualizing the current carying orbitals is by 
constructure the eigenchannel state. This can be done by running::

    psi = calc_tran.get_eigenchannel_array(energy=0.0,channel=0)
    write(filename='psi.cube',atoms=atoms,data=psi)

Try to plot the eigenchannel state at the energy of the bonding- and
anti-bonding molecular orbital.
