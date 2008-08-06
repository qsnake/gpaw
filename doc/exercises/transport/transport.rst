.. _transport_exercise:

==============================
Exercise on electron transport
==============================

Recent experiments suggests that a hydrogen molecule trapped between metal 
electrodes has a conductance close to the quantum unit of conductance. 

.. image:: pt_h2.png

To setup the KS-Hamiltonian

.. literalinclude:: transport.py

Run this :svn:`script <doc/exercises/transport/transport.py?format=txt>`.

What is the conductance?
    
We will now investigate the current carrying orbitals::
    
    eps_n, psi_jn = calc.sub_diagonalize([24,24+2])

Argue that ``psi_n[:,0]`` and ``psi_n[:,1]`` corresponds to the bonding and 
anti-bonding molecular hydrogen orbitals, respectively. 

What is the calculated band-gap of the hydrogen-molecule?


