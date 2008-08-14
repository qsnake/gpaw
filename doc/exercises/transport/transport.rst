.. _transport_exercise:

==================
Electron transport
==================

.. default-role:: math

This exercise shows how to use the ase transport module for performing
realistic calculations of the electron transport in nanoscale contacts.

The exercise demonstrates the use of the classes:

* :class:`~ase.transport.calculators.TransportCalculator` used to
  calculate transmission functions.
* :class:`GPAWTransport` used to extract realistic descriptions of
  real systems using the GPAW DFT-LCAO mode.

First-time users of the ASE transport module, should start by reading
the methodology in the :ase:`ASE manual <ase/transport/transport.html>`.

Recent experiments suggests that a hydrogen molecule trapped between
metal electrodes has a conductance close to the quantum unit of
conductance (`1G_0=2e^2/h`). The Pt-H2-Pt system will be the system we
have in mind throughout the exercise.

Tight-binding description
=========================

In this part of the exercise, we illustrate the use of the
:class:`~ase.transport.calculators.TransportCalculator` class by means
of a simple tight-binding (TB) model for the Pt-H2-Pt system.

The Hamiltonian of the entire system may be represented by three
Hamiltonian matrices: (i) `H_\text{scat}` which describes the
scattering region including one princpial layer on either side, and
(ii) `H_L` and `H_R` describing lead left and lead right lead
respectively.  Thus `H_\text{scat}` have the generic shape:

.. math::

    H_\text{scat} = \begin{pmatrix}
                       H_{LL} & H_{LS} &  0      \\
                       H_{RL} & H_{SS} &  H_{SR} \\
                       0      & H_{RS} &  H_{RR}
                    \end{pmatrix}

The dimension of `H_\text{scat}` is `(N_S+N_L+N_R) \times
(N_S+N_L+N_R)`.  The dimension of `H_i` is `N_i\times N_i`, `i=L, R`.
Here `N_i` is the number of basis function in one principal layer of
lead i. `N_S` is the number of basis functions in the scattering
region.

We now consider electron transport through a simple model system where
the leads are one-dimensional TB chains with up to second-nearest
neightbour coupling.  The scattering region consists of two TB sites
coupled to the leads.  The system can be viewed as a simple model of a
hydrogen molecule sandwiched between electrodes.

The lead Hamiltonian should include at least two principal layers and
may be constructed like this::

    import numpy as npy
    
    # onsite energies 0.0, nearest neighbor hopping -1.0, and
    # second nearest neighbor hopping 0.2
    H_lead = npy.array([[ 0. , -1. ,  0.2,  0. ],
	                [-1. ,  0. , -1. ,  0.2],
                        [ 0.2, -1. ,  0. , -1. ],
                        [ 0. ,  0.2, -1. ,  0. ]])
    
Next, the Hamiltonian for the scattering region plus one principal
layer on each side should be constructed.  Since this region should
contain 2 TB sites (the 2 hydrogen orbitals) and since a principal
layer contains 2 TB sites, the dimension of `H_\text{scat}` is
`6\times 6`::

    H_scat = npy.zeros((6, 6))

    #Principal layers on either side of S
    H_scat[:2, :2] = H_scat[-2:, -2:] = H_lead[:2, :2]

    #Scatering region (hydrogen molecule) - onsite 0.0 and hopping -0.8
    H_scat[2:4, 2:4] = [[0.0, -0.8], [-0.8, 0.0]]

    #coupling to the leads - nearest neighbor only
    H_scat[1, 2] = H_scat[2, 1] = H_scat[3, 4] = H_scat[4, 3] = 0.2

You are now ready to initialize the ``TransportCalculator``::
    
    from ase.transport.calculators import TransportCalculator

    tcalc = TransportCalculator(h=H_scat,  # Scattering Hamiltonian
                                h1=H_lead, # Lead 1 (left)
                                h2=H_lead, # Lead 2  (right)
                                pl=2)      # principal layer size


To select the energy grid on which we want the transmission use the
``set`` method::

    tcalc.set(energies=npy.arange(-3, 3, 0.02))

Perform the tranmission function calculation::

    T_e = tcalc.get_transmission()

You can try to plot it (e.g. using ``pylab.plot(tcalc.energies, T_e)``).
The projected density of states (pdos) for the two hydrogen TB sites can
be calculated using::

    tcalc.set(pdos=[0, 1])
    pdos_ne = tcalc.get_pdos()
    
Why do you think the pdos of each the hydrogen TB sites has two peaks?

To investigate the system you can try to diagonalize the subspace
spanned by the hydrogen TB sites::

    h_rot, s_rot, eps_n, vec_nn = tcalc.subdiagonalize_bfs([0, 1])
    tcalc.set(h=h_rot,s=s_rot) # Set the rotated matrices

``eps_n[i]`` and ``vec_nn[:,i]`` contains the i'th
eigenvalue and eigenvector of the hydrogen molecule.  
Try to calculate the pdos again. What happpened?

You can try to remove the coupling to the bonding state and
calculate the calculate the transmission function::
    
    tcalc.cutcupling_bfs([0])
    T_cut_bonding_e = tcalc.get_transmission()

You may now understand the transport behavior of the simple model system.
The transmission peak at -0.8 eV and 0.8 eV are due to the
bonding and antibonding states of the TB described hydrogen molecule.
A script containing the above can be found here:
:svn:`script <doc/exercises/transport/pt_h2_tb_transport.py?format=txt>`.

DFT description
===============

We now continue to explore the Pt-H2-Pt system using a more realistic
desciption derived from ab-initio calculations.

The class :class:`GPAWTransport` (in gpaw.lcao.gpawtransport) allows
you to construct such a Hamiltonian within DFT in terms of pseudo
atomic orbitals.

As a managable model for the Pt-H2-Pt, we consider a hydrogen molecule
sandwiched between semi-infinite one dimensional Pt leads. The figure
below shows the scattering region.

.. image:: pt_h2.png
  :align: center

To obtain the matrices for the scattering region and the leads using
DFT and pseudo atomic orbitals using a szp basis set run this 
:svn:`script <doc/exercises/transport/pt_h2_lcao.py?format=txt>`.

You should now have the files scat_hs.pickle, lead1_hs.pickle and
lead2_hs.pickle in your directory.

The ``TransportCalculator`` can now be setup::
    
    from ase.transport.calculators import TransportCalculator
    import numpy as npy
    import pickle


    #Read in the hamiltoniansh, s = pickle.load(file('scat_hs.pickle'))
    h1, s1 = pickle.load(file('lead1_hs.pickle'))
    h2, s2 = pickle.load(file('lead2_hs.pickle'))
    pl1 = len(h1) / 2 # left principal layer size
    pl2 = len(h2) / 2 # right principal layer size

    tcalc = TransportCalculator(h=h, h1=h1, h2=h2, #hamiltonian matrices
                                s=s, s1=s1, s2=s2, #overlap matrices
                                pl1=pl1, pl2=pl2,  #principal layer sizes
                                energies=[0.0],    #energies
                                align_bf=1,        #align the the Fermi levels
                                verbose=False)     #print extra information?


What is the conductance?
    
We will now try to investigate transport properties in more detail.
Try to subdiagonalize the molecular subspace::
   
    Pt_N = 5 # 
    Pt_nbf = 9 #number of bf per Pt atom (basis=szp)
    H_nbf = 4  # number of bf per H atom (basis=szp)
    bf_H1 = Pt_nbf * Pt_N
    bfs = range(bf_H1, bf_H1 + 2 * H_nbf)
    h_rot, s_rot, eps_n, vec_jn = tcalc.subdiagonalize_bfs(bfs)
    for n in range(len(eps_n)):
        print "bf %i correpsonds to the eigenvalue %.2f eV" % (bfs[n],eps_n[n])

Argue that ``vec_jn[:,0]`` and ``vec_jn[:,1]`` corresponds to the bonding and 
anti-bonding molecular hydrogen orbitals, respectively. 

What is the calculated band-gap of the hydrogen-molecule?

Try to plot the molecular orbital projected density of states.

Which orbital do you think is responsible for the high conductance?

Here is a script if you need some inspiration:
:svn:`script <doc/exercises/transport/pt_h2_lcao_transport.py?format=txt>`.

.. default-role::
