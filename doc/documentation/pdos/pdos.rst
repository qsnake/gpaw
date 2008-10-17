.. _pdos:

===========================
Projected Density Of States
===========================

.. default-role:: math

The Projected density of states on a state `|\psi_i\rangle` is

.. math::

  \rho_i(\varepsilon) = \sum_n|\langle\psi_i|\psi_n\rangle|^2
  \delta(\varepsilon-\varepsilon_n),

where `\varepsilon_n` is the eigenvalue of the state `|\psi_n\rangle`.

--------------------------------------------
All-Electron PDOS
--------------------------------------------

The all-electron overlaps `\langle\psi_n|\psi_i\rangle` for a Kohn-Sham 
system can be calculated within the PAW formalism from the 
pseudowavefunctions `|\tilde\psi\rangle` and their projector 
overlaps \ [#Blo94]_:

.. math::

  \langle\psi_n|\psi_i\rangle=\langle\tilde\psi_n|\tilde\psi_i\rangle
  +\sum_{a,k,l}\langle\tilde\psi_n|\tilde p_k^a\rangle
  \Big(\langle\phi_k^a|\phi_l^a\rangle-
  \langle\tilde\phi_k^a|\tilde\phi_l^a\rangle\Big)
  \langle\tilde p_l^a|\tilde\psi_i\rangle,

where `\phi_k^a(r)` and  `\tilde p^a_k(r)` are partial waves and 
projector functions of atom  `a`. Thus the projected density of states 
can be calculated for a set of Kohn-Sham states provided that the set is 
projected onto a state `|\psi_i\rangle` which is an eigenstate of another 
Kohn-Sham system.

The example below calculates the density of states for CO adsorbed 
on a Pt(111) slab and the density of states projected onto the gasphase 
orbitals of CO. The ``.gpw`` files can be generated with the script 
:svn:`~doc/documentation/pdos/top.py?format=raw`

PDOS script::

    from gpaw import *
    from pylab import *
    
    # Density of States
    subplot(211)
    calc = Calculator('top.gpw')
    e, dos = calc.get_dos(spin=0, npts=2001, width=0.1)
    plot(e, dos)
    grid(True)
    axis([-15, 10, None, None])
    ylabel('DOS')

    molecule = range(len(calc.nuclei))[-2:]

    subplot(212)
    c_mol = Calculator('CO.gpw')
    for n in range(2,7):
        print 'Band', n
	# PDOS on the band n
        wf_k = [c_mol.get_pseudo_wave_function(band=n, kpt=k, spin=0, pad=False)
                for k in range(c_mol.nkpts)]
        P_aui = [a.P_uni[:,n,:] for a in c_mol.nuclei] # Inner products of pseudo wavefunctions and projectors
        e, dos = calc.get_all_electron_ldos(mol=molecule, spin=0, npts=2001,
                                            width=0.1, wf_k=wf_k, P_aui=P_aui)
        plot(e, dos, label='Band: '+str(n))
    legend()
    grid(True)
    axis([-15, 10, None, None])
    xlabel('Energy [eV]')
    ylabel('All-Electron PDOS')

    show()

When running the script `\int d\varepsilon\rho_i(\varepsilon)` is printed for
each spin and k-point. The value should be close to one if the orbital `\psi_i(r)`
is well represented by an expansion in Kohn-Sham orbitals and thus the integral
is a measure of the completeness of the Kohn-Sham system. The bands 7 and 8 are
delocalized and are not well represented by an expansion in the slab eigenstates 
(Try changing ``range(2,7)`` to ``range(2,9)`` and note the integral is less than 
one).

The function ``calc.get_all_electron_ldos()`` calculates the square modulus
of the overlaps and multiply by normalized gaussians of a certain width. 
The energies is in ``eV`` and relative 
to the Fermi level. Setting the keyword ``raw=True`` will return only the 
overlaps and energies in Hartree. It is useful to simply save these in a ``.pickle``
file since the ``.gpw`` files with wavefunctions can be quite large. The following
script pickles the overlaps

Pickle script::

    from gpaw import *
    import pickle

    calc = GPAW('top.gpw')
    c_mol = GPAW('CO.gpw')
    molecule = range(len(calc.nuclei))[-2:]
    e_n = []
    P_n = []
    for n in range(c_mol.nbands):
        print 'Band: ', n
        wf_k = [c_mol.get_pseudo_wave_function(band=n, kpt=k, spin=0, pad=False)
                for k in range(calc.nkpts)]
        P_aui = [a.P_uni[:,n,:] for a in c_mol.nuclei]
        e, P = calc.get_all_electron_ldos(mol=molecule, wf_k=wf_k, spin=0, P_aui=P_aui, raw=True)
        e_n.append(e)
        P_n.append(P)
    pickle.dump((e_n, P_n), open('top.pickle', 'w'))

and the ``top.pickle`` file can be plottet with

Plot PDOS::

    from ase.units import Hartree
    from gpaw import *
    from gpaw.utilities.dos import fold
    import pickle
    from pylab import *

    e_f = GPAW('top.gpw').get_fermi_level()

    e_n, P_n = pickle.load(open('top.pickle'))
    for n in range(2,7):
        e, ldos = fold(e_n[n] * Hartree, P_n[n], npts=2001, width=0.2)
        plot(e-e_f, ldos, label='Band: '+str(n))
    legend()
    axis([-15, 10, None, None])
    xlabel('Energy [eV]')
    ylabel('PDOS')
    grid(True)

    show()

.. [#Blo94] P. E. Blöchl, Phys. Rev. B 50, 17953 (1994)