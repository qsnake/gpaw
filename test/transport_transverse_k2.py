from ase import *
from gpaw import *
from ase.transport.calculators import TransportCalculator as TC
from gpaw.lcao.tools import get_lead_lcao_hamiltonian, remove_pbc

"""
1. calculate the transmission function of Sodium bulk with one atom in the
   unit cell and sampling over (n, m) k-points in the transverse ibz.

2. calculate the transmission function of that same system repeated n, m 
   times in the transverse directions and where the transverse bz is sampled 
   by the gamma point only.

Ideally the two transmission functions should be the same.
"""

L = 3.0 # Na binding length
direction = 'x'
dir = 'xyz'.index(direction)
energies = np.arange(-5, 10, 0.2)

def get_trans(h, s):
    return TC(energies=energies, h=h, s=s, h1=h, s1=s, h2=h, s2=s,
            align_bf=0).get_transmission()

def get_hs(natoms, nkpts):
    calc = GPAW(h=0.25, mode='lcao', basis='sz', width=0.2, kpts=nkpts,
                mixer=Mixer(0.1, 5, weight=80.0), usesymm=False)
    atoms = Atoms('Na', pbc=True, cell=(L, L, L)).repeat(natoms)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    fermi = calc.get_fermi_level()
    ibz, w, h, s = get_lead_lcao_hamiltonian(calc, direction=direction)
    h = h[0] - fermi * s
    for h1, s1 in zip(h, s):
        remove_pbc(atoms, h1, s1, d=dir)
    return ibz, w, h, s

# First with transverse kpts
ibz1_kc, w1_k, h1_kmm, s1_kmm = get_hs(natoms=(3, 1, 1), nkpts=(4, 3, 3))
T1_k = [get_trans(h, s) for h, s in zip(h1_kmm, s1_kmm)]
T1 = np.dot(w1_k, T1_k)


# Second without transverse kpts
ibz2_kc, w2_k, h2_kmm, s2_kmm = get_hs(natoms=(3, 3, 3), nkpts=(4, 1, 1))
T2_k = [get_trans(h, s) for h, s in zip(h2_kmm, s2_kmm)]
T2 = np.dot(w2_k, T2_k)

if 1:
    import pylab as pl
    pl.plot(energies, T1, 'r--', label='With trans kpts')
    pl.plot(energies, T2 / (3 * 3), 'b:', label='Without trans kpts')
    pl.axis('tight')
    pl.legend()
    pl.show()

