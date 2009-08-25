from ase import *
from gpaw import *
from ase.transport.calculators import TransportCalculator as TC
from gpaw.lcao.tools import get_lead_lcao_hamiltonian

'''
1. calculate the transmission function through a chain of Sodium atoms,
   with kpoint sampling over (n, m) k-points of the transverse ibz.

2. calculate the transmission function of that same system repeated n, m 
   times in the transverse directions and where the transverse bz is sampled 
   by the gamma point only.

Ideally the two transmission functions should be the same.

'''

a = 3.00 # Na binding length
L= 7.00 # width of the unit cell
n = 1 # number of atoms in the transport direction

kpts = (5, 5, 14)
energies = np.arange(-5, 20, 0.3)
usesymm = False
basis = 'szp'
direction = 'z'

calc = GPAW(h=0.3,
            xc='PBE',
            mode='lcao',
            mixer=Mixer(0.1, 5, metric='new',weight=80.),
            width=0.1,
            basis=basis)

dir = 'xyz'.index(direction)
transverse_dirs = np.delete([0, 1, 2], [dir]).astype(int)
cell = [L, L, L]
cell[dir] = n * a

atoms = Atoms('Na'+str(n), pbc=1, cell=cell)
atoms.positions[:n, dir] = [i * a for i in range(len(atoms))]
atoms.positions[:, transverse_dirs] = L / 2.
atoms.set_calculator(calc)
calc.set(kpts=kpts, usesymm=usesymm)
atoms.get_potential_energy()

efermi = calc.get_fermi_level()
ibzk2d_k, weight2d_k, h_skmm, s_kmm\
= get_lead_lcao_hamiltonian(calc, direction=direction)
h_kmm = h_skmm[0] - efermi * s_kmm

T_kpts = np.zeros_like(energies)

i = 0
for h_mm, s_mm, weight in zip(h_kmm, s_kmm, weight2d_k):
    print i
    tc = TC(energies=energies,
            h=h_mm, s=s_mm,
            h1=h_mm, s1=s_mm,
            h2=h_mm, s2=s_mm,
            align_bf=0)
    tc.initialize()
    T_kpts += tc.get_transmission() * weight
    i += 1

# repeat the system in the transverse directions
repeat = np.ones((3,))
repeat[transverse_dirs] = np.asarray(kpts)[transverse_dirs].astype(int)
atoms2 = atoms.repeat(repeat)

kpts2 = np.ones((3,))
kpts2[dir] = kpts[dir]
calc.set(kpts=tuple(kpts2.astype(int)), usesymm=usesymm)
atoms2.set_calculator(calc)
atoms2.get_potential_energy()

efermi = calc.get_fermi_level()
ibzk2d_k, weight2d_k, h_skmm, s_kmm\
= get_lead_lcao_hamiltonian(calc, direction=direction)

h_kmm = h_skmm[0] - efermi * s_kmm
s_mm = s_kmm[0]
h_mm = h_kmm[0]

tc = TC(energies=energies,
        h=h_mm, s=s_mm,
        h1=h_mm, s1=s_mm,
        h2=h_mm, s2=s_mm,
        align_bf=0,
        lofile='-')
tc.initialize()
T_gamma = tc.get_transmission()

if 0:
    import pylab
    pylab.plot(energies, T_kpts, '--r')
    pylab.plot(energies, 
               T_gamma / (kpts[transverse_dirs[0]] * kpts[transverse_dirs[1]]), 
               '-b')
    pylab.show()

