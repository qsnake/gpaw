from ase import Atoms, Atom
from gpaw import GPAW, Mixer
from ase.lattice.surface import fcc100
from gpaw.transport.jstm import dump_hs, dump_lead_hs

calc = GPAW(h=0.2, 
            mixer=Mixer(0.03, 5, weight=140.0),
            width=0.1, 
            mode='lcao',
            basis='szp(dzp)',
            txt='dumphs.txt',
            usesymm=False)

# surface calculation
a = 4.0 
srf = fcc100('Al', size=(2, 2, 10), vacuum=8.0)
srf.translate([0, 0, -4]) 
srf.pbc= (1, 1, 0)
srf += Atom('H', (a / 2**.5, a / 2**.5, srf.positions[-1][2] + 1.55))

srf.set_calculator(calc)
srf.get_potential_energy()
calc.write('srf')

# Dump the overlap matrix and the Hamiltonian matrix to the local directory. 
# Here the keyword 'cvl' refers  to the number of basis functions in the 
# in the convergence layer, i.e. for the present system four atomic layers are used.
dump_hs(calc, 'srf', region='surface', cvl=4*4*9) 
                                                 
# tip calculation 
a = 0.75 # lattice constant
tip = Atoms('H12', pbc=(1, 1, 0), cell=[5, 5, 12 * a + 7])
tip.positions[:,2] = [i * a for i in range(12)]
tip.positions[:] += (tip.cell / 2.0)[0, :] + (tip.cell / 2.0)[1, :]
tip.translate([0, 0, 6])

tip.set_calculator(calc)
tip.get_potential_energy()
calc.write('tip')
dump_hs(calc, 'tip', region='tip', cvl=4) # dump overlap and hamiltonian matrix


calc.set(kpts=(1, 1, 7)) # for the lead calculations we use kpoints in the z-direction

# surface principal layer calculation
srf_p = fcc100('Al', size=(2, 2, 4)) 
srf_p.pbc = (1, 1, 1)

srf_p.set_calculator(calc)
srf_p.get_potential_energy()
dump_lead_hs(calc, 'srf_p') # dump overlap and hamiltonian matrix

# tip principal layer calculation
tip_p = Atoms('H4', pbc=(1,1,1), cell=[5, 5, 4*a])
tip_p.positions[:,2] = [i * a for i in range(4)]
tip_p.positions[:] += (tip_p.cell / 2.0)[0, :] + (tip_p.cell / 2.0)[1, :]

tip_p.set_calculator(calc)
tip_p.get_potential_energy()
dump_lead_hs(calc, 'tip_p') # dump overlap and hamiltonian matrix

