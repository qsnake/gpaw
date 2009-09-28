from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate setup for oxygen with a core-hole:
if rank == 0:
    g = Generator('O', xcname='PBE', scalarrel=True,
                  corehole=(1, 0, 1.0), nofiles=True)
    g.run(name='fch1s', **parameters['O'])
barrier()
setup_paths.insert(0, '.')

atoms = molecule('H2O')
atoms.center(vacuum=2.5)

calc = GPAW(xc='PBE')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy() + calc.get_reference_energy()

atoms[0].magmom = 1
calc.set(charge=-1, setups={'O': 'fch1s'},fixmom=True,spinpol=True)
e2 = atoms.get_potential_energy() + calc.get_reference_energy()

atoms[0].magmom = 0
calc.set(charge=0, setups={'O': 'fch1s'},fixmom=True,spinpol=True)
e3 = atoms.get_potential_energy() + calc.get_reference_energy()


print 'Energy difference %.3f eV' % (e2 - e1)
print 'XPS %.3f eV' % (e3 - e1)

assert abs(e2 - e1 - 533.117) < 0.001
assert abs(e3 - e1 - 538.623) < 0.001 
