from ase import Atoms
from ase.io import write
from gpaw import GPAW, Mixer

# Oxygen atom:
atom = Atoms(['O'], cell=[6.,6.,6.], pbc=False)
atom.center()

# GPAW calculator with 2 unonccupied bands:
calc = GPAW(h=0.2,
            nbands=-2,
            hund=True, #assigns the atom its correct magnetic momentum
            convergence={'bands':'all','eigenstates':1e-5},
            mixer=Mixer(beta=0.1, nmaxold=5, weight=50.0),
            txt='O.txt')

atom.set_calculator(calc)
atom.get_potential_energy()

# Write wave functions to gpw file
calc.write('O.gpw', mode='all')

# Generate cube-files of the orbitals.
for n in range(calc.get_number_of_bands()):
    wf = calc.get_pseudo_wave_function(band=n)
    write('O%d.cube' % n, atom, data=wf)
