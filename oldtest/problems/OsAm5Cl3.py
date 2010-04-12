# Calculations of Os complex: Os(NH3)5Cl-Cl-Cl
from ase import *
from gpaw import GPAW, MixerSum, Mixer

molecule = Atoms('OsN5H15Cl2')
molecule.positions = [
    (6.550, 6.546, 6.701),
    (6.558, 8.693, 6.783),
    (4.406, 6.546, 6.792),
    (6.558, 4.399, 6.783),
    (8.700, 6.546, 6.776),
    (6.569, 6.546, 4.534),
    (5.730, 3.938, 6.371),
    (7.367, 3.948, 6.324),
    (6.587, 4.124, 7.778),
    (3.946, 5.728, 6.360),
    (3.946, 7.364, 6.360),
    (4.138, 6.546, 7.789),
    (5.730, 9.153, 6.371),
    (7.368, 9.143, 6.324),
    (6.587, 8.967, 7.778),
    (9.152, 5.726, 6.339),
    (9.152, 7.365, 6.339),
    (8.981, 6.546, 7.770),
    (7.052, 5.724, 4.128),
    (7.052, 7.367, 4.128),
    (5.627, 6.546, 4.105),
    (6.608, 6.631, 9.102),
    (4.108, 4.248, 4.703)]
molecule += Atom('Cl', (8.72, 9.04, 4.6))
molecule.center(vacuum=4.0)
molecule.set_constraint(FixAtoms(indices=[0]))
molecule[0].magmom = 1
# Calculator
name = 'osam5cl3'
mixer = MixerSum(0.05, 5, weight=100.0)
calc = GPAW(h=0.18,
            xc='PBE',
            #nbands=-5,
            txt=name + '.txt',
            maxiter=200,
            stencils=(3, 3),
            width=0.1,
            mixer=mixer)
molecule.set_calculator(calc)
molecule.get_potential_energy()
calc.set(width=0.05)
# Relaxation of the Cl atom, and calculations...
qn = QuasiNewton(molecule, trajectory=name + '.traj')
qn.run(fmax=0.05)

write(name + '.cube', molecule)
write(name + '.xyz', molecule)
calc.write(name, mode='all')
