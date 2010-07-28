from gpaw import GPAW, FermiDirac
from ase import Atoms

a = 4.23
atoms = Atoms('Na2',
              scaled_positions=[[0, 0, 0], [.5, .5, .5]],
              cell=(a, a, a),
              pbc=True)

# Make self-consistent calculation and save results
calc = GPAW(h=0.25,
            kpts=(8, 8, 8),
            occupations=FermiDirac(width=0.05),
            nbands=3,
            txt='Na_sc.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Na_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 40
kpts = [(0.5 * k / (nkpt - 1), 0, 0) for k in range(nkpt)]

calc = GPAW('Na_sc.gpw',
            txt='Na_harris.txt',
            kpts=kpts,
            fixdensity=True,
            nbands=7,
            parallel={'domain': 1},
            usesymm=None,
            eigensolver='cg',
            convergence={'bands': 'all'})

if calc.input_parameters['mode'] == 'lcao':
    calc.scf.reset()

calc.get_potential_energy()
calc.write('Na_harris.gpw')
