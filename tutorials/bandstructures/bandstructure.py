from gpaw import Calculator
from ase import *

a = 4.23
atoms = Atoms('Na2',
              [(0, 0, 0),
               (0.5 * a, 0.5 * a, 0.5 * a)],
              pbc=True, cell=(a, a, a))

# Make self-consistent calculation and save results
h = 0.25
calc = Calculator(h=.25, kpts=(8, 8, 8), width=0.05,
                   nbands=3, txt='out_Na_sc.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Na_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(k / float(2 * nkpt), 0, 0) for k in range(nkpt)]
calc = Calculator('Na_sc.gpw', txt='out_Na_harris.txt',
                    kpts=kpts, fixdensity=True, nbands=7,
                  eigensolver='cg', convergence={'bands': 'all'})
ef = calc.get_fermi_level()
calc.get_potential_energy()

# Write the results to a file e.g. for plotting with gnuplot
f = open('Na_bands.txt', 'w')
for k, kpt_c in enumerate(calc.get_ibz_k_points()):
     for eig in calc.get_eigenvalues(kpt=k):
         print >> f, kpt_c[0], eig - ef
