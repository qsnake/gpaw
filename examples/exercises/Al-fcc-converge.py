from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from ASE.Visualization.VMD import VMD
from ASE.Visualization.gnuplot import gnuplot

filename = 'Al-fcc-converge'

a = 4.05   # fcc lattice paramter
b = a / 2. 

bulk = ListOfAtoms([Atom('Al', (0, 0, 0)),
                    Atom('Al', (b, b, 0)),
                    Atom('Al', (0, b, b)),
                    Atom('Al', (b, 0, b)),],
                   cell=(a, a, a),
                   periodic=(1, 1, 1))

calc = Calculator(nbands=16,              # Set the number of electronic bands
                  h=0.2,                  # Set the grid spacing
                  kpts=(1,1,1),           # Set the k-points
                  txt=filename+'.txt')    # Set output file

bulk.SetCalculator(calc)


# Make a plot of the convergence with respect to k-points
kpt_energies = []
for k in [2, 4, 6, 8, 10, 12]: 
    calc.set(kpts=(k, k, k))
    energy = bulk.GetPotentialEnergy() 
    kpt_energies.append((k, energy))

#kpt_plot = gnuplot(kpt_energies) 

# Make a plot of the convergence with respect to  grid spacing
k = 6
calc.set(kpts=(k, k, k))
gs_energies = []
for gs in [0.4, 0.3, 0.25, 0.2, 0.15]:
    calc.set(h=gs)
    energy = bulk.GetPotentialEnergy()
    gs_energies.append((gs, energy))

#gs_plot = gnuplot(gs_energies)

print kpt_energies
print gs_energies
