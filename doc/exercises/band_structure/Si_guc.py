from ase import *
from gpaw import *

a = 5.475
atoms = Atoms(symbols='Si2', pbc=True,
              cell=.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              scaled_positions=[(.00, .00, .00),
                                (.25, .25, .25)])

#set usenewlfc=1 for this to work!

if 1:
    # Make self-consistent calculation and save results
    calc = GPAW(h=.25, kpts=(8, 8, 8), width=.05,
                nbands=6, txt='Si_sc.txt')
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Si_sc.gpw')

# Special points in the IBZ of an fcc primitive cell
G = np.array([0., 0., 0.])
L = np.array([0.5, 0.5, 0.5])
X = np.array([0.0, 0.5, 0.5])
K = np.array([0.375, 0.375, 0.75])

# The path for the band plot
path = [L, G, X, K, G]
textpath = [r'$L$', r'$\Gamma$', r'$X$', r'$K$', r'$\Gamma$']

# Make kpts list
Npoints = 20
previous = path[0]
kpts = []
points = [0,] # Indices in the kpts list of the special points
for next in path[1:]:
    length = np.linalg.norm(next - previous)
    for t in np.linspace(0, 1, int(round(Npoints * length))):
        kpts.append((1 - t) * previous + t * next)
    points.append(len(kpts))
    previous = next

if 1: # Calculate band structure along specified path
    calc = GPAW('Si_sc.gpw', txt='Si_harris.txt',
                kpts=kpts, fixdensity=True, nbands=8,
                eigensolver='cg', convergence={'bands': -2})
    calc.get_potential_energy()
    calc.write('Si_harris.gpw')

if 1: # Plot the band structure
    import pylab as pl

    # Extract the eigenvalues
    calc = GPAW('Si_harris.gpw', txt=None)
    nbands = calc.get_number_of_bands()
    kpts = calc.get_ibz_k_points()
    nkpts = len(kpts)
    eigs_nk = np.empty((nbands, nkpts), float)
    for k in range(nkpts):
        eigs_nk[:, k] = calc.get_eigenvalues(kpt=k)

    # Subtract Fermi level from the self-consistent calculation
    eigs_nk -= GPAW('Si_sc.gpw', txt=None).get_fermi_level()

    # Do the plot
    for eigs_k in eigs_nk:
        pl.plot(eigs_k, '.m')
    lim = pl.axis()
    for p in points:
        pl.plot([p, p], lim[2:], 'k-')
    pl.xticks(points, textpath)
    pl.title('LDA bandstructure of Silicon')
    pl.show()
