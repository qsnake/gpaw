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

# Make kpts list representing the path between
# special points in the IBZ of an fcc primitive cell
from gpaw.utilities.tools import get_bandpath
G = [.000, .000, .000]
X = [.000, .500, .500]
W = [.500, .750, .250]
K = [.375, .375, .750]
U = [.250, .625, .625]
L = [.500, .500, .500]
path = [L, G, G, X, X, U, K, G]
point_names = [r'$L$', r'$\Gamma$', r'$X$', r'$U,K$', r'$\Gamma$']
kpts, point_indices = get_bandpath(path, atoms.cell, 62)

if 1: # Calculate band structure along specified path
    calc = GPAW('Si_sc.gpw', txt='Si_harris.txt',
                kpts=kpts, fixdensity=True, nbands=9,
                eigensolver='cg', convergence={'bands': 7})
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
    fig = pl.figure(1, figsize=(9, 8), dpi=120)
    for eigs_k in eigs_nk:
        pl.plot(range(nkpts), eigs_k, '.k')
    lim = [0, nkpts, -13, 6]
    for p in point_indices:
        pl.plot([p, p], lim[2:], 'k-')
    pl.xticks(point_indices, point_names)
    pl.axis(lim)
    pl.xlabel('k-vector')
    pl.ylabel('Energy (eV)')
    pl.title('LDA bandstructure of Silicon')
    pl.savefig('Si_guc_band.png', dpi=120)
    pl.show()
