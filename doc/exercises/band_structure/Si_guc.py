"""
Example of calculating Si bandstructure for the primitive unit cell.
Compare to e.g. http://en.wikipedia.org/wiki/Electronic_band_structure
"""
from ase import *
from gpaw import *

a = 5.475
atoms = Atoms(symbols='Si2', pbc=True,
              cell=.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              scaled_positions=[(.00, .00, .00),
                                (.25, .25, .25)])

if 1: # Make self-consistent calculation and save results
    calc = GPAW(h=.25, kpts=(8, 8, 8), width=.05,
                nbands=6, txt='Si_sc.txt')
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Si_sc.gpw')

# Make kpts list representing the path between
# special points in the IBZ of an fcc primitive cell
from ase.dft.kpoints import ibz_points, get_bandpath
G = ibz_points['fcc']['Gamma']
X = ibz_points['fcc']['X']
W = ibz_points['fcc']['W']
K = ibz_points['fcc']['K']
U = ibz_points['fcc']['U']
L = ibz_points['fcc']['L']
path = [L, G, G, X, X, U, K, G]
point_names = [r'$L$', r'$\Gamma$', r'$X$', r'$U,K$', r'$\Gamma$']
kpts, point_indices = get_bandpath(path, atoms.cell, npoints=62)

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
    fig = pl.figure(1, figsize=(6, 5))
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
    pl.savefig('Si_guc_band.png')
    pl.show()
