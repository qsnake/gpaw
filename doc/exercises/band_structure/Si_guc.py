""" Example of calculating Si bandstructure for the primitive unit cell.

Compare to e.g. http://en.wikipedia.org/wiki/Electronic_band_structure

"""

import numpy as np
import pylab as plt

from ase import Atoms
from ase.dft.kpoints import ibz_points, get_bandpath

from gpaw import GPAW
from gpaw import FermiDirac

# Use existing calculation
load_gpw = False

# Lattice constant
a = 5.475

atoms = Atoms('Si2',
              scaled_positions=[(.00, .00, .00),
                                (.25, .25, .25)],              
              cell=.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              pbc=True)


if load_gpw:
    calc = GPAW('Si_sc')
else:
    # Make self-consistent calculation and save results
    calc = GPAW(h=0.25,
                kpts=(8, 8, 8),
                occupations=FermiDirac(width=0.05),
                nbands=6,
                txt='Si_sc.txt')

    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    calc.write('Si_sc.gpw')

# Make kpts list representing the path between
# special points in the IBZ of an fcc primitive cell
G = ibz_points['fcc']['Gamma']
X = ibz_points['fcc']['X']
W = ibz_points['fcc']['W']
K = ibz_points['fcc']['K']
U = ibz_points['fcc']['U']
L = ibz_points['fcc']['L']

path = [L, G, G, X, X, U, K, G]

point_names = [r'$L$', r'$\Gamma$', r'$X$', r'$U,K$', r'$\Gamma$']

kpts, point_indices = get_bandpath(path, atoms.cell, npoints=62)

if 1:
    # Calculate band structure along specified path
    calc = GPAW('Si_sc.gpw',
                txt='Si_harris.txt',
                kpts=kpts,
                fixdensity=True,
                nbands=9,
                eigensolver='cg',
                convergence={'bands': 7})
    
    calc.get_potential_energy()

    calc.write('Si_harris.gpw')

if 1:
    # Plot the band structure

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
    fig = plt.figure(1, figsize=(6, 5))

    for eigs_k in eigs_nk:
        plt.plot(range(nkpts), eigs_k, '.k')

    lim = [0, nkpts, -13, 6]

    for p in point_indices:
        plt.plot([p, p], lim[2:], 'k-')

    plt.xticks(point_indices, point_names)
    plt.axis(lim)
    plt.xlabel('k-vector')
    plt.ylabel('Energy (eV)')
    plt.title('LDA bandstructure of Silicon')
    plt.savefig('Si_guc_band.png')
    plt.show()
