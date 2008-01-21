"""Module for building atomic structures"""

from ase import *


def fcc100(symbol, a, layers, L):
    """Build a fcc(100) surface

    symbol: chmical symbol ('H', 'Li', ...)
    a     : lattice constant
    layers: number of layers
    L     : height of unit cell"""

    # Distance between atoms:
    d = a / sqrt(2)

    # Distance between layers:
    z = a / 2

    assert L > layers * z, 'Unit cell too small!'
    
    # Start with an empty Atoms object:
    atoms = Atoms(cell=(d, d, L), pbc=(1, 1, 0))
    
    # Fill in the atoms:
    for n in range(layers):
        position = [d / 2 * (n % 2),
                    d / 2 * (n % 2),
                    n * z]
        atoms.append(Atom(symbol, position))

    atoms.center(axis=2)
    return atoms
