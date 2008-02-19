from math import pi, cos, sin, sqrt, acos

import numpy as npy

from ase.atoms import Atoms
from ase.parallel import paropen


def read_xyz(fileobj, index=-1):
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    lines = fileobj.readlines()
    L1 = lines[0].split()
    if len(L1) == 1:
        cell = line2cell(lines[1])
        del lines[:2]
        natoms = int(L1[0])
    else:
        natoms = len(lines)
    images = []
    while len(lines) > 0:
        positions = []
        symbols = []
        for line in lines[:natoms]:
            symbol, x, y, z = line.split()[:4]
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        if len(lines) > natoms + 2:
            newcell = line2cell(lines[natoms + 1])
            if newcell is not None:
                cell = newcell
        images.append(Atoms(symbols=symbols, positions=positions, cell=cell))
        del lines[:natoms + 2]
    return images[index]

def line2cell(line):
    x = line.split()
    if len(x) != 7:
        return None
    try:
        a, b, c = [float(z) for z in x[1:4]]
        A, B, C = [float(z) * pi / 180 for z in x[4:]]
    except ValueError:
        return None
    cell = npy.zeros((3, 3))
    cell[0, 0] = a
    cell[1, 0] = b * cos(C)
    cell[1, 1] = b * sin(C)
    x = c * cos(B)
    y = (c * cos(A) - x * cos(C)) / sin(C)
    z = sqrt(c**2 - x**2 - y**2)
    cell[2] = (x, y, z)
    # handle rounding errors in cos, sin
    cell = npy.where(cell < 1.e-15, 0.0, cell)
    return cell

def write_xyz(fileobj, images):
    if isinstance(fileobj, str):
        fileobj = paropen(fileobj, 'w')

    if not isinstance(images, (list, tuple)):
        images = [images]

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)
    for atoms in images:
        fileobj.write('%d\nUnitCell:' % natoms)
        cell = atoms.get_cell()
        A = [npy.linalg.norm(a) for a in cell]
        for c1 in range(3):
            c2 = (c1 + 1) % 3
            c3 = (c1 + 2) % 3
            A.append(180 / pi *
                     acos(npy.dot(cell[c2], cell[c3]) / A[c2] / A[c3]))
        fileobj.write((' %.6f' * 6) % tuple(A) + '\n')
        for s, (x, y, z) in zip(symbols, atoms.get_positions()):
            fileobj.write('%-2s %22.15f %22.15f %22.15f\n' % (s, x, y, z))
