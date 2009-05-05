"""Minimal orthorhombic unitcells for bulk crystals"""

from ase import Atoms, view
from math import sqrt

binary_compounds = {
    'LiF':   (('Li', 1), ('F' , 1)),
    'LiCl':  (('Li', 1), ('Cl', 1)),
    'NaF':   (('Na', 1), ('F' , 1)),
    'NaCl':  (('Na', 1), ('Cl', 1)),
    'TiC':   (('Ti', 1), ('C' , 1)),
    'VC':    (('V' , 1), ('C' , 1)),
    'ZrC':   (('Zr', 1), ('C' , 1)),
    'NbC':   (('Nb', 1), ('C' , 1)),
    'HfC':   (('Hf', 1), ('C' , 1)),
    'ScN':   (('Sc', 1), ('N' , 1)),
    'TiN':   (('Ti', 1), ('N' , 1)),
    'VN':    (('V' , 1), ('N' , 1)),
    'YN':    (('Y' , 1), ('N' , 1)),
    'ZrN':   (('Zr', 1), ('N' , 1)),
    'NbN':   (('Nb', 1), ('N' , 1)),
    'LaN':   (('La', 1), ('N' , 1)),
    'HfN':   (('Hf', 1), ('N' , 1)),
    'MgO':   (('Mg', 1), ('O' , 1)),
    'CaO':   (('Ca', 1), ('O' , 1)),
    'MgS':   (('Mg', 1), ('S' , 1)),
    'MnO':   (('Mn', 1), ('O' , 1)),
    'FeO':   (('Fe', 1), ('O' , 1)),
    'CoO':   (('Co', 1), ('O' , 1)),
    'NiO':   (('Ni', 1), ('O' , 1)),
    'ZnO':   (('Zn', 1), ('O' , 1)),
    'FeAl':  (('Fe', 1), ('Al', 1)),
    'CoAl':  (('Co', 1), ('Al', 1)),
    'NiAl':  (('Ni', 1), ('Al', 1)),
    'BN':    (('B' , 1), ('N' , 1)),
    'BP':    (('B' , 1), ('P' , 1)),
    'BAs':   (('B' , 1), ('As', 1)),
    'AlN':   (('Al', 1), ('N' , 1)),
    'AlP':   (('Al', 1), ('P' , 1)),
    'AlAs':  (('Al', 1), ('As', 1)),
    'GaN':   (('Ga', 1), ('N' , 1)),
    'GaP':   (('Ga', 1), ('P' , 1)),
    'GaAs':  (('Ga', 1), ('As', 1)),
    'InN':   (('In', 1), ('N' , 1)),
    'InP':   (('In', 1), ('P' , 1)),
    'InAs':  (('In', 1), ('As', 1)),
    'SiC':   (('Si', 1), ('C' , 1)),
    'BN':    (('B' , 1), ('N' , 1)),
    'CeO2':  (('Ce', 1), ('O' , 2)),
    'MoSe2': (('Mo', 1), ('Se', 2))
    }

def perovskite(symbol1, symbol2, symbol3, a):
    """Perovskite - Calcium Titanate"""
    return Atoms(symbols='%s%s%s3' % (symbol1, symbol2, symbol3), pbc=True,
                 cell=[a, a, a],
                 scaled_positions=[(.0, .0, .0),
                                   (.5, .5, .5),
                                   (.5, .0, .5),
                                   (.5, .5, .0),
                                   (.0, .5, .5),])

def wurtzite(symbol1, symbol2, a, u=None, c=None):
    """Wurtzite - Zinc Oxide"""
    if c is None:
        c = sqrt(8 / 3.) * a
    if u is None:
        u = 3 / 8.
    return Atoms(symbols='%s4%s4' % (symbol1, symbol2), pbc=True,
                 cell=[a, a * sqrt(3), c],
                 scaled_positions=[(0.00, 0.00,  0.00),
                                   (1/2., 1/2.,  0.00),
                                   (0.00, 1/3.,  1/2.),
                                   (1/2., 5/6.,  1/2.),
                                   (0.00, 0.00,     u),
                                   (1/2., 1/2.,     u),
                                   (0.00, 1/3.,u+1/2.),
                                   (1/2., 5/6.,u+1/2.),])

def c7(symbol1, symbol2, a):
    """C7"""
    z = 0.6210
    c = 12.927
    return Atoms(symbols='%s4%s8' % (symbol1, symbol2), pbc=True,
                 cell=[a, a * sqrt(3), c],
                 scaled_positions=[(0.,    4./6.,    1./4.),
                                   (0.,    2./6.,    3./4.),
                                   (1./2., 1./6.,    1./4.),
                                   (1./2., 5./6.,    3./4.),
                                   (0.,    2./6.,  z-1./2.),
                                   (0.,    4./6.,        z),
                                   (0.,    4./6., -z+3./2.),
                                   (0.,    2./6.,     -z+1),
                                   (1./2., 5./6.,  z-1./2.),
                                   (1./2., 1./6.,        z),
                                   (1./2., 1./6., -z+3./2.),
                                   (1./2., 5./6.,    -z+1.),])

def fluorite(symbol1, symbol2, a):
    """Flourite - Calcium Flouride"""
    return Atoms(symbols='%s4%s8' % (symbol1, symbol2), pbc=True,
                 cell=[a, a, a],
                 scaled_positions=[(.0, .0, .0),
                                   (.0, .5, .5),
                                   (.5, .0, .5),
                                   (.5, .5, .0),
                                   (.25, .25, .25),
                                   (.25, .75, .75),
                                   (.75, .25, .75),
                                   (.75, .75, .25),
                                   (.75, .75, .75),
                                   (.75, .25, .25),
                                   (.25, .75, .25),
                                   (.25, .25, .75),])

def zincblende(symbol1, symbol2, a):
    """Zinc Blende - Zinc Sulfide"""
    return Atoms(symbols='%s2%s2' % (symbol1, symbol2), pbc=True,
                 cell=[a / sqrt(2), a / sqrt(2), a],
                 scaled_positions=[(.0, .0, .0),
                                   (.5, .5, .5),
                                   (.0, .5, .75),
                                   (.5, .0, .25),])

def cesiumchloride(symbol1, symbol2, a):
    """Cesium Chloride"""
    return Atoms(symbols='%s%s' % (symbol1, symbol2), pbc=True, cell=[a, a, a],
                 scaled_positions=[(.0, .0, .0), (.5, .5, .5),])

def rocksalt(symbol1, symbol2, a):
    """Rock Salt - Sodium Chloride"""
    return Atoms(symbols='%s2%s2' % (symbol1, symbol2), pbc=True,
                 cell=[a / sqrt(2), a / sqrt(2), a],
                 scaled_positions=[(.0, .0, .0),
                                   (.5, .5, .5),
                                   (.5, .5, .0),
                                   (.0, .0, .5),])

def hcp(symbol, a, c=None):
    """Hexagonal Close-Packed Lattice"""
    if c is None:
        c = sqrt(8/3.) * a
    return Atoms(symbols='%s4' % symbol, pbc=True,
                 cell=[a, a * sqrt(3), c],
                 scaled_positions=[(0.00, 0.00, 0.00),
                                   (1/2., 1/2., 0.00),
                                   (0.00, 1/3., 1/2.),
                                   (1/2., 5/6., 1/2.),])

def fcc(symbol, a):
    """Face Centered Cubic"""
    return Atoms(symbols='%s2' % symbol, pbc=True,
                cell=[a / sqrt(2), a / sqrt(2), a],
                scaled_positions=[(.0, .0, .0), (.5, .5, .5),])

def bcc(symbol, a):
    """Body Centered Cubic"""
    return Atoms(symbols='%s2' % symbol, pbc=True, cell=[a, a, a],
                 scaled_positions=[(.0, .0, .0), (.5, .5, .5),])

def sc(symbol, a):
    """Simple Cubic"""
    return Atoms(symbols='%s2' % symbol, pbc=True, cell=[a, a, a])

def alloy(structure, symbol1, symbol2, a):
    return eval(structure)(symbol1, symbol2, a)

## ZnS = zincblende('Zn', 'S', 5.41).repeat([2, 2, 2])
## view(ZnS)

## NaCl = rocksalt('Na', 'Cl', 5.64)
## view(NaCl)

## CaF2 = fluorite('Ca', 'F', 5.64).repeat([2, 2, 2])
## view(CaF2)

## MoSe2 = c7('Si', 'O', 3.289).repeat([4, 4, 3])
## view(MoSe2)

## CaTiO3 = perovskite('Ca', 'Ti', 'O', 3.84)
## view(CaTiO3)

## Be = hcp('Be', 2.29)
## view(Be)

## ZnO = wurtzite('Zn', 'O', 3.25, c=5.23).repeat([2, 2, 2])
## view(ZnO)
