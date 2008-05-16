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

def c7(symbol1, symbol2, a):
    z = 0.6210
    c = 12.927
    atoms = Atoms(symbols='%s2%s4%s2%s4' % (symbol1, symbol2, symbol1, symbol2),
                  pbc=True,
                  positions=[
        (0.,    4./6.,    1./4.),
        (0.,    2./6.,    3./4.),
        (0.,    2./6.,  z-1./2.),
        (0.,    4./6.,  z),
        (0.,    4./6., -z+3./2.),
        (0.,    2./6., -z+1),
        (1./2., 1./6.,    1./4.),
        (1./2., 5./6.,    3./4.),
        (1./2., 5./6.,  z-1./2.),
        (1./2., 1./6.,  z),
        (1./2., 1./6., -z+3./2.),
        (1./2., 5./6., -z+1.)
        ])
    atoms.set_cell([a, a*sqrt(3.0), c])
    return atoms

def fluorite(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4%s4' % (symbol1, symbol2, symbol2),
                  pbc=True,
                  positions=[
        (.0, .0, .0),
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
        (.25, .25, .75)
        ])
    atoms.set_cell([a, a, a])
    return atoms

def zincblende(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4' % (symbol1, symbol2),
                  pbc=True,
                  positions=[
        (.0, .0, .0),
        (.0, .5, .5),
        (.5, .0, .5),
        (.5, .5, .0),
        (.25, .25, .25),
        (.25, .75, .75),
        (.75, .25, .75),
        (.75, .75, .25)
        ])
    atoms.set_cell([a, a, a])
    return atoms

def cesiumchloride(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s1%s1' % (symbol1, symbol2),
                  pbc=True,
                  positions=[
        (.0, .0, .0),
        (.5, .5, .5)
        ])
    atoms.set_cell([a, a, a])
    return atoms

def rocksalt(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4' % (symbol1, symbol2),
                  pbc=True,
                  positions=[
        (.0, .0, .0),
        (.0, .5, .5),
        (.5, .0, .5),
        (.5, .5, .0),
        (.5, .5, .5),
        (.5, .0, .0),
        (.0, .5, .0),
        (.0, .0, .5)
        ])
    atoms.set_cell([a, a, a])
    return atoms

def alloy(structure, symbol1, symbol2, a):
    atoms = {
        'fluorite':   fluorite(symbol1, symbol2, a),
        'zincblende': zincblende(symbol1, symbol2, a),
        'cesiumchloride': cesiumchloride(symbol1, symbol2, a),
        'rocksalt':   rocksalt(symbol1, symbol2, a),
        'c7':   c7(symbol1, symbol2, a)
        }[structure]
    return atoms

#SiO = zincblende('Si', 'O', 7.)
#view(SiO)

#NaCl = rocksalt('Na', 'Cl', 5.64)
#view(NaCl)

#CaF2 = fluorite('Ca', 'F', 5.64).repeat([2,2,2])
#view(CaF2)

#MoSe2 = c7('Si', 'O', 3.289).repeat([4,4,3])
#view(MoSe2)
