from ase import Atoms, view

def zincblende(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4' % (symbol1, symbol2),
                  pbc=True,
                  positions=[(.0, .0, .0),
                             (.0, .5, .5),
                             (.5, .0, .5),
                             (.5, .5, .0),
                             (.25, .25, .25),
                             (.75, .75, .25),
                             (.25, .75, .75),
                             (.75, .25, .75)])
    atoms.set_cell([a, a, a])
    return atoms

def rocksalt(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4' % (symbol1, symbol2),
                  pbc=True,
                  positions=[(.0, .0, .0),
                             (.0, .5, .5),
                             (.5, .0, .5),
                             (.5, .5, .0),
                             (.5, .0, .0),
                             (.0, .5, .0),
                             (.0, .0, .5),
                             (.5, .5, .5)])
    atoms.set_cell([a, a, a])
    return atoms

SiO = zincblende('Si', 'O', 7.)
view(SiO)

NaCl = rocksalt('Na', 'Cl', 5.64)
view(NaCl)
