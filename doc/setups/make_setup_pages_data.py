import os
import pickle
from sys import argv

from ase.atoms import string2symbols
from ase.data import atomic_numbers, atomic_names
from ase.data.molecules import rest
from ase.data.molecules import data as molecule_data

from gpaw.atom.generator import Generator, parameters
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.analyse_setup import analyse
from gpaw.testing.dimer import TestAtom
from gpaw.testing.atomization_data import atomization_vasp

systems = atomization_vasp.keys()

def make(symbol, show):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    # Some elements must be done non-scalar-relatistic first:
    if symbol in ['Pt', 'Au', 'Ir', 'Os']:
        AllElectron(symbol, 'LDA', scalarrel=False, nofiles=False).run()
    gen = Generator(symbol, 'PBE', scalarrel=True)
    gen.run(logderiv=True, **parameters[symbol])
    gen.txt = None
    
    tables = analyse(gen, show=show)

    t = TestAtom(symbol)
    t.run(False, False)
    hmin, B, n = t.summary(show=show)

    molecules = []
    for x in systems:
        if symbol in string2symbols(molecule_data[x]['symbols']):
            molecules.append(molecule_data[x]['name'])
    mols = ''
    if molecules:
        names = [rest(m) for m in molecules]
        if len(molecules) == 1:
            mols = names[0]
        elif len(molecules) > 5:
            mols = ', '.join(names[:5]) + ' ...'
        else:
            mols = ', '.join(names[:-1]) + ' and ' + names[-1]
    pickle.dump((tables, hmin, B, n, mols), open(symbol + '.pckl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

if len(argv) == 1:
    symbols = parameters.keys()
    pass  # ...
else:
    symbol = argv[1]
    make(symbol, show=False)
