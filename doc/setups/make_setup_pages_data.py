import os
import pickle
from sys import argv

from ase.atoms import string2symbols
from ase.data import atomic_numbers, atomic_names
from ase.data.molecules import rest
from ase.data.molecules import data as molecule_data

from gpaw.atom.generator import Generator, parameters
from gpaw.atom.analyse_setup import analyse
from gpaw.testing.dimer import TestAtom
from gpaw.testing.atomization_data import atomization_vasp
from gpaw.mpi import world

systems = atomization_vasp.keys()


def make(symbol, show):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    if not os.path.isfile(symbol + '.pckl'):
        if os.path.isfile(symbol + '-generator.pckl'):
            gen = pickle.load(open(symbol + '-generator.pckl'))
        else:
            gen = Generator(symbol, 'PBE', scalarrel=True)
            gen.run(logderiv=True, **parameters[symbol])
            gen.txt = None
            pickle.dump(gen, open(symbol + '-generator.pckl', 'w'))
    
        tables = analyse(gen, show=show)

        t = TestAtom(symbol)
        t.run(True, False)
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
        pickle.dump((tables, hmin, B, n, mols), open(symbol + '.pckl', 'w'))
    else:
        tables, hmin, B, n, mols = pickle.load(open(symbol + '.pckl'))

    return tables, hmin, B, n, mols

if len(argv) == 1:
    symbols = parameters.keys()
    symbols = ['H', 'Li', 'He', 'Be']
    show = False
else:
    symbols = argv[1:]
    show = True

for n, symbol in enumerate(symbols):
    if n % world.size == world.rank:
        make(symbol, show)
world.barrier()
