from ase.parallel import rank
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional

"""Test generation of setups

Run this script to generate exactly those setups needed for running
the tests.
"""

if rank == 0:
    def gen(symbol, xcname):
        g = Generator(symbol, xcname, scalarrel=True, nofiles=True)
        g.run(exx=True, **parameters[symbol])

    gen('Si','GLLBSC')
    for symbol in ['H', 'He', 'Li', 'C', 'N', 'O', 'Cl', 'Al', 'Si',
                   'Na', 'Fe', 'Cu']:
        gen(symbol, 'LDA')
    for symbol in ['H', 'He', 'Li', 'N']:
        gen(symbol, 'PBE')
    for symbol in ['He', 'Li']:
        gen(symbol, 'revPBE')
