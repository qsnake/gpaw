from gpaw.atom.generator import Generator, parameters

"""Test generation of setups

Run this script to generate exactly those setups needed for running
the tests.
"""

files = []

def gen(symbol, xcname):
    g = Generator(symbol, xcname, scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters[symbol])
    files.append('%s.%s' % (symbol, xcname))

for symbol in ['H', 'He', 'C', 'N', 'O', 'Cl', 'Al', 'Si', 'Na', 'Fe', 'Cu']:
    gen(symbol, 'lxcX-C_PW')
for symbol in ['H', 'He', 'Li', 'N']:
    gen(symbol, 'lxcX_PBE-C_PBE')
for symbol in ['He', 'Li']:
    gen(symbol, 'lxcX_PBE_R-C_PBE')
##for symbol in ['Mg']:
##    gen(symbol, 'GLLB')

if __name__ != '__main__':
    # We have been imported by test.py, so we should clean up:
    from os import remove
    for file in files:
        remove(file)
