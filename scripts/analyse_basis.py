#!/usr/bin/env python

import sys

import pylab
import numpy as npy
from ase.data import atomic_names

from gpaw.basis_data import Basis

filename = sys.argv[-1]
splitfilename = filename.split('.')
symbol = splitfilename[0]
extension = splitfilename[-1]
name = '.'.join(splitfilename[1:-1])

# Use -f to specify file directly rather than through gpaw setup paths
load_specific_file = ('-f' in sys.argv)

if load_specific_file:
    basis = Basis(symbol, name, False)
    basis.read_xml(filename)
else:
    basis = Basis(symbol, name)

print 'Element  :', basis.symbol
print 'Name     :', basis.name
print 'Filename :', basis.filename
print
print 'Basis functions'
print '---------------'
for bf in basis.bf_j:
    print bf.type


rc = basis.d * (basis.ng - 1)
r_g = npy.linspace(0., rc, basis.ng)

for bf in basis.bf_j:
    phitr_g = bf.phit_g * r_g
    dr = r_g[1]
    pylab.plot(r_g, phitr_g, label=bf.type[:12])
axis = pylab.axis()
rc = max([bf.rc for bf in basis.bf_j])
newaxis = [0., rc, axis[2], axis[3]]
pylab.axis(newaxis)
pylab.legend()
pylab.title('Basis functions')
pylab.xlabel('r [Bohr]')
pylab.ylabel(r'$\tilde{\phi} r, \rm{[Bohr}^{-3/2}\rm{]}$')
pylab.show()
