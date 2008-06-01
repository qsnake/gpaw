#!/usr/bin/env python

import os
import sys
from optparse import OptionParser

import pylab
import numpy as npy
from ase.data import atomic_names

from gpaw.basis_data import Basis

usage = '%prog [options] bases'
parser = OptionParser(usage=usage, version='%prog 0.1')
parser.add_option('-f', '--files', action='store_true',
                  dest='actual_filenames',
                  help='Read from specified filenames rather than searching '+
                  'GPAW setup directories')
parser.add_option('-s', '--save-figs', action='store_true', dest='save',
                  help='Save figures to disk rather than showing plots')
parser.add_option('-r', '--no-r-multiplication', action='store_true',
                  help='Do not pre-multiply wave functions by r in plots')
parser.add_option('-n', '--normalize', action='store_true',
                  help='Plot normalized wave functions')
# The --setups parameter is actually handled by setup_data module internally
# We just provide it here so the parser won't complain when it is used
parser.add_option('--setups', metavar='dir',
                  help='Read setups from specified directory')

opts, files = parser.parse_args()

for path in files:
    dir, filename = os.path.split(path)

    splitfilename = filename.split('.')
    symbol = splitfilename[0]
    extension = splitfilename[-1]
    name = '.'.join(splitfilename[1:-1])

    if opts.actual_filenames:
        basis = Basis(symbol, name, False)
        basis.read_xml(path)
    else: # Search GPAW setup dirs
        basis = Basis(symbol, name)

    rc = basis.d * (basis.ng - 1)
    r_g = npy.linspace(0., rc, basis.ng)

    print 'Element  :', basis.symbol
    print 'Name     :', basis.name
    print 'Filename :', basis.filename
    print
    print 'Basis functions'
    print '---------------'

    norm_j = []
    for j, bf in enumerate(basis.bf_j):
        rphit_g = r_g * bf.phit_g
        norm = (npy.dot(rphit_g, rphit_g) * basis.d) ** .5
        norm_j.append(norm)
        print bf.type, '[norm=%0.4f]' % norm

    print
    print 'Generator'
    for key, item in basis.generatorattrs.iteritems():
        print '   ', key, ':', item
    print
    print 'Generator data'
    print basis.generatordata

    if opts.no_r_multiplication:
        factor = 1.
        ylabel = r'$\tilde{\phi}$'
    else:
        factor = r_g
        ylabel = r'$\tilde{\phi} r$'

    pylab.figure()
    for norm, bf in zip(norm_j, basis.bf_j):
        y_g = bf.phit_g * factor
        if opts.normalize:
            y_g /= norm
        pylab.plot(r_g, y_g, label=bf.type[:12])
    axis = pylab.axis()
    rc = max([bf.rc for bf in basis.bf_j])
    newaxis = [0., rc, axis[2], axis[3]]
    pylab.axis(newaxis)
    pylab.legend()
    pylab.title('Basis functions: %s' % filename)
    pylab.xlabel(r'r [Bohr]')
    pylab.ylabel(ylabel)
    
    if opts.save:
        pylab.savefig('%s.%s.png' % (basis.symbol, basis.name))

if not opts.save:
    pylab.show()
