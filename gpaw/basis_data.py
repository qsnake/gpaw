# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import xml.sax

import numpy as np

from gpaw import setup_paths
from gpaw.setup_data import search_for_file

try:
    import gzip
except:
    has_gzip = False
else:
    has_gzip = True


class Basis:
    def __init__(self, symbol, name, readxml=True):
        self.symbol = symbol
        self.name = name
        self.bf_j = []
        self.ng = None
        self.d = None
        self.generatorattrs = {}
        self.generatordata = ''

        if readxml:
            self.read_xml()

    def read_xml(self, filename=None):
        parser = BasisSetXMLParser(self)
        parser.parse(filename)
            
    def write_xml(self):
        """Write basis functions to file.
        
        Writes all basis functions in the given list of basis functions
        to the file "<symbol>.<name>.basis".
        """
        if self.name is None:
            filename = '%s.basis' % self.symbol
        else:
            filename = '%s.%s.basis' % (self.symbol, self.name)
        write = open(filename, 'w').write
        write('<paw_basis version="0.1">\n')
        
        generatorattrs = ' '.join(['%s="%s"' % (key, value)
                                   for key, value
                                   in self.generatorattrs.iteritems()])
        write('  <generator %s>' % generatorattrs)
        for line in self.generatordata.split('\n'):
            write('\n    '+line)
        write('\n  </generator>\n')
        write(('  <radial_grid eq="r=d*i" d="%f" istart="0" iend="%d" ' +
               'id="lingrid"/>\n') % (self.d, self.ng - 1))

        for bf in self.bf_j:
            write('  <basis_function l="%d" rc="%f" type="%s" '
                  'grid="lingrid" ng="%d">\n'% 
                  (bf.l, bf.rc, bf.type, bf.ng))
            write('   ')
            for value in bf.phit_g:
                write(' %16.12e' % value)
            write('\n')
            write('  </basis_function>\n')
        write('</paw_basis>\n')


class BasisFunction:
    """Encapsulates various basis function data."""
    def __init__(self, l=None, rc=None, phit_g=None, type=''):
        self.l = l
        self.rc = rc
        self.phit_g = phit_g
        self.ng = None
        if phit_g is not None:
            self.ng = len(phit_g)
        self.type = type


class BasisSetXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self, basis):
        xml.sax.handler.ContentHandler.__init__(self)
        self.basis = basis
        self.type = None
        self.rc = None
        self.data = None
        self.l = None

    def parse(self, filename=None):
        basis = self.basis
        name = '%s.%s.basis' % (basis.symbol, basis.name)
        if filename is None:
            basis.filename, source = search_for_file(name)
            if source is None:
                print """
You need to set the GPAW_SETUP_PATH environment variable to point to
the directory where the basis set files are stored.  See

  http://wiki.fysik.dtu.dk/gpaw/Setups

for details."""

                raise RuntimeError('Could not find "%s" basis for "%s".' %
                                   (basis.name, basis.symbol))
        else:
            basis.filename = filename
            source = open(filename).read()

        self.data = None
        xml.sax.parseString(source, self)

    def startElement(self, name, attrs):
        basis = self.basis
        if name == 'paw_basis':
            basis.version = attrs['version']
        elif name == 'generator':
            basis.generatorattrs = dict(attrs)
            self.data = []
        elif name == 'radial_grid':
            assert attrs['eq'] == 'r=d*i'
            basis.ng = int(attrs['iend']) + 1
            basis.d = float(attrs['d'])
            assert int(attrs['istart']) == 0
        elif name == 'basis_function':
            self.l = int(attrs['l'])
            self.rc = float(attrs['rc'])
            self.type = attrs.get('type')
            self.ng = int(attrs.get('ng'))
            self.data = []

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        basis = self.basis
        if name == 'basis_function':
            phit_g = np.array([float(x) for x in ''.join(self.data).split()])
            bf = BasisFunction(self.l, self.rc, phit_g, self.type)
            assert bf.ng == self.ng, ('Bad grid size %d vs ng=%d!'
                                      % (bf.ng, self.ng))
            basis.bf_j.append(bf)
        elif name == 'generator':
            basis.generatordata = ''.join([line for line in self.data])

class BasisPlotter:
    def __init__(self, premultiply=True, normalize=False,
                 show=False, save=False, ext='png'):
        self.premultiply = premultiply
        self.show = show
        self.save = save
        self.ext = ext
        self.default_filename = '%(symbol)s.%(name)s.' + ext

        self.title = 'Basis functions: %(symbol)s %(name)s'
        self.xlabel = r'r [Bohr]'
        if premultiply:
            ylabel = r'$\tilde{\phi} r$'
        else:
            ylabel = r'$\tilde{\phi}$'
        self.ylabel = ylabel

        self.normalize = normalize

    def plot(self, basis, filename=None, **plot_args):
        import pylab # Should not import in module namespace
        if plot_args is None:
            plot_args = {}
        rc = basis.d * (basis.ng - 1)
        r_g = np.linspace(0., rc, basis.ng)

        print 'Element  :', basis.symbol
        print 'Name     :', basis.name
        print 'Filename :', basis.filename
        print
        print 'Basis functions'
        print '---------------'

        norm_j = []
        for j, bf in enumerate(basis.bf_j):
            rphit_g = r_g[:bf.ng] * bf.phit_g
            norm = (np.dot(rphit_g, rphit_g) * basis.d) ** .5
            norm_j.append(norm)
            print bf.type, '[norm=%0.4f]' % norm

        print
        print 'Generator'
        for key, item in basis.generatorattrs.iteritems():
            print '   ', key, ':', item
        print
        print 'Generator data'
        print basis.generatordata

        if self.premultiply:
            factor = r_g
        else:
            factor = np.ones_like(r_g)

        pylab.figure()
        for norm, bf in zip(norm_j, basis.bf_j):
            y_g = bf.phit_g * factor[:bf.ng]
            if self.normalize:
                y_g /= norm
            pylab.plot(r_g[:bf.ng], y_g, label=bf.type[:12],
                       **plot_args)
        axis = pylab.axis()
        rc = max([bf.rc for bf in basis.bf_j])
        newaxis = [0., rc, axis[2], axis[3]]
        pylab.axis(newaxis)
        pylab.legend()
        pylab.title(self.title % basis.__dict__)
        pylab.xlabel(self.xlabel)
        pylab.ylabel(self.ylabel)

        if filename is None:
            filename = self.default_filename
        if self.save:
            pylab.savefig(filename % basis.__dict__)

        if self.show:
            pylab.show()
