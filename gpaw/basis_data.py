# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import xml.sax
import md5

import numpy as npy

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
        # NOTE: rcs should perhaps be different for orig. and split waves!
        # I.e. have the same shape as basis_lm
        # but right now we just have one rc for each l
        # In fact even that is not supported, so we write the single largest rc
        if self.name is None:
            filename = '%s.basis' % self.symbol
        else:
            filename = '%s.%s.basis' % (self.symbol, self.name)
        write = open(filename, 'w').write
        write('<paw_basis version="0.1">\n')

        #rc = max([bf.rc for bf in self.bf_j])
        # hack since elsewhere multiple rcs are not supported
        generatorattrs = ' '.join(['%s="%s"' % (key, value)
                                   for key, value
                                   in self.generatorattrs.iteritems()])
        write('  <generator %s>' % generatorattrs)
        for line in self.generatordata.split('\n'):
            write('\n    '+line)
        write('\n  </generator>\n')

        #write(('  <radial_grid eq="r=a*i/(n-i)" a="%f" n="%d" ' +
        #      'istart="0" iend="%d" id="g1"/>\n') % (self.beta, self.ng,
        #                                             self.ng-1))
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
        self.rc = rc # Guaranteed to correspond to last element in phit_g
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
            basis.ng = int(attrs['iend']) + 1#int(attrs['n'])
            basis.d = float(attrs['d'])
            assert int(attrs['istart']) == 0
            #assert int(attrs['iend']) == basis.ng - 1
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
            phit_g = npy.array([float(x) for x in ''.join(self.data).split()])
            bf = BasisFunction(self.l, self.rc, phit_g, self.type)
            assert bf.ng == self.ng, ('Bad grid size %d vs ng=%d!'
                                      % (bf.ng, self.ng))
            basis.bf_j.append(bf)
        elif name == 'generator':
            basis.generatordata = ''.join([line for line in self.data])
