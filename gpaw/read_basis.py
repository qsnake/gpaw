# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import xml.sax
import md5

import Numeric as num

from gpaw import setup_paths

try:
    import gzip
except:
    has_gzip = False
else:
    has_gzip = True

class BasisSetXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)

        self.l_j = []
        self.phit_jg = []

    def parse(self, symbol, basis_name):
        name = '%s.%s.basis' % (symbol, basis_name)

        source = None
        for path in setup_paths:
            filename = os.path.join(path, name)
            if os.path.isfile(filename):
                source = open(filename).read()
                break
            else:
                filename += '.gz'
                if os.path.isfile(filename):
                    if has_gzip:
                        source = gzip.open(filename).read()
                    else:
                        source = os.popen('gunzip -c ' + filename, 'r').read()
                    break
        if source is None:
            print """
You need to set the GPAW_SETUP_PATH environment variable to point to
the directory where the basis set files are stored.  See

  http://wiki.fysik.dtu.dk/gpaw/Setups

for details."""

            raise RuntimeError('Could not find "%s" basis for "%s".' %
                               (basis_name, symbol))

        #fingerprint = md5.new(source).hexdigest()
        self.data = None
        xml.sax.parseString(source, self)
        return self.l_j, self.rc, self.phit_jg, filename

    def startElement(self, name, attrs):
        if name == 'paw_basis':
            self.version = attrs['version']
        elif name == 'radial_grid':
            self.ng = int(attrs['n'])
            assert int(attrs['istart']) == 0
            assert int(attrs['iend']) == self.ng - 1
        elif name == 'basis_function':
            self.l_j.append(int(attrs['l']))
            self.rc = float(attrs['rc'])
            self.data = []

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        if self.data is None:
            return
        phit_g = num.array([float(x) for x in ''.join(self.data).split()])
        assert len(phit_g) == self.ng
        self.phit_jg.append(phit_g)
