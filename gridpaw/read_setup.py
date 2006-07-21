# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import xml.sax
import md5
import re
from cStringIO import StringIO

import Numeric as num

from gridpaw import setup_paths


class PAWXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)

        self.n_j = []
        self.l_j = []
        self.f_j = []
        self.eps_j = []
        self.rcut_j = []
        self.id_j = []
        self.phi_jg = []
        self.phit_jg = []
        self.pt_jg = []
        self.X_p = []
        self.ExxC = None

    def parse(self, symbol, xcname):
        if xcname == 'EXX': # XXX EXX hack 
            xcname = 'LDA'
        name = symbol + '.' + xcname
        source = None
        for path in setup_paths:
            filename = os.path.join(path, name)
            if os.path.isfile(filename):
                source = open(filename).read()
                break
            else:
                filename += '.gz'
                if os.path.isfile(filename):
                    source = os.popen('gunzip -c ' + filename, 'r').read()
##                  source = gzip.open(filename).read() ibm has no zlib! XXX
                    break
        if source is None:
            raise RuntimeError('Could not find %s-setup for %s.' %
                               (xcname, symbol))

        fingerprint = md5.new(source).hexdigest()

        # XXXX There must be a better way!
        # We don't want to look at the dtd now.  Remove it:
        source = re.compile(r'<!DOCTYPE .*?>', re.DOTALL).sub('', source, 1)
        xml.sax.parse(StringIO(source), self) # XXX There is a special parse
                                              # function that takes a string 
        return (self.Z, self.Nc, self.Nv,
                self.e_total,
                self.e_kinetic,
                self.e_electrostatic,
                self.e_xc,
                self.e_kinetic_core,
                self.n_j,
                self.l_j,
                self.f_j,
                self.eps_j,
                self.rcut_j,
                self.id_j,
                self.ng,
                self.beta,
                self.nc_g,
                self.nct_g,
                self.vbar_g,
                self.rcgauss,
                self.phi_jg,  
                self.phit_jg,
                self.pt_jg,
                self.e_kin_j1j2,
                self.X_p,
                self.ExxC,
                fingerprint,
                filename)
    
    def startElement(self, name, attrs):
        if name == 'paw_setup':
            self.version = attrs['version']
            assert self.version >= '0.4'
        if name == 'atom':
            self.Z = int(attrs['Z'])
            self.Nc = int(attrs['core'])
            self.Nv = int(attrs['valence'])
        elif name == 'xc_functional':
            if attrs['type'] == 'LDA':
                self.xcname = 'LDA'
            else:
                assert attrs['type'] == 'GGA'
                self.xcname = attrs['name']
        elif name == 'generator':
            assert attrs['type'] != 'non-relativistic'
        elif name == 'ae_energy':
            self.e_total = float(attrs['total'])
            self.e_kinetic = float(attrs['kinetic'])
            self.e_electrostatic = float(attrs['electrostatic'])
            self.e_xc = float(attrs['xc'])
        elif name == 'core_energy':
            self.e_kinetic_core = float(attrs['kinetic'])
        elif name == 'state':
            self.n_j.append(int(attrs.get('n', -1)))
            self.l_j.append(int(attrs['l']))
            self.f_j.append(int(attrs.get('f', 0)))
            self.eps_j.append(float(attrs['e']))
            self.rcut_j.append(float(attrs.get('rc', -1)))
            self.id_j.append(attrs['id'])
            # Compatibility with old setups:
            if self.version < '0.6' and self.f_j[-1] == 0:
                self.n_j[-1] = -1
        elif name in ['grid', 'radial_grid']:  # XXX
            assert attrs['eq'] == 'r=a*i/(n-i)'
            self.ng = int(attrs['n'])
            self.beta = float(attrs['a'])
        elif name == 'shape_function':
            if attrs.has_key('rc'):
                assert attrs['type'] == 'gauss'
                self.rcgauss = float(attrs['rc'])
            else:
                # Old style: XXX
                from math import sqrt
                self.rcgauss = max(self.rcut_j) / sqrt(float(attrs['alpha']))
        elif name in ['ae_core_density', 'pseudo_core_density',
                      'localized_potential', 'zero_potential',  # XXX
                      'kinetic_energy_differences', 'exact_exchange_X_matrix']:
            self.data = []
        elif name in ['ae_partial_wave', 'pseudo_partial_wave']:
            self.data = []
            self.id = attrs['state']
        elif name == 'projector_function':
            self.id = attrs['state']
            self.data = []
        elif name == 'exact_exchange':
            self.ExxC = float(attrs['core-core'])
        else:
            self.data = None
            
    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        if self.data is None:
            return
        x_g = num.array([float(x) for x in ''.join(self.data).split()])
        if name == 'ae_core_density':
            self.nc_g = x_g
        elif name == 'pseudo_core_density':
            self.nct_g = x_g
        elif name == 'kinetic_energy_differences':
            self.e_kin_j1j2 = x_g
        elif name in ['localized_potential', 'zero_potential']: # XXX
            self.vbar_g = x_g
        elif name == 'ae_partial_wave':
            j = len(self.phi_jg)
            assert self.id == self.id_j[j]
            self.phi_jg.append(x_g)
        elif name == 'pseudo_partial_wave':
            j = len(self.phit_jg)
            assert self.id == self.id_j[j]
            self.phit_jg.append(x_g)
        elif name == 'projector_function':
            j = len(self.pt_jg)
            assert self.id == self.id_j[j]
            self.pt_jg.append(x_g)
        elif name == 'exact_exchange_X_matrix':
            self.X_p = x_g
