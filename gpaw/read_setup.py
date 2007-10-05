# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import xml.sax
import md5
import re
from cStringIO import StringIO

import Numeric as num

from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

try:
    import gzip
except:
    has_gzip = False
else:
    has_gzip = True

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
        self.core_hole_state = None
        self.fcorehole = 0.0
        self.core_hole_e = None
        self.core_hole_e_kin = None
        self.core_response = []

    def parse(self, symbol, xcfunc):
        exx = False
        xcname = xcfunc.get_name()
        if xcfunc.hybrid > 0:
            if xcname == 'EXX': # XXX EXX hack
                exx = True
                xcname = XCFunctional('LDA').get_name()
            elif xcname == XCFunctional('oldPBE0').get_name(): # XXX EXX hack
                exx = True
                xcname = XCFunctional('oldPBE').get_name()
            elif xcname == XCFunctional('PBE0').get_name(): # XXX EXX hack
                exx = True
                xcname = XCFunctional('PBE').get_name()

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
                    if has_gzip:
                        source = gzip.open(filename).read()
                    else:
                        source = os.popen('gunzip -c ' + filename, 'r').read()
                    break
        if source is None:
            print """
You need to set the GPAW_SETUP_PATH environment variable to point to
the directory where the setup files are stored.  See
http://wiki.fysik.dtu.dk/gpaw/Setups for details."""

            raise RuntimeError('Could not find %s-setup for "%s".' %
                               (xcname, symbol))

        fingerprint = md5.new(source).hexdigest()

        # XXXX There must be a better way!
        # We don't want to look at the dtd now.  Remove it:
        source = re.compile(r'<!DOCTYPE .*?>', re.DOTALL).sub('', source, 1)
        xml.sax.parse(StringIO(source), self) # XXX There is a special parse
                                              # function that takes a string

        if exx:
            self.e_total = 0.0
            self.e_kinetic = 0.0
            self.e_electrostatic = 0.0
            self.e_xc = 0.0

        if not hasattr(self, 'tauc_g'):
            self.tauc_g = self.tauct_g = None

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
                self.tauc_g,
                self.tauct_g,
                fingerprint,
                filename,
                self.core_hole_state,
                self.fcorehole,
                self.core_hole_e,
                self.core_hole_e_kin,
                self.core_response)

    def startElement(self, name, attrs):
        if name == 'paw_setup':
            self.version = attrs['version']
            assert self.version >= '0.4'
        if name == 'atom':
            self.Z = int(attrs['Z'])
            self.Nc = float(attrs['core'])
            self.Nv = int(attrs['valence'])
        elif name == 'xc_functional':
            if attrs['type'] == 'LDA':
                self.xcname = 'LDA'
            else:
                assert attrs['type'] == 'GGA'
                self.xcname = attrs['name']
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
                      'kinetic_energy_differences', 'exact_exchange_X_matrix',
                      'ae_core_kinetic_energy_density',
                      'pseudo_core_kinetic_energy_density', 'core_response']:
            self.data = []
        elif name in ['ae_partial_wave', 'pseudo_partial_wave']:
            self.data = []
            self.id = attrs['state']
        elif name == 'projector_function':
            self.id = attrs['state']
            self.data = []
        elif name == 'exact_exchange':
            self.ExxC = float(attrs['core-core'])
        elif name == 'core_hole_state':
            self.fcorehole = float(attrs['removed'])
            self.core_hole_e = float(attrs['eig'])
            self.core_hole_e_kin = float(attrs['ekin'])
            self.data = []
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
        elif name == 'ae_core_kinetic_energy_density':
            self.tauc_g = x_g
        elif name == 'pseudo_core_kinetic_energy_density':
            self.tauct_g = x_g
        elif name in ['localized_potential', 'zero_potential']: # XXX
            self.vbar_g = x_g
        elif name == 'core_response':
            print "core_response", x_g
            self.core_response = x_g
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
        elif name == 'core_hole_state':
            self.core_hole_state = x_g
