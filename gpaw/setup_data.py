# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
import os
import xml.sax
import re
from cStringIO import StringIO
from math import sqrt

import numpy as npy
from ase.data import atomic_names

from gpaw.utilities.tools import md5
from gpaw import setup_paths

try:
    import gzip
except:
    has_gzip = False
else:
    has_gzip = True

class SetupData:
    """Container class for persistent setup attributes and XML I/O."""
    def __init__(self, symbol, xcsetupname, name='paw', zero_reference=False,
                 readxml=True):
        self.symbol = symbol
        self.setupname = xcsetupname
        self.name = name
        self.zero_reference = zero_reference
        self.softgauss = False

        if name is None or name == 'paw':
            self.stdfilename = '%s.%s' % (symbol, self.setupname)
        else:
            self.stdfilename = '%s.%s.%s' % (symbol, name, self.setupname)

        self.filename = None # full path
        self.fingerprint = None
        
        self.n_j = []
        self.l_j = []
        self.f_j = []
        self.eps_j = []
        self.rcut_j = []
        self.id_j = []
        self.phi_jg = []
        self.phit_jg = []
        self.pt_jg = []
        self.tauc_g = None
        self.tauct_g = None
        self.X_p = None
        self.ExxC = None
        self.phicorehole_g = None
        self.fcorehole = 0.0
        self.lcorehole = None
        self.ncorehole = None
        self.core_hole_e = None
        self.core_hole_e_kin = None
        self.extra_xc_data = {}

        self.Z = None
        self.Nc = None
        self.Nv = None
        self.beta = None
        self.ng = None
        self.rcgauss = None
        self.e_kinetic = None
        self.e_xc = None
        self.e_electrostatic = None
        self.e_total = None
        self.e_kinetic_core = None

        self.nc_g = None
        self.nct_g = None
        self.nvt_g = None
        self.vbar_g = None

        self.e_kin_jj = None

        self.generatorattrs = []
        self.generatordata = ''
        
        self.has_corehole = False

        if readxml:
            PAWXMLParser(self).parse()
            nj = len(self.l_j)
            self.e_kin_jj.shape = (nj, nj)

    def write_xml(self):
        l_j = self.l_j
        xml = open(self.stdfilename, 'w')

        print >> xml, '<?xml version="1.0"?>'
        print >> xml, '<paw_setup version="0.6">'
        name = atomic_names[self.Z].title()
        comment1 = name + ' setup for the Projector Augmented Wave method.'
        comment2 = 'Units: Hartree and Bohr radii.'
        comment2 += ' ' * (len(comment1) - len(comment2))
        print >> xml, '  <!--', comment1, '-->'
        print >> xml, '  <!--', comment2, '-->'

        print >> xml, ('  <atom symbol="%s" Z="%d" core="%.1f" valence="%d"/>'
                       % (self.symbol, self.Z, self.Nc, self.Nv))
        if self.setupname == 'LDA':
            type = 'LDA'
            name = 'PW'
        else:
            type = 'GGA'
            name = self.setupname
        print >> xml, '  <xc_functional type="%s" name="%s"/>' % (type, name)
        gen_attrs = ' '.join(['%s="%s"' % (key, value) for key, value 
                              in self.generatorattrs])
        print >> xml, '  <generator %s>' % gen_attrs
        print >> xml, '    %s' % self.generatordata
        print >> xml, '  </generator>'
        print >> xml, '  <ae_energy kinetic="%f" xc="%f"' % \
              (self.e_kinetic, self.e_xc)
        print >> xml, '             electrostatic="%f" total="%f"/>' % \
              (self.e_electrostatic, self.e_total)

        print >> xml, '  <core_energy kinetic="%f"/>' % self.e_kinetic_core
        print >> xml, '  <valence_states>'
        line1 = '    <state n="%d" l="%d" f=%s rc="%5.3f" e="%8.5f" id="%s"/>'
        line2 = '    <state       l="%d"        rc="%5.3f" e="%8.5f" id="%s"/>'

        for id, l, n, f, e, rc in zip(self.id_j, l_j, self.n_j, self.f_j,
                                      self.eps_j, self.rcut_j):
            if n > 0:
                f = '%-4s' % ('"%d"' % f)
                print >> xml, line1 % (n, l, f, rc, e, id)
            else:
                print >> xml, line2 % (l, rc, e, id)
        print >> xml, '  </valence_states>'

        print >> xml, ('  <radial_grid eq="r=a*i/(n-i)" a="%f" n="%d" ' +
                       'istart="0" iend="%d" id="g1"/>') % \
                       (self.beta, self.ng, self.ng - 1)

        print >> xml, ('  <shape_function type="gauss" rc="%.12e"/>' %
                       self.rcgauss)

        if self.has_corehole:
            print >> xml, (('  <core_hole_state state="%d%s" ' +
                           'removed="%.1f" eig="%.8f" ekin="%.8f">') %
                           (self.ncorehole, 'spd'[self.lcorehole],
                            self.fcorehole,
                            self.core_hole_e, self.core_hole_e_kin))
            for x in self.phicorehole_g:
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </core_hole_state>'

        for name, a in [('ae_core_density', self.nc_g),
                        ('pseudo_core_density', self.nct_g),
                        ('pseudo_valence_density', self.nvt_g),
                        ('zero_potential', self.vbar_g),
                        ('ae_core_kinetic_energy_density', self.tauc_g),
                        ('pseudo_core_kinetic_energy_density', self.tauct_g)]:
            print >> xml, '  <%s grid="g1">\n    ' % name,
            for x in a:
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </%s>' % name

        # Print xc-specific data to setup file (used so for KLI and GLLB)
        for name, a in self.extra_xc_data.iteritems():
            newname = 'GLLB_'+name
            print >> xml, '  <%s grid="g1">\n    ' % newname,
            for x in a:
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </%s>' % newname

        for id, l, u, s, q, in zip(self.id_j, l_j, self.phi_jg, self.phit_jg,
                                   self.pt_jg):
            for name, a in [('ae_partial_wave', u),
                            ('pseudo_partial_wave', s),
                            ('projector_function', q)]:
                print >> xml, ('  <%s state="%s" grid="g1">\n    ' %
                               (name, id)),
                #p = a.copy()
                #p[1:] /= r[1:]
                #if l == 0:
                #    # XXXXX go to higher order!!!!!
                #    p[0] = (p[2] +
                #            (p[1] - p[2]) * (r[0] - r[2]) / (r[1] - r[2]))
                for x in a:
                    print >> xml, '%16.12e' % x,
                print >> xml, '\n  </%s>' % name

        print >> xml, '  <kinetic_energy_differences>',
        nj = len(self.e_kin_jj)
        for j1 in range(nj):
            print >> xml, '\n    ',
            for j2 in range(nj):
                print >> xml, '%16.12e' % self.e_kin_jj[j1, j2],
        print >> xml, '\n  </kinetic_energy_differences>'

        if self.X_p is not None:
            print >> xml, '  <exact_exchange_X_matrix>\n    ',
            for x in self.X_p:
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </exact_exchange_X_matrix>'

            print >> xml, '  <exact_exchange core-core="%f"/>' % self.ExxC

        print >> xml, '</paw_setup>'


def search_for_file(name):
    """Traverse gpaw setup paths to find file.

    Returns the file path and file contents.  If the file is not found,
    contents will be None."""
    source = None
    filename = None
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

    return filename, source


class PAWXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self, setup):
        xml.sax.handler.ContentHandler.__init__(self)
        self.setup = setup
        self.id = None
        self.data = None

    def parse(self):
        setup = self.setup

        (setup.filename, source) = search_for_file(setup.stdfilename)

        if source is None:
            print """
You need to set the GPAW_SETUP_PATH environment variable to point to
the directory where the setup files are stored.  See
http://wiki.fysik.dtu.dk/gpaw/install/installationguide.html for details."""
            raise RuntimeError('Could not find %s-setup for "%s".' %
                               (setup.name + '.' + setup.setupname, 
                                setup.symbol))

        setup.fingerprint = md5.new(source).hexdigest()

        # XXXX There must be a better way!
        # We don't want to look at the dtd now.  Remove it:
        source = re.compile(r'<!DOCTYPE .*?>', re.DOTALL).sub('', source, 1)
        xml.sax.parse(StringIO(source), self) # XXX There is a special parse
                                              # function that takes a string
        if setup.zero_reference:
            setup.e_total = 0.0
            setup.e_kinetic = 0.0
            setup.e_electrostatic = 0.0
            setup.e_xc = 0.0

        if not hasattr(setup, 'tauc_g'):
            setup.tauc_g = setup.tauct_g = None


    def startElement(self, name, attrs):
        setup = self.setup
        if name == 'paw_setup':
            setup.version = attrs['version']
            assert setup.version >= '0.4'
        if name == 'atom':
            setup.Z = int(attrs['Z'])
            setup.Nc = float(attrs['core'])
            setup.Nv = int(attrs['valence'])
        elif name == 'xc_functional':
            if attrs['type'] == 'LDA':
                setup.xcname = 'LDA'
            else:
                assert attrs['type'] == 'GGA'
                setup.xcname = attrs['name']
        elif name == 'ae_energy':
            setup.e_total = float(attrs['total'])
            setup.e_kinetic = float(attrs['kinetic'])
            setup.e_electrostatic = float(attrs['electrostatic'])
            setup.e_xc = float(attrs['xc'])
        elif name == 'core_energy':
            setup.e_kinetic_core = float(attrs['kinetic'])
        elif name == 'state':
            setup.n_j.append(int(attrs.get('n', -1)))
            setup.l_j.append(int(attrs['l']))
            setup.f_j.append(int(attrs.get('f', 0)))
            setup.eps_j.append(float(attrs['e']))
            setup.rcut_j.append(float(attrs.get('rc', -1)))
            setup.id_j.append(attrs['id'])
            # Compatibility with old setups:
            if setup.version < '0.6' and setup.f_j[-1] == 0:
                setup.n_j[-1] = -1
        elif name in ['grid', 'radial_grid']:  # XXX
            assert attrs['eq'] == 'r=a*i/(n-i)'
            setup.ng = int(attrs['n'])
            setup.beta = float(attrs['a'])
        elif name == 'shape_function':
            if attrs.has_key('rc'):
                assert attrs['type'] == 'gauss'
                setup.rcgauss = float(attrs['rc'])
            else:
                # Old style: XXX
                setup.rcgauss = max(setup.rcut_j) / sqrt(float(attrs['alpha']))
        elif name in ['ae_core_density', 'pseudo_core_density',
                      'localized_potential', 'zero_potential',  # XXX
                      'kinetic_energy_differences', 'exact_exchange_X_matrix',
                      'ae_core_kinetic_energy_density',
                      'pseudo_core_kinetic_energy_density']:
            self.data = []
        elif name.startswith('GLLB_'):
            self.data = []
        elif name in ['ae_partial_wave', 'pseudo_partial_wave']:
            self.data = []
            self.id = attrs['state']
        elif name == 'projector_function':
            self.id = attrs['state']
            self.data = []
        elif name == 'exact_exchange':
            setup.ExxC = float(attrs['core-core'])
        elif name == 'core_hole_state':
            setup.has_corehole = True
            setup.fcorehole = float(attrs['removed'])
            setup.core_hole_e = float(attrs['eig'])
            setup.core_hole_e_kin = float(attrs['ekin'])
            self.data = []
        else:
            self.data = None

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        setup = self.setup
        if self.data is None:
            return
        x_g = npy.array([float(x) for x in ''.join(self.data).split()])
        if name == 'ae_core_density':
            setup.nc_g = x_g
        elif name == 'pseudo_core_density':
            setup.nct_g = x_g
        elif name == 'kinetic_energy_differences':
            setup.e_kin_jj = x_g
        elif name == 'ae_core_kinetic_energy_density':
            setup.tauc_g = x_g
        elif name == 'pseudo_valence_density':
            setup.nvt_g = x_g
        elif name == 'pseudo_core_kinetic_energy_density':
            setup.tauct_g = x_g
        elif name in ['localized_potential', 'zero_potential']: # XXX
            setup.vbar_g = x_g
        elif name.startswith('GLLB_'):
            # Add setup tags starting with GLLB_ to extra_xc_data. Remove GLLB_ from front of string.
            setup.extra_xc_data[name[5:]] = x_g
        elif name == 'ae_partial_wave':
            j = len(setup.phi_jg)
            assert self.id == setup.id_j[j]
            setup.phi_jg.append(x_g)
        elif name == 'pseudo_partial_wave':
            j = len(setup.phit_jg)
            assert self.id == setup.id_j[j]
            setup.phit_jg.append(x_g)
        elif name == 'projector_function':
            j = len(setup.pt_jg)
            assert self.id == setup.id_j[j]
            setup.pt_jg.append(x_g)
        elif name == 'exact_exchange_X_matrix':
            setup.X_p = x_g
        elif name == 'core_hole_state':
            setup.phicorehole_g = x_g
