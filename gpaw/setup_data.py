# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
import os
import xml.sax
import re
from cStringIO import StringIO
from math import sqrt, pi

import numpy as npy
from ase.data import atomic_names
from ase.units import Bohr, Hartree

from gpaw import setup_paths
from gpaw.spline import Spline
from gpaw.utilities import fac, divrl
from gpaw.utilities.tools import md5_new
from gpaw.xc_functional import XCRadialGrid
from gpaw.xc_correction import XCCorrection

try:
    import gzip
except:
    has_gzip = False
else:
    has_gzip = True


class SetupData:
    """Container class for persistent setup attributes and XML I/O."""
    def __init__(self, symbol, xcsetupname, name='paw', readxml=True,
                 zero_reference=False):
        self.symbol = symbol
        self.setupname = xcsetupname
        self.name = name
        self.zero_reference = zero_reference

        # Default filename if this setup is written 
        if name is None or name == 'paw':
            self.stdfilename = '%s.%s' % (symbol, self.setupname)
        else:
            self.stdfilename = '%s.%s.%s' % (symbol, name, self.setupname)

        self.filename = None # full path if this setup was loaded from file
        self.fingerprint = None # hash value of file data if applicable

        self.Z = None
        self.Nc = None
        self.Nv = None

        # Quantum numbers, energies
        self.n_j = []
        self.l_j = []
        self.f_j = []
        self.eps_j = []
        self.e_kin_jj = None # <phi | T | phi> - <phit | T | phit>
        
        self.beta = None
        self.ng = None
        self.rcgauss = None # For compensation charge expansion functions
        
        # State identifier, like "X-2s" or "X-p1", where X is chemical symbol,
        # for bound and unbound states
        self.id_j = [] 
        
        # Partial waves, projectors
        self.phi_jg = []
        self.phit_jg = []
        self.pt_jg = []
        self.rcut_j = []
        
        # Densities, potentials
        self.nc_g = None
        self.nct_g = None
        self.nvt_g = None
        self.vbar_g = None
        
        # Kinetic energy densities of core electrons
        self.tauc_g = None
        self.tauct_g = None
        
        # Reference energies
        self.e_kinetic = 0.0
        self.e_xc = 0.0
        self.e_electrostatic = 0.0
        self.e_total = 0.0
        self.e_kinetic_core = 0.0
        
        # Generator may store description of setup in this string
        self.generatorattrs = []
        self.generatordata = ''
        
        # Optional quantities, normally not used
        self.X_p = None
        self.ExxC = None
        self.extra_xc_data = {}
        self.phicorehole_g = None
        self.fcorehole = 0.0
        self.lcorehole = None
        self.ncorehole = None
        self.core_hole_e = None
        self.core_hole_e_kin = None
        self.has_corehole = False        

        if readxml:
            PAWXMLParser(self).parse()
            nj = len(self.l_j)
            self.e_kin_jj.shape = (nj, nj)

    def is_compatible(self, xcfunc):
        return xcfunc.get_setup_name() == self.setupname

    def print_info(self, text, setup):
        if self.phicorehole_g is None:
            text(self.symbol + '-setup:')
        else:
            text('%s-setup (%.1f core hole):' % (self.symbol, self.fcorehole))
        text('  name   :', atomic_names[self.Z])
        text('  id     :', self.fingerprint)
        text('  Z      :', self.Z)
        text('  valence:', self.Nv)
        if self.phicorehole_g is None:
            text('  core   : %d' % self.Nc)
        else:
            text('  core   : %.1f' % self.Nc)
        text('  charge :', self.Z - self.Nv - self.Nc)
        text('  file   :', self.filename)
        text(('  cutoffs: %4.2f(comp), %4.2f(filt), %4.2f(core),'
              ' lmax=%d' % (sqrt(10) * self.rcgauss * Bohr,
                            # XXX is this really true?  I don't think this is
                            # actually the cutoff of the compensation charges
                            setup.rcutfilter * Bohr,
                            setup.rcore * Bohr,
                            setup.lmax)))
        text('  valence states:')
        text('            energy   radius')
        j = 0
        for n, l, f, eps in zip(self.n_j, self.l_j, self.f_j, self.eps_j):
            if n > 0:
                f = '(%d)' % f
                text('    %d%s%-4s %7.3f   %5.3f' % (
                    n, 'spdf'[l], f, eps * Hartree, self.rcut_j[j] * Bohr))
            else:
                text('    *%s     %7.3f   %5.3f' % (
                    'spdf'[l], eps * Hartree, self.rcut_j[j] * Bohr))
            j += 1
        text()

    def get_smooth_core_density_integral(self, Delta0):
        return -Delta0 * sqrt(4 * pi) - self.Z + self.Nc

    def get_overlap_correction(self, Delta0_ii):
        return sqrt(4.0 * pi) * Delta0_ii
    
    def get_linear_kinetic_correction(self, T0_qp):
        e_kin_jj = self.e_kin_jj
        nj = len(e_kin_jj)
        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(e_kin_jj[j1, j2])
        K_p = sqrt(4 * pi) * npy.dot(K_q, T0_qp)
        return K_p

    def get_ghat(self, lmax, alpha, r, rcut):
        d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
               for l in range(lmax + 1)]
        g = alpha**1.5 * npy.exp(-alpha * r**2)
        g[-1] = 0.0
        ghat_l = [Spline(l, rcut, d_l[l] * alpha**l * g)
                  for l in range(lmax + 1)]
        return ghat_l

    def find_core_density_cutoff(self, r_g, dr_g, nc_g):
        if self.Nc == 0:
            return 0.5
        else:
            N = 0.0
            g = self.ng - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * r_g[g]**2 * dr_g[g]
                g -= 1
            return r_g[g]

    def get_xc_correction(self, rgd, xcfunc, gcut2, lcut):
        xc = XCRadialGrid(xcfunc, rgd, xcfunc.nspins)
        phicorehole_g = self.phicorehole_g
        if phicorehole_g is not None:
            phicorehole_g = phicorehole_g[:gcut2].copy()

        xc_correction = XCCorrection(
            xc,
            [divrl(phi_g[:gcut2].copy(), l, rgd.r_g)
             for l, phi_g in zip(self.l_j, self.phi_jg)],
            [divrl(phit_g[:gcut2].copy(), l, rgd.r_g)
             for l, phit_g in zip(self.l_j, self.phit_jg)],
            self.nc_g[:gcut2].copy() / sqrt(4 * pi),
            self.nct_g[:gcut2].copy() / sqrt(4 * pi),
            rgd,
            list(enumerate(self.l_j)),
            #[(j, self.l_j[j]) for j in range(len(self.l_j))],
            min(2 * lcut, 4),
            self.e_xc,
            phicorehole_g,
            self.fcorehole,
            xcfunc.nspins,
            self.tauc_g[:gcut2].copy())
        return xc_correction

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

        setup.fingerprint = md5_new(source).hexdigest()

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

        #if not hasattr(setup, 'tauc_g'):
        #    setup.tauc_g = setup.tauct_g = None


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
