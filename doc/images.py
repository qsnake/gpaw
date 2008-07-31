#!/usr/bin/env python

"""
Note:

Images can be put anywhere and will be copied to the equivalent
position in the html dir if they are referenced somewhere.  The images
in the html dir will not be linked correctly (unless the referincing
.rst file is in the base source dir), however, which appears to be a
bug in sphinx.  If the reference in the .rst file is changed to
reflect the *actual* location to which the images are copied, sphinx
doesn't think the ".. image:" directives point to the images anymore,
and will not copy those images during build.

Therefore we must put everything in _static, which is horrible.  Egads!

Except for .rst files which are in the source dir directly.  For those
it actually works.  One way to actually make it work is to make sure
that every image is linked from an unreachable page in the source dir,
but that is just blasphemous.  """

from urllib2 import urlopen, HTTPError
import os

srcpath = 'http://web2.fysik.dtu.dk//gpaw-files'

def get(path, names, target='_static'):#, target=None):
    """Get files from web-server.

    Returns True if something new was fetched."""
    
    if target is None:
        target = path
    got_something = False
    for name in names:
        src = os.path.join(srcpath, path, name)
        dst = os.path.join(target, name)

        if not os.path.isfile(dst):
            print dst,
            try:
                data = urlopen(src).read()
                sink = open(dst, 'w')
                sink.write(data)
                sink.close()
                print 'OK'
                got_something = True                
            except HTTPError:
                print 'HTTP Error!'
    return got_something

get('test', ['test.png'])


literature = """
askhl_10302_report.pdf  mortensen_gpaw-dev.pdf      rostgaard_master.pdf
askhl_master.pdf        mortensen_mini2003talk.pdf  rostgaard_paw_notes.pdf
marco_master.pdf        mortensen_paw.pdf
""".split()

logos = """
logo-csc.png  logo-fmf.png   logo-hut.png  logo-tree.png
logo-dtu.png  logo-gpaw.png  logo-jyu.png  logo-tut.png
""".split()


# flowchart.pdf  flowchart.sxd <-- where?
devel_stuff = """
gpaw-logo.odg  overview.odg overview.pdf
""".split()

architectures_stuff = """
dynload_redstorm.c
numpy-1.0.4-gnu.py.patch
numpy-1.0.4-gnu.py.patch.powerpc-bgp-linux-gfortran
numpy-1.0.4-site.cfg.lapack_bgp_esslbg
numpy-1.0.4-system_info.py.patch.lapack_bgp_esslbg
setup
unixccompiler.py
""".split()

get('logos', logos)
get('architectures', architectures_stuff)
get('doc/literature', literature)
get('doc/devel', devel_stuff, '_static')
get('devel', ['bslogo.png', 'overview.png', 'stat.png'])
get('exercises', ['silicon_banddiagram.png', 'co_bonding.jpg',
                  'NEB_Al-Al110.traj'])
get('tutorials', ['ensemble.png', 'sodium_bands.png', 'bz-all.png',
                  'gridrefinement.png', 'ae_density_H2O.png',
                  'ae_density_NaCl.png'])

get('.', ['2sigma.png', 'co_wavefunctions.png', 'molecules.png'])
get('tddft', ['spectrum.png'])
get('exx', 'g2test_pbe0.png  g2test_pbe.png  results.png'.split())
get('xas', ['xas_32H2O.png', 'xas.png', 'xas_exp.png', 'xas_H2O.png'])
get('performance', 'dacapoperf.png  goldwire.png  gridperf.png'.split())
get('vdw', ['phi.dat', 'makephi.tar.gz'])

# Generate one page for each setup:
if get('setups', ['setup-images.tar.gz', 'setup-data.pckl']):
    print 'Extracting setup images ...'
    os.system('tar --directory=_static -xzf _static/setup-images.tar.gz')
    print 'Generating setup pages ...'
    os.system('cd setups; python make_setup_pages.py')
