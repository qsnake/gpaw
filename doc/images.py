#!/usr/bin/env python

""" TODO:

We should find a good way in which to store files elsewhere than _static

Make sure that downloaded files are copied to build dir on build
This must (probably) be done *after* compilation because otherwise dirs
may not exist.

"""

from urllib2 import urlopen, HTTPError
import os
from sys import executable

srcpath = 'http://wiki.fysik.dtu.dk/gpaw-files'

def get(path, names, target=None, source=None):
    """Get files from web-server.

    Returns True if something new was fetched."""
    
    if target is None:
        target = path
    if source is None:
        source = srcpath
    got_something = False
    for name in names:
        src = os.path.join(source, path, name)
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


literature = """
askhl_10302_report.pdf  mortensen_gpaw-dev.pdf      rostgaard_master.pdf
askhl_master.pdf        mortensen_mini2003talk.pdf  rostgaard_paw_notes.pdf
marco_master.pdf        mortensen_paw.pdf
""".split()

logos = """
logo-csc.png  logo-fmf.png   logo-hut.png  logo-tree.png
logo-dtu.png  logo-gpaw.png  logo-jyu.png  logo-tut.png  logo-anl.png
""".split()


# flowchart.pdf  flowchart.sxd <-- where?
devel_stuff = """
gpaw-logo.svg gpaw-logo.odg overview.odg overview.pdf svn-refcard.pdf
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

get('logos', logos, '_static')
get('architectures', architectures_stuff, '_static')
get('doc/literature', literature, '_static')
get('doc/devel', devel_stuff, '_static')
get('devel', ['bslogo.png', 'overview.png', 'stat.png'])

# Note: bz-all.png is used both in an exercise and a tutorial.  Therefore
# we put it in the common dir so far, rather than any of the two places
get('.', ['bz-all.png'], '_static')
get('exercises/band_structure', ['silicon_banddiagram.png'])
get('exercises/wavefunctions', ['co_bonding.jpg'])

get('tutorials/bandstructures', ['sodium_bands.png'])
get('tutorials/ensembles', ['ensemble.png'])

get('.', ['2sigma.png', 'co_wavefunctions.png', 'molecules.png'], '_static')
get('exercises/lrtddft', ['spectrum.png'])
get('documentation/xc', 'g2test_pbe0.png  g2test_pbe.png  results.png'.split())
get('performance', 'dacapoperf.png  goldwire.png  gridperf.png'.split(),
    '_static')
get('documentation/xc', ['phi.dat', 'makephi.tar.gz'], '_static')

# Generate one page for each setup:
if get('setups', ['setups-data.tar.gz'], '_static'):
    print 'Extracting setup data ...'
    os.system('tar --directory=_static -xzf _static/setups-data.tar.gz')
    print 'Generating setup pages ...'
    os.system('cd setups; %s make_setup_pages.py' % executable)

get('tutorials/lattice_constants', ['Fe_conv_k.png', 'Fe_conv_h.png'])
get('tutorials/negfstm', ['fullscan.png', 'linescan.png'])
get('tutorials/xas', ['h2o_xas_3.png', 'h2o_xas_4.png'])

# Retrieve latest code coverage pages:
if get('.', ['gpaw-coverage-latest.tar.gz'], '_static',
       source='http://dcwww.camp.dtu.dk/~chlg'):
    print 'Extracting coverage pages ...'
    os.system('tar --directory=devel -xzf _static/gpaw-coverage-latest.tar.gz')


jjwww = 'http://dcwww.camp.dtu.dk/~jensj'

# Get png files and othe stuff from the AGTS scripts that run every weekend:
from gpaw.test.big.agts import AGTSQueue
queue = AGTSQueue()
queue.collect()
names = set()
for job in queue.jobs:
    if job.creates:
        for name in job.creates:
            assert name not in names, 'Name clash!'
            names.add(name)
            if job.dir.startswith('doc'):
                dir = doc.dir
            else:
                # This file was created by a test from the
                # gpaw/test/big folder:
                dir = '_static'
            get('gpaw-files', [name], dir, source=jjwww)

get('gpaw-stuff', ['xas_illustration.png'], 'tutorials/xas', jjwww)
