#!/usr/bin/python
import os
import sys
import time
import glob
import trace
import tempfile

tmpdir = tempfile.mkdtemp(prefix='gpaw-sphinx-')
os.chdir(tmpdir)

def build():
    if os.system('svn export ' +
                 'https://svn.fysik.dtu.dk/projects/ase/trunk ase') != 0:
        raise RuntimeError('Checkout of ASE failed!')
    os.chdir('ase')
    if os.system('python setup.py install --home=..') != 0:
        raise RuntimeError('Installation of ASE failed!')
    os.chdir('..')
    if os.system('svn checkout ' +
                 'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
        raise RuntimeError('Checkout of GPAW failed!')
    os.chdir('gpaw')
    if os.system('python setup.py install --home=.. 2> error') != 0:
        raise RuntimeError('Installation of GPAW failed!')

    os.system('grep -v "c/libxc/src" error 1>&2')

    os.system('wget --no-check-certificate --quiet ' +
              'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')

    os.system('tar xvzf gpaw-setups-latest.tar.gz')

    setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]

    # Generate tar-file:
    assert os.system('python setup.py sdist') == 0

    if os.system('epydoc --docformat restructuredtext --parse-only ' +
                 '--name GPAW ' +
                 '--url http://wiki.fysik.dtu.dk/gpaw ' +
                 '--show-imports --no-frames -v gpaw >& epydoc.out') != 0:
        raise RuntimeError('Epydoc failed!')

    epydoc_errors = open('epydoc.out').read()
    if ' Warning:' in epydoc_errors:
        sys.stderr.write(epydoc_errors)

    sys.path.insert(0, tmpdir + '/lib/python')
    from gpaw.version import version

    os.chdir('doc')
    os.system('sed -i s/gpaw-snapshot/gpaw-%s/ download.rst' % version)
    os.mkdir('_build')
    if os.system('PYTHONPATH=%s/lib/python ' % tmpdir +
                 'GPAW_SETUP_PATH=%s ' % setups +
                 'sphinx-build . _build') != 0:
        raise RuntimeError('Sphinx failed!')
    os.system('cd _build; cp _static/searchtools.js .')

    if 0:
        if os.system('sphinx-build -b latex . _build') != 0:
            raise RuntimeError('Sphinx failed!')
        os.chdir('_build')
        #os.system('cd ../..; ln -s doc/_static')
        if os.system('make gpaw-manual.pdf') != 0:
            raise RuntimeError('pdflatex failed!')
    else:
        os.chdir('_build')

    assert os.system('mv ../../html epydoc;' +
                     'mv ../../dist/gpaw-%s.tar.gz .' % version) == 0

tarfiledir = None
if len(sys.argv) == 2:
    tarfiledir = sys.argv[1]
    try:
        os.remove(tarfiledir + '/gpaw-webpages.tar.gz')
    except OSError:
        pass

build()
    
if tarfiledir is not None:
    os.system('cd ..; tar czf %s/gpaw-webpages.tar.gz _build' % tarfiledir)
