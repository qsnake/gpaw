#!/usr/bin/python
import os
import time
import glob
import tempfile

def fail(subject, filename='/dev/null'):
    assert os.system('mail -s "%s" jensj@fysik.dtu.dk < %s' %
                     (subject, filename)) == 0
    raise SystemExit

tmpdir = tempfile.mkdtemp(prefix='gpaw-parallel-')
os.chdir(tmpdir)

# Checkout a fresh version and install:
if os.system('svn export ' +
             'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
    fail('Checkout of gpaw failed!')
if os.system('svn export ' +
             'https://svn.fysik.dtu.dk/projects/ase/trunk ase') != 0:
    fail('Checkout of ASE failed!')

os.chdir('gpaw')
if os.system('source /home/camp/modulefiles.sh&& ' +
             'module load NUMPY&& ' +
             'python setup.py --remove-default-flags ' +
             '--customize=doc/install/Linux/Niflheim/customize-thul-acml.py ' +
             'install --home=%s 2>&1 | ' % tmpdir +
             'grep -v "c/libxc/src"') != 0:
    fail('Installation failed!')

os.system('mv ../ase/ase ../lib64/python')

os.system('wget --no-check-certificate --quiet ' +
          'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
os.system('tar xvzf gpaw-setups-latest.tar.gz')
setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]

day = time.localtime()[6]
cpus = 2 + 2 * (day % 2)
if day % 4 < 2:
    args = '--debug'
else:
    args = ''
    
# Run test-suite:
os.chdir('test')
if os.system('source /home/camp/modulefiles.sh; ' +
             'module load NUMPY; ' +
             'module load openmpi/1.3.3-1.el5.fys.gfortran43.4.3.2; ' +
             'export PYTHONPATH=%s/lib64/python:$PYTHONPATH; ' % tmpdir +
             'export GPAW_SETUP_PATH=%s; ' % setups +
             'mpiexec -np %d ' % cpus +
             tmpdir + '/bin/gpaw-python ' +
             'test.py -u %s >& test.out' % args) != 0:
    fail('Testsuite failed!')

try:
    failed = open('failed-tests.txt').readlines()
except IOError:
    pass
else:
    # Send mail:
    n = len(failed)
    if n == 1:
        subject = 'One failed test: ' + failed[0][:-1]
    else:
        subject = '%d failed tests: %s, %s' % (n,
                                               failed[0][:-1], failed[1][:-1])
        if n > 2:
            subject += ', ...'
    fail(subject, 'test.out')

#os.system('cd; rm -r ' + tmpdir)
