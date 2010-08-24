#!/usr/bin/python

import os
import sys
import time
import glob
import trace
import tempfile

def send_email(subject, filename='/dev/null'):
    assert os.system(
        'mail -s "%s" gpaw-developers@listserv.fysik.dtu.dk < %s' %
        (subject, filename)) == 0

def fail(msg, filename='/dev/null'):
    send_email(msg, filename)
    raise SystemExit

if '--dir' in sys.argv:
    i = sys.argv.index('--dir')
    sys.argv.pop(i)
    dir = sys.argv.pop(i)
else:
    dir = None

tmpdir = tempfile.mkdtemp(prefix='gpaw-', dir=dir)
os.chdir(tmpdir)

day = time.localtime()[6]

# Checkout a fresh version and install:
if os.system('svn checkout ' +
             'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
    fail('Checkout of gpaw failed!')

if day % 2:
    exec([line for line in open('gpaw/gpaw/version.py').readlines()
          if line.startswith('ase_required_svnversion')][0])
else:
    ase_required_svnversion = 'HEAD'

if os.system('svn checkout ' +
             'https://svn.fysik.dtu.dk/projects/ase/trunk ase -r %s' %
             ase_required_svnversion) != 0:
    fail('Checkout of ASE failed!')
try: 
    # subprocess was introduced with python 2.4
    from subprocess import Popen, PIPE
    cmd = Popen('svnversion ase',
                shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).stdout
except ImportError:
    cmd = popen3('svnversion ase')[1] # assert that we are in gpaw project
aserevision = int(cmd.readline())
cmd.close()

os.chdir('gpaw')

try: 
    # subprocess was introduced with python 2.4
    from subprocess import Popen, PIPE
    cmd = Popen('svnversion', 
                shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).stdout
except ImportError:
    cmd = popen3('svnversion')[1] # assert that we are in gpaw project
gpawrevision = int(cmd.readline().strip('M\n'))
cmd.close()

if os.system('python setup.py install --home=%s ' % tmpdir +
             '2>&1 | grep -v "c/libxc/src"') != 0:
    fail('Installation failed!')

os.system('mv ../ase/ase ../lib/python')

os.system('wget --no-check-certificate --quiet ' +
          'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')

os.system('tar xvzf gpaw-setups-latest.tar.gz')

setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]
sys.path.insert(0, '%s/lib/python' % tmpdir)

if day % 4 < 2:
    sys.argv.append('--debug')

from gpaw import setup_paths
setup_paths.insert(0, setups)

# Run test-suite:
from gpaw.test import TestRunner, tests
os.mkdir('gpaw-test')
os.chdir('gpaw-test')
out = open('test.out', 'w')
#tests = ['ase3k.py', 'jstm.py']
failed = TestRunner(tests, stream=out).run()
out.close()
if failed:
    # Send mail:
    n = len(failed)
    if n == 1:
        subject = 'One failed test: ' + failed[0]
    else:
        subject = ('%d failed tests: %s, %s' %
                   (n, failed[0], failed[1]))
        if n > 2:
            subject += ', ...'
    fail(subject, 'test.out')

open('/home/camp/jensj/gpawrevision.ok', 'w').write('%d %d\n' %
                                                    (aserevision,
                                                     gpawrevision))

def count(dir, pattern):
    p = os.popen('wc -l `find %s -name %s` | tail -1' % (dir, pattern), 'r')
    return int(p.read().split()[0])

os.chdir('..')
libxc = count('c/libxc', '\\*.[ch]')
ch = count('c', '\\*.[ch]') - libxc
test = count('gpaw/test', '\\*.py')
py = count('gpaw', '\\*.py') - test

import pylab
# Update the stat.dat file:
dir = '/scratch/jensj/nightly-test/'
f = open(dir + 'stat.dat', 'a')
print >> f, pylab.epoch2num(time.time()), libxc, ch, py, test
f.close()

# Construct the stat.png file:
lines = open(dir + 'stat.dat').readlines()
date, libxc, c, code, test = zip(*[[float(x) for x in line.split()]
                                   for line in lines[1:]])
date = pylab.array(date)
code = pylab.array(code)
test = pylab.array(test)
c = pylab.array(c)

def polygon(x, y1, y2, *args, **kwargs):
    x = pylab.concatenate((x, x[::-1]))
    y = pylab.concatenate((y1, y2[::-1]))
    pylab.fill(x, y, *args, **kwargs)

fig = pylab.figure()
ax = fig.add_subplot(111)
polygon(date, code + test, code + test + c,
        facecolor='r', label='C-code')
polygon(date, code, code + test,
        facecolor='y', label='Tests')
polygon(date, [0] * len(date), code,
        facecolor='g', label='Python-code')
polygon(date, [0] * len(date), [0] * len(date),
        facecolor='b', label='Fortran-code')

months = pylab.MonthLocator()
months3 = pylab.MonthLocator(interval=3)
month_year_fmt = pylab.DateFormatter("%b '%y")

ax.xaxis.set_major_locator(months3)
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_formatter(month_year_fmt)
labels = ax.get_xticklabels()
pylab.setp(labels, rotation=30)
pylab.axis('tight')
pylab.legend(loc='upper left')
pylab.title('Number of lines')
pylab.savefig(dir + 'stat.png')

os.system('cd; rm -r ' + tmpdir)
