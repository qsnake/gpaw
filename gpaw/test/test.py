#!/usr/bin/env python
import os
import gc
import sys
import time
import tempfile
from optparse import OptionParser

from gpaw import mpi, hooks


parser = OptionParser(usage='%prog [options] [tests]',
                      version='%prog 0.1')

parser.add_option('-x', '--exclude',
                  type='string', default=None,
                  help='Exclude tests (comma separated list of tests).',
                  metavar='test1.py,test2.py,...')

parser.add_option('-f', '--run-failed-tests-only',
                  action='store_true',
                  help='Run failed tests only.')

parser.add_option('--from', metavar='TESTFILE', dest='from_test',
                  help='Run remaining tests, starting from TESTFILE')

parser.add_option('--after', metavar='TESTFILE', dest='after_test',
                  help='Run remaining tests, starting after TESTFILE')

parser.add_option('-j', '--jobs', type='int', default=1,
                  help='Run JOBS threads.')

parser.add_option('--reverse', action='store_true',
                  help=('Run tests in reverse order (less overhead with '
                        'multiple jobs)'))

parser.add_option('-k', '--keep-temp-dir', action='store_true',
                  dest='keep_tmpdir', help='Do not delete temporary files.')

opt, tests = parser.parse_args()


if len(tests) == 0:
    from gpaw.test import tests

if opt.reverse:
    tests.reverse()

if opt.run_failed_tests_only:
    tests = [line.strip() for line in open('failed-tests.txt')]

exclude = []
if opt.exclude is not None:
    exclude += opt.exclude.split(',')

if opt.from_test:
    fromindex = tests.index(opt.from_test)
    tests = tests[fromindex:]

if opt.after_test:
    index = tests.index(opt.after_test) + 1
    tests = tests[index:]

for test in exclude:
    if test in tests:
        tests.remove(test)

from gpaw.test import TestRunner

old_hooks = hooks.copy()
hooks.clear()
if mpi.rank == 0:
    tmpdir = tempfile.mkdtemp(prefix='gpaw-test-')
else:
    tmpdir = None
tmpdir = mpi.broadcast_string(tmpdir)
cwd = os.getcwd()
os.chdir(tmpdir)
if mpi.rank == 0:
    print 'Running tests in', tmpdir
failed = TestRunner(tests, jobs=opt.jobs).run()
os.chdir(cwd)
if mpi.rank == 0:
    if len(failed) > 0:
        open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
    elif not opt.keep_tmpdir:
        os.system('rm -rf ' + tmpdir)
hooks.update(old_hooks.items())

