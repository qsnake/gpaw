#!/usr/bin/env python
import os
import gc
import sys
import time
import tempfile
from optparse import OptionParser

import gpaw.mpi as mpi


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

parser.add_option('-s', '--subprocesses', action='store_true',
                  help='Run tests in subprocesses.')

opt, tests = parser.parse_args()


if len(tests) == 0:
    from gpaw.test import tests

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

from gpaw.test import run_all

if mpi.rank == 0:
    tmpdir = tempfile.mkdtemp(prefix='gpaw-test-')
else:
    tmpdir = None
tmpdir = mpi.broadcast_string(tmpdir)
cwd = os.getcwd()
os.chdir(tmpdir)
if mpi.rank == 0:
    print 'Running tests in', tmpdir
failed = run_all(tests, jobs=opt.jobs, subprocesses=opt.subprocesses)
if mpi.rank == 0:
    os.chdir(cwd)
    if len(failed) > 0:
        open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
    else:
        os.system('rm -rf ' + tmpdir)
