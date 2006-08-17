"""
A simple test script.

Runs and times all scripts named ``*.py``.  The tests that execute
the fastest will be run first.
"""
import glob
import time
import pickle
import sys
import os
import gc
import StringIO
from optparse import OptionParser

from ASE.Units import units


parser = OptionParser(usage='%prog [options] [tests]',
                      version='%prog 0.1')

parser.add_option('-x', '--exclude',
                  type='string', default=None,
                  help='Exclude tests (comma separated list of tests).',
                  metavar='test1.py,test2.py,...')

parser.add_option('-f', '--run-failed-tests-only',
                  action='store_true',
                  help='Run failed tests only.')

parser.add_option('-d', '--debug',
                  action='store_true', default=False,
                  help='Run tests in debug mode.')

parser.add_option('-p', '--parallel',
                  action='store_true',
                  help='Add parallel tests.')

options, tests = parser.parse_args()

path = ''

if len(tests) == 0:
    tests = glob.glob(path + '*.py')

if options.debug:
    sys.argv.append('--gridpaw-debug')

exclude = ['__init__.py', 'test.py', 'grr.py', 'C-force.py']
if options.exclude is not None:
    exclude += options.exclude.split(',')

# exclude parallel tests if options.parallel is not set
if not options.parallel: 
    exclude.extend(['parallel-restart.py', 'parmigrate.py']) 

for test in exclude:
    if path + test in tests:
        tests.remove(path + test)
    
gc.set_debug(gc.DEBUG_SAVEALL)

# Read old timings if they are present:
machine=os.uname()[4]
host=os.uname()[1]
timings_file='timings.pickle_' + machine
try:
    timings = pickle.loads(file(timings_file).read())
except IOError:
    timings = {}

# Make a list of tests to do, and sort it with the fastest/new
# tests first:
tests = [(timings.get(test, 0.0), test) for test in tests]
tests.sort()

if options.run_failed_tests_only:
    tests = [(t, test) for t, test in tests if t == 0.0]

L = max([len(test) for told, test in tests])
print '-----------------------------------------------------------------'
print ' Running tests in ', host, ', architecture ', machine
print ' test', ' ' * (L - 4), 'result      time (old)'
print '-----------------------------------------------------------------'

garbage = []
failed = []
# Do tests:
for told, test in tests:
    units.length_used = False 
    units.energy_used = False
    units.SetUnits('Ang', 'eV')
    
    sys.stdout = StringIO.StringIO()
    sys.stderr = StringIO.StringIO()

    print >> sys.__stdout__, '%-*s' % (L + 2, test),
    sys.__stdout__.flush()

    ok = False
    
    module = test[:-3]
    if options.parallel:
        module = module.replace('/', '.')
        
    t = time.time()
    try:
        module = __import__(module, globals(), locals(), [])
        del module
    except KeyboardInterrupt:
        failed.append(test)
        print >> sys.__stdout__, 'STOPPED!'
        print >> sys.__stdout__, ('Hit [enter] to continue with next test, ' +
                                  '[ctrl-C] to stop.')
        try:
            raw_input()
        except KeyboardInterrupt:
            break
        continue
    except AssertionError, msg:
        print >> sys.__stdout__, 'FAILED!'
        print >> sys.__stdout__, msg
    except:
        print >> sys.__stdout__, 'CRASHED!'
        type, value = sys.exc_info()[:2]
        print >> sys.__stdout__, str(type) + ":", value
    else:
        ok = True
        
    t = time.time() - t

    gc.collect()
    n = len(gc.garbage)
    if n > 0:
        print >> sys.__stdout__, ' LEAK!'
        print >> sys.__stdout__, ('Uncollectable garbage (%d object%s)' %
                                  (n, 's'[:n > 1]))
        garbage += gc.garbage
        del gc.garbage[:]
        ok = False
        
    if ok:
        print >> sys.__stdout__, '  OK     %7.3f (%.3f)' % (t, told)
    else:
        failed.append(test)
        out = sys.stdout.getvalue()
        if len(out) > 0:
            open(test + '.output', 'w').write(out)
        err = sys.stderr.getvalue()
        if len(err) > 0:
            open(test + '.error', 'w').write(err)
        t = 0
    timings[test] = t
        
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print '-----------------------------------------------------------------'

if len(tests) > 1:
    print
    if len(failed) == 0:
        print 'All tests passed!'
    elif len(failed) == 1:
        print 'One test out of %d failed: %s' % (len(tests), failed[0])
    else:
        print '%d tests out of %d failed:'% (len(failed), len(tests))
        for test in failed:
            print ' ', test
    print

# Save new timings:
file(timings_file, 'w').write(pickle.dumps(timings))
