#!/usr/bin/python

import os
import sys
import tempfile
import numpy as np

from trace import Trace, pickle

def fail(txt):
    print 'FAIL: %s' % txt

def _err_exit(msg):
    sys.stderr.write("%s: %s\n" % (sys.argv[0], msg))
    sys.exit(1)


def main(outfile, tests=None, testfile='test.py'):
    assert 'gpaw' not in sys.modules, 'GPAW must be unloaded first!'

    home = os.environ['HOME']
    ignore_dirs = [home]

    #import ase
    #ignore_dirs.extend(ase.__path__)
    ignore_dirs.extend(np.__path__)
    ignore_modules = []

    # Temporary file and directory for coverage results on this core
    coverfile = tempfile.mktemp(prefix='gpaw-coverfile-')
    coverdir = tempfile.mkdtemp(prefix='gpaw-coverdir-')

    trace = Trace(count=1, trace=0, countfuncs=False,
                  countcallers=False, ignoremods=ignore_modules,
                  ignoredirs=ignore_dirs, infile=None,
                  outfile=coverfile)

    if tests is not None:
        sys.argv.extend(tests)
    try:
        trace.run('execfile(%r, {})' % (testfile,))
    except IOError, err:
        _err_exit('Cannot run file %r because: %s' % (testfile, err))
    except SystemExit:
        pass

    coverage = trace.results()
    coverage.write_results(summary=False, coverdir=coverdir) #XXX not final

    from gpaw import mpi # do not import from gpaw before trace has been run!

    """
    def receive_strings(rank, comm=mpi.world, sep='\n'):
        return mpi.receive_string(rank, comm).split(sep)

    def send_strings(strings, rank, comm=mpi.world, sep='\n'):
        assert all(map(lambda s: sep not in s, strings))
        mpi.send_string(sep.join(strings), rank, comm)

    def broadcast_strings(strings, root=0, comm=mpi.world, sep='\n'):
        if comm.rank == root:
            assert all(map(lambda s: sep not in s, strings))
            return mpi.broadcast_string(sep.join(strings), root, comm).split(sep)
        else:
            return mpi.broadcast_string(None, root, comm).split(sep)
    """

    mycounts = pickle.load(open(coverfile, 'rb'))[0]

    if mpi.world.rank == 0:
        if os.path.isfile(outfile):
            print 'Continuing cover file...'
            counts = pickle.load(open(outfile, 'rb'))[0]
        else:
            print 'Initializing cover file...'
            counts = {}

    if mpi.world.size == 1:
        # Merge generated cover file with existing (if any)
        for filename,line in mycounts.keys():
            counts[(filename,line)] = mycounts.pop((filename,line)) \
                + counts.get((filename,line), 0)
    else:
        # Find largest line number detected for each locally executed file
        myfiles = {}
        for filename,line in mycounts.keys():
            assert '\n' not in filename
            myfiles[filename] = max(myfiles.get(filename,0), line)

        # Agree on which files have been executed on at least one core
        filenames = myfiles.keys()
        if mpi.world.rank == 0:
            for rank in range(1, mpi.world.size):
                filenames.extend(mpi.receive_string(rank).split('\n'))
            filenames = np.unique(filenames).tolist()
            tmp = '\n'.join(filenames)
        else:
            mpi.send_string('\n'.join(filenames), 0)
            tmp = None
        filenames = mpi.broadcast_string(tmp).split('\n')

        # Map out largest line number detected for each globally executed file
        filesizes = np.array([myfiles.get(filename,0) for filename in filenames])
        mpi.world.max(filesizes)

        # Merge global totals of generated cover files with existing (if any)
        for filename,lines in zip(filenames,filesizes):
            numexecs = np.zeros(lines, dtype=int)
            for l in range(lines):
                if (filename,l+1) in mycounts:
                    numexecs[l] = mycounts.pop((filename,l+1))
            mpi.world.sum(numexecs, 0)
            if mpi.world.rank == 0:
                for l in np.argwhere(numexecs).ravel():
                    counts[(filename,l+1)] = numexecs[l] \
                        + counts.get((filename,l+1), 0)

    # Store as 3-tuple of dicts in a new pickle (i.e. same format)
    if mpi.world.rank == 0:
        pickle.dump((counts,{},{},), open(outfile, 'wb'), 1)
    mpi.world.barrier()

    del mpi, sys.modules['gpaw'] # unload gpaw module before next trace

    assert not mycounts, 'Not all entries were processed: %s' % mycounts.keys()
    os.system('rm -rf ' + coverfile + ' ' + coverdir)

if __name__ == '__main__':
    outfile = 'counts.pickle'
    outdir = 'coverage'

    # Coverage test:
    #tests = ['gemm.py', 'gemv.py']
    tests = []

    main(outfile, tests)
    #os.system('python -m trace --report --file %s --coverdir %s --missing' % (outfile, outdir))

    """
    names = []
    for filename in filenames:
        missing = 0
        for line in open(filename):
            if line.startswith('>>>>>>'):
                missing += 1
        if missing > 0:
            if filename.startswith('coverage/gpaw.'):
                name = filename[14:-6]
                if os.system('cp %s %s/%s.cover' %
                             (filename, dir, name)) != 0:
                    fail('????')
            else:
                name = filename[28:-6]
            names.append((-missing, name))
    """
