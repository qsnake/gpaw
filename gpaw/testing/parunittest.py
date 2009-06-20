
import numpy as np
from unittest import TestResult, _TextTestResult, TextTestRunner#, TestCase
from ase.test import CustomTestCase as TestCase
from gpaw.mpi import world, broadcast_string

# ------------------------------------------------------------------

__all__ = ['ParallelTestResult', 'ParallelTextTestRunner', 'ParallelTestCase']

class ParallelTestResult(TestResult):
    def __init__(self, comm=None):
        if comm is None:
            comm = world
        self.comm = comm
        self.outcomes = []
        TestResult.__init__(self)

    def addError(self, test, err):
        self.has_errors = True
        TestResult.addError(self, test, err)

    def addFailure(self, test, err):
        self.has_failed = True
        TestResult.addFailure(self, test, err)

    def startTest(self, test):
        self.has_errors = False
        self.has_failed = False
        TestResult.startTest(self, test)

    def stopTest(self, test):
        errors_r = np.empty(self.comm.size, dtype=bool)
        self.comm.all_gather(np.array([self.has_errors]), errors_r)

        failed_r = np.empty(self.comm.size, dtype=bool)
        self.comm.all_gather(np.array([self.has_failed]), failed_r)

        TestResult.stopTest(self, test)
        self.outcomes.append((test, errors_r, failed_r))
        return errors_r, failed_r

class _ParallelTextTestResult(ParallelTestResult, _TextTestResult):

    def __init__(self, comm, *args, **kwargs):
        ParallelTestResult.__init__(self, comm)
        _TextTestResult.__init__(self, *args, **kwargs) #BAD FORM!

    def startTest(self, test):
        ParallelTestResult.startTest(self, test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")

    def addSuccess(self, test):
        ParallelTestResult.addSuccess(self, test)

    def addError(self, test, err):
        ParallelTestResult.addError(self, test, err)

    def addFailure(self, test, err):
        ParallelTestResult.addFailure(self, test, err)

    def stopTest(self, test):
        errors_r, failed_r = ParallelTestResult.stopTest(self, test)

        self.stream.flush()
        self.comm.barrier()

        def findranks(status_r):
            if status_r.all():
                return 'all'
            else:
                return ','.join(map(str,np.argwhere(status_r).ravel()))

        if errors_r.any() and failed_r.any():
            if self.showAll:
                self.stream.writeln("BOTH (%s)" % 'WHAT!?')
            elif self.dots:
                self.stream.writeln('B')
            raise RuntimeError('Parallel unittest can\'t handle simultaneous' \
                               + ' errors and failures within a single test.')
        elif errors_r.any():
            if self.showAll:
                self.stream.writeln("ERROR (ranks: %s)" % findranks(errors_r))
            elif self.dots:
                self.stream.writeln('E')
        elif failed_r.any():
            if self.showAll:
                self.stream.writeln("FAIL (ranks: %s)" % findranks(failed_r))
            elif self.dots:
                self.stream.writeln('F')
        else:
            if self.showAll:
                self.stream.writeln("ok")
            else:
                self.stream.writeln('.')

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.writeln()
        self.printErrorList('ERROR-RANK%d' % self.comm.rank, self.errors)
        self.printErrorList('FAIL-RANK%d' % self.comm.rank, self.failures)

    def printErrorList(self, flavour, errors):
        for test, errors_r, failed_r in self.outcomes:
            if flavour.startswith('ERROR'):
                status_r = errors_r
            elif flavour.startswith('FAIL'):
                status_r = failed_r
            else:
                raise RuntimeError('Error type %s is unsupported.' % flavour)

            if status_r.any():
                self.stream.writeln(self.separator1)

            i = 0
            for rank in np.argwhere(status_r).ravel():
                if rank == self.comm.rank:
                    assert status_r[self.comm.rank]
                    test, err = errors[i]
                    text1 = '%s: %s' % (flavour,self.getDescription(test))
                    text2 = '%s' % err
                    i += 1
                else:
                    text1 = None
                    text2 = None
                text1 = broadcast_string(text1, root=rank, comm=self.comm)
                text2 = broadcast_string(text2, root=rank, comm=self.comm)
                self.stream.writeln(text1)
                self.stream.writeln(self.separator2)
                self.stream.writeln(text2)


class ParallelTextTestRunner(TextTestRunner):
    def __init__(self, comm=None, **kwargs):
        if comm is None:
            comm = world
        self.comm = comm
        TextTestRunner.__init__(self, **kwargs)

    def _makeResult(self):
        return _ParallelTextTestResult(self.comm, self.stream, \
            self.descriptions, self.verbosity)

class ParallelTestCase(TestCase):
    def defaultTestResult(self):
        return ParallelTestResult()

