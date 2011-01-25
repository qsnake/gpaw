def agts(queue):
    queue.add('testsuite.agts.py', ncpus=8, walltime=30)

if __name__ == '__main__':
    # Run test suite
    import gpaw.test.test
    import os
    assert not os.path.isfile('failed-tests.txt')
