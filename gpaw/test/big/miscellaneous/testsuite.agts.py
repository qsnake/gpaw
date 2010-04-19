def agts(queue):
    queue.add('testsuite.agts.py', ncpus=8)

if __name__ == '__main__':
    # Run test suite
    import gpaw.test.test
