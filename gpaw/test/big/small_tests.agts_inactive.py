def agtsmain(env):
    env.process('small_tests.agts.py', metadata=dict(ncpu=4))

if __name__ == '__main__':
    # Run test suite
    import gpaw.test.test
