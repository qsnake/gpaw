def agtsmain(env):
    env.add('testsuite.agts.py', ncpu=4)

if __name__ == '__main__':
    # Run test suite
    import gpaw.test.test
