def main(env):
    metadata0 = dict(ncpus=1,
                     walltime=1)
    setups = env.add('setups.wrap.py', metadata0)

    metadata1 = dict(ncpus=8,
                     walltime=25,
                     depends=[setups])
    run = env.add('run.wrap.py', metadata1)

    metadata2 = dict(ncpus=8,
                     walltime=25,
                     depends=[setups])
    dks = env.add('dks.wrap.py', metadata2)

    metadata3 = dict(ncpus=1,
                     walltime=5,
                     type='post',
                     depends=[run])
    plot = env.add('dks.wrap.py', metadata3)
