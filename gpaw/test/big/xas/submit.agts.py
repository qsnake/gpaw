def agtsmain(env):
    setups = env.add('setups.wrap.py')
    run = env.add('run.wrap.py', ncpus=8, walltime=25, depends=[setups])
    dks = env.add('dks.wrap.py', ncpus=8, walltime=25, depends=[setups])
    plot = env.add('plot.wrap.py', depends=[run])
