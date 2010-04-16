def agts(queue):
    setups = queue.add('setups.wrap.py')
    run = queue.add('run.wrap.py', ncpus=8, walltime=25, deps=[setups])
    dks = queue.add('dks.wrap.py', ncpus=8, walltime=25, deps=[setups])
    plot = queue.add('plot.wrap.py', deps=[run])
