def agts(queue):
    setups = queue.add('setups.py')
    run = queue.add('run.py', args='--setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    dks = queue.add('dks.py', args='--setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    plot = queue.add('submit.agts.py', deps=[run, dks],
                     creates=['xas_h2o_spectrum.png'])

if __name__ == '__main__':
    from gpaw.test import wrap_pylab, equal
    wrap_pylab(['xas_h2o_spectrum.png'])
    execfile('plot.py')

    e_dks = float(open('dks.py.output').readline().split()[2])
    equal(e_dks, 532.774, 0.001)
