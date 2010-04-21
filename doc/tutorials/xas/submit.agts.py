def agts(queue):
    setups = queue.add('setups.py')
    run = queue.add('run.py --setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    dks = queue.add('dks.py --setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    queue.add('submit.agts.py --setups=.', deps=[run, dks],
              creates=['xas_h2o_spectrum.png'],
              show=['xas_h2o_spectrum.png'])

if __name__ == '__main__':
    from gpaw.test import equal
    execfile('plot.py')
    e_dks = float(open('dks.py_--setups=..output').readline().split()[2])
    equal(e_dks, 532.774, 0.001)
