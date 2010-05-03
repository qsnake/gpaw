def agts(queue):
    setups = queue.add('setups.py')
    run = queue.add('run.py --setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    dks = queue.add('dks.py --setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    box = queue.add('h2o_xas_box1.py --setups=.',
                    ncpus=8, walltime=25, deps=[setups])
    queue.add('submit.agts.py --setups=.', deps=[run, dks, box],
              creates=['xas_h2o_spectrum.png', 'h2o_xas_box.png'],
              show=['xas_h2o_spectrum.png', 'h2o_xas_box.png'])

if __name__ == '__main__':
    from gpaw.test import equal
    execfile('plot.py')
    e_dks = float(open('dks.py_--setups=..output').readline().split()[2])
    equal(e_dks, 532.774, 0.001)
    execfile('h2o_xas_box2.py')
