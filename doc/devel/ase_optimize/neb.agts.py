def agts(queue):
    queue.add('neb.agts.py',
              walltime=36*60,
              ncpus=8,
              creates=['neb-emt.csv', 'neb-gpaw.csv'])

if __name__ == "__main__":
    from ase.optimize.test.neb import *

