def agts(queue):
    queue.add('N2Ru_relax.agts.py',
              creates=['N2Ru-N2.csv', 'N2Ru-surf.csv'])

if __name__ == "__main__":
    from ase.optimize.test.N2Ru_relax import *
