def agts(queue):
    queue.add('H2.agts.py',
              walltime=25,
              ncpus=8)

if __name__ == "__main__":
    from ase.optimize.test.H2 import *
