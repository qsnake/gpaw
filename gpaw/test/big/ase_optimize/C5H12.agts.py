def agts(queue):
    queue.add('C5H12.agts.py',
              walltime=25,
              ncpus=8)

if __name__ == "__main__":
    from ase.optimize.test.C5H12 import *
