def agts(queue):
    queue.add('nanoparticle.agts.py',
              walltime=2 * 60 + 15,
              ncpus=8)

if __name__ == "__main__":
    from ase.optimize.test.nanoparticle import *
