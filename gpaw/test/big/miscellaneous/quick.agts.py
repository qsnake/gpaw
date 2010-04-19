def agts(queue):
    queue.add('quick.agts.py', walltime=2, creates=['hydrogen.txt'])

if __name__ == '__main__':
    from gpaw import GPAW
    from ase import Atoms
    
    a = Atoms('H', magmoms=[1])
    a.center(vacuum=3.5)
    a.set_calculator(GPAW(txt='hydrogen.txt'))
    e = a.get_potential_energy()
    
