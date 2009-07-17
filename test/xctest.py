from ase import *
from gpaw import GPAW
from gpaw.utilities import equal

a = 2.8
k = 6
g = 12
def printstr(calc):
    print ('Energy = %0.5f eV, '
           'niter = %i, '
           'xc_correction = %0.3f sec') % (
        calc.get_potential_energy(),
        calc.iter,
        calc.timer.timers['Hamiltonian: xc_correction'])

if 1: # spin polarized LDA
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None, xc='LDA')
    bulk = Atoms('Li', magmoms=[1.], pbc=True, cell=[a, a, a], calculator=calc)
    E = bulk.get_potential_energy()
    printstr(calc)
    equal(E, -1.87472627206, 1e-4) # reference value from gpaw rev. 4362

if 1: # spin paired GGA (libxc)
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None, xc='PBE')
    bulk = Atoms('Li', pbc=True, cell=[a, a, a], calculator=calc)
    E = bulk.get_potential_energy()
    printstr(calc)
    equal(E, -1.74718132024, 1e-4) # reference value from gpaw rev. 4362

if 1: # spin polarized GGA (libxc)
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None, xc='PBE')
    bulk = Atoms('Li', magmoms=[1.], pbc=True, cell=[a, a, a], calculator=calc)
    E = bulk.get_potential_energy()
    printstr(calc)
    equal(E, -1.74848080152, 1e-4) # reference value from gpaw rev. 4362

if 1:
    # spin polarized GGA (gpaw built_in)
    calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None,xc='oldPBE')
    bulk = Atoms('Li', magmoms=[1.], pbc=True, cell=[a, a, a], calculator=calc)
    E = bulk.get_potential_energy()
    printstr(calc)
    equal(E, -1.74853307793, 1e-4) # reference value from gpaw rev. 4362

## python xctest.py --gpaw usenewxc=False
## Energy = -1.87473 eV, niter = 24, xc_correction = 1.092 sec
## Energy = -1.74718 eV, niter = 13, xc_correction = 0.672 sec
## Energy = -1.74848 eV, niter = 38, xc_correction = 5.295 sec
## Energy = -1.74853 eV, niter = 32, xc_correction = 2.922 sec

## python xctest.py --gpaw usenewxc=True
## Energy = -1.87473 eV, niter = 24, xc_correction = 1.178 sec
## Energy = -1.74718 eV, niter = 13, xc_correction = 0.717 sec
## Energy = -1.74848 eV, niter = 38, xc_correction = 5.229 sec
## Energy = -1.74853 eV, niter = 32, xc_correction = 3.001 sec
