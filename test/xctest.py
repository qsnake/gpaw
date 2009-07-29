from ase import *
from gpaw import GPAW, extra_parameters
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
        calc.timer.timers.get('Hamiltonian: atomic: xc_correction', 0))

atom_kwargs = dict(pbc=True, cell=[a, a, a])
calc_kwargs = dict(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None)

# Reference value are from gpaw rev. 4362
def run_test(tests=[0, 1, 2, 3]):
    if 0 in tests: # spin polarized LDA (libxc)
        calc = GPAW(xc='LDA', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        printstr(calc)
        equal(E, -1.87472627206, 1e-4)

    if 1 in tests:# spin paired GGA (libxc)
        calc = GPAW(xc='PBE', **calc_kwargs)
        bulk = Atoms('Li', calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        printstr(calc)
        equal(E, -1.74718132024, 1e-4)

    if 2 in tests: # spin polarized GGA (libxc)
        calc = GPAW(xc='PBE', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        printstr(calc)
        equal(E, -1.74848080152, 1e-4)

    if 3 in tests: # spin polarized GGA (gpaw built_in)
        calc = GPAW(xc='oldPBE', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        printstr(calc)
        equal(E, -1.74853307793, 1e-4)

usenewxc = extra_parameters.get('usenewxc')
try:
    print 'Old xc_correction'
    extra_parameters['usenewxc'] = False
    run_test()

    print
    print 'New xc_correction'
    extra_parameters['usenewxc'] = True
    run_test()
finally:
    extra_parameters['usenewxc'] = usenewxc

## Old xc_correction
## Energy = -1.87473 eV, niter = 24, xc_correction = 1.093 sec
## Energy = -1.74718 eV, niter = 13, xc_correction = 0.666 sec
## Energy = -1.74848 eV, niter = 38, xc_correction = 5.393 sec
## Energy = -1.74853 eV, niter = 32, xc_correction = 3.010 sec

## New xc_correction
## Energy = -1.87473 eV, niter = 24, xc_correction = 1.197 sec
## Energy = -1.74718 eV, niter = 13, xc_correction = 0.727 sec
## Energy = -1.74848 eV, niter = 38, xc_correction = 5.280 sec
## Energy = -1.74853 eV, niter = 32, xc_correction = 3.103 sec
