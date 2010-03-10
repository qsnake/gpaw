from ase import *
from gpaw import GPAW, extra_parameters
from gpaw.test import equal, gen

gen('Li', xcname='oldPBE')

a = 2.8
k = 6
g = 12
def printstr(calc):
    print ('Energy = %0.6f eV, '
           'niter = %i, '
           'xc_correction = %0.3f sec') % (
        calc.get_potential_energy(),
        calc.iter,
        calc.timer.timers.get('Hamiltonian: atomic: xc_correction', 0))

atom_kwargs = dict(pbc=True, cell=[a, a, a])
calc_kwargs = dict(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None)

# Reference value are from gpaw rev. 5066 with stencils=(3,3)
def run_test(tests=[0, 1, 2, 3]):
    if 0 in tests: # spin polarized LDA (libxc)
        calc = GPAW(xc='LDA', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        niter = calc.get_number_of_iterations()
        printstr(calc)
        equal(E, -1.873879, 1e-4)
        assert 26 <= niter <= 27, niter

    if 1 in tests:# spin paired GGA (libxc)
        calc = GPAW(xc='PBE', **calc_kwargs)
        bulk = Atoms('Li', calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        niter = calc.get_number_of_iterations()
        printstr(calc)
        equal(E, -1.746562, 2e-5)
        equal(niter, 19, 0)

    if 2 in tests: # spin polarized GGA (libxc)
        calc = GPAW(xc='PBE', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        niter = calc.get_number_of_iterations()
        printstr(calc)
        equal(E, -1.747841, 3e-5)
        assert 34 <= niter <= 51, niter

    if 3 in tests: # spin polarized GGA (gpaw built_in)
        calc = GPAW(xc='oldPBE', **calc_kwargs)
        bulk = Atoms('Li', magmoms=[1.], calculator=calc, **atom_kwargs)
        E = bulk.get_potential_energy()
        niter = calc.get_number_of_iterations()
        printstr(calc)
        equal(E, -1.747855, 1e-6)
        equal(niter, 39, 0)

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

## svnversion 5252
## Energy = -1.873575 eV, niter = 27, xc_correction = 2.309 sec
## Energy = -1.746265 eV, niter = 19, xc_correction = 1.967 sec
## Energy = -1.747591 eV, niter = 34, xc_correction = 8.891 sec
## Energy = -1.747517 eV, niter = 44, xc_correction = 7.662 sec
