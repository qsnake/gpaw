import sys

from ase import Atoms
from gpaw import GPAW
from gpaw import KohnShamConvergenceError
from gpaw.utilities import devnull, scalapack

from ase.data.molecules import molecule

# Calculates energy and forces for various parallelizations

tolerance = 4e-5

parallel = dict()

basekwargs = dict(mode='lcao',
                  maxiter=3,
                  #basis='dzp',
                  #nbands=18,
                  nbands=6,
                  parallel=parallel)

Eref = None
Fref_av = None


def run(formula='H2O', vacuum=2.0, cell=None, pbc=0, **morekwargs):
    print formula, parallel
    system = molecule(formula)
    kwargs = dict(basekwargs)
    kwargs.update(morekwargs)
    calc = GPAW(**kwargs)
    system.set_calculator(calc)
    system.center(vacuum)
    if cell is None:
        system.center(vacuum)
    else:
        system.set_cell(cell)
    system.set_pbc(pbc)

    try:
        system.get_potential_energy()
    except KohnShamConvergenceError:
        pass

    E = calc.hamiltonian.Etot
    F_av = calc.forces.calculate(calc.wfs, calc.density,
                                 calc.hamiltonian)

    global Eref, Fref_av
    if Eref is None:
        Eref = E
        Fref_av = F_av

    eerr = abs(E - Eref)
    ferr = abs(F_av - Fref_av).max()

    if calc.wfs.world.rank == 0:
        print 'Energy', E
        print
        print 'Forces'
        print F_av
        print
        print 'Errs', eerr, ferr

    if eerr > tolerance or ferr > tolerance:
        if calc.wfs.world.rank == 0:
            stderr = sys.stderr
        else:
            stderr = devnull
        if eerr > tolerance:
            print >> stderr, 'Failed!'
            print >> stderr, 'E = %f, Eref = %f' % (E, Eref)
            msg = 'Energy err larger than tolerance: %f' % eerr
        if ferr > tolerance:
            print >> stderr, 'Failed!'
            print >> stderr, 'Forces:'
            print >> stderr, F_av
            print >> stderr
            print >> stderr, 'Ref forces:'
            print >> stderr, Fref_av
            print >> stderr
            msg = 'Force err larger than tolerance: %f' % ferr
        print >> stderr
        print >> stderr, 'Args:'
        print >> stderr, formula, vacuum, cell, pbc, morekwargs
        print >> stderr, parallel
        raise AssertionError(msg)
        

run()

parallel['band'] = 2
parallel['domain'] = (1, 2, 1)

run()

if scalapack():
    parallel['scalapack'] = (2, 2, 2, 'd')
    run()

    OH_kwargs = dict(formula='NH2', vacuum=1.5, pbc=1, spinpol=1, width=0.1)

    Eref = None
    Fref = None

    del parallel['domain']
    parallel['scalapack'] = (2, 1, 2, 'd')

    run(**OH_kwargs)

    parallel['domain'] = (1, 2, 1)

    run(**OH_kwargs)




# gamma point, kpts, kpt parallelization
# spinpair, spinpol, spin parallelization
# pbc, non-pbc
# blacs, noblacs, state parallelization, domain decomposition
