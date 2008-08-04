import traceback

from gpaw import Calculator
from gpaw.mpi import rank, MASTER
from ase.parallel import paropen
from ase.data.molecules import atoms, g1, molecule
from ase.optimize.qn import QuasiNewton
from sys import stderr

cell = [12., 13., 14.]
data = paropen('g1data.txt', 'w')
systems = atoms + g1

for formula in systems:
    # Load the molecular structure from database
    loa = molecule(formula)
    loa.set_cell(cell)
    loa.center()

    # Attach a gpaw calculator
    calc = Calculator(h=.18,
                      nbands=-5,
                      xc='PBE',
                      fixmom=True,
                      txt=formula + '.txt')
    loa.set_calculator(calc)

    # For atoms: set the initial occupations according to Hund's rules
    if len(loa) == 1:
        calc.set(hund=True)

    # BeH has a weird magnetic structure, so we need to set the initial guess
    # manually
    if formula == 'BeH':
        calc.initialize(loa)
        calc.nuclei[0].f_si = [(1, 0, 0.5, 0),
                               (0.5, 0, 0, 0)]

    # Attach a structure optimizer
    qn = QuasiNewton(loa, logfile=formula + '.qnlog',
                     trajectory=formula + '.traj')
    
    try:
        # Get the energy at the experimental structure
        e0 = loa.get_potential_energy()

##         # Here you could do geometry optimization:
##         qn.run(fmax=0.02, steps=80)
##         e1 = loa.get_potential_energy()

##         # evaluate different xc functionals self-consistently:
##         calc.set(xc='RPBE')
##         e2 = loa.get_potential_energy()

##         # ... or non-selfconsistently:
##         e3 = e1 + calc.get_xc_difference('PW91')

##         # or find the HOMO-LUMO gap:
##         occ = calc.get_occupation_numbers()
##         HOMO = calc.nbands - occ[::-1].searchsorted(1) - 1
##         gap = -calc.get_eigenvalues()[HOMO]
##         if calc.get_spin_polarized():
##             occ_minor = calc.get_occupation_numbers(spin=1)
##             LUMO_minor = calc.nbands - occ_minor[::-1].searchsorted(1)
##             gap += min(calc.get_eigenvalues(spin=0)[HOMO + 1],
##                        calc.get_eigenvalues(spin=1)[LUMO_minor])
##         else:
##             gap += calc.get_eigenvalues()[HOMO + 1]
        
    except:
        print >> data, formula, 'Error'
        if rank == MASTER:
            print >> stderr, 'Error in', formula
            traceback.print_exc(file=stderr)
    else:
        print >> data, formula, repr(e0) # repr is used to get all digits
        calc.write(formula + '.gpw')#, mode='all')
        data.flush()
