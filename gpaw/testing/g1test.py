#PBS -m abe -l nodes=1:ppn=4,walltime=02:00:00 -N g1test

from gpaw import GPAW
from ase.parallel import paropen
from ase.data.molecules import atoms, g1, molecule
from ase.optimize.qn import QuasiNewton

cell = [11., 12., 13.]
data = paropen('g1data.txt', 'w')

for formula in atoms + g1:
    # Make a gpaw calculator
    calc = GPAW(h=.19,
                nbands=-5,
                xc='PBE',
                fixmom=True,
                txt=formula + '.txt')

    # Load the molecular structure from database
    atoms = molecule(formula, cell=cell, calculator=calc)
    atoms.center()

    # For atoms: set the initial occupations according to Hund's rules
    if len(atoms) == 1:
        calc.set(hund=True)

    # BeH has a weird magnetic structure, so we need to set the initial guess
    # manually
    if formula == 'BeH':
        calc.initialize(atoms)
        calc.nuclei[0].f_si = [(1, 0, 0.5, 0),
                               (0.5, 0, 0, 0)]

    # Attach a structure optimizer
    qn = QuasiNewton(atoms, logfile=formula + '.qnlog',
                     trajectory=formula + '.traj')
    
    # Get the energy at the experimental structure
    e0 = atoms.get_potential_energy()

##     # Here you could do geometry optimization:
##     if len(atoms) > 1:
##         qn.run(fmax=0.02, steps=80)
##     e1 = atoms.get_potential_energy()

##     # evaluate different xc functionals self-consistently:
##     calc.set(xc='RPBE')
##     e2 = atoms.get_potential_energy()

##     # ... or non-selfconsistently:
##     e3 = e1 + calc.get_xc_difference('PW91')

##     # or find the HOMO-LUMO gap:
##     h, l = calc.get_homo_lumo()
##     gap = l - h
        
    print >> data, formula, repr(e0) # repr is used to get all digits
    #calc.write(formula + '.gpw')
    data.flush()
