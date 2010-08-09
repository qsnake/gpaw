import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.odd.sic import SIC
from gpaw.test import equal
#
# reference values for H
eref = 24.936209
niterref = 24
#
# reference values for H2
e2ref = 45.040328
niter2ref = 25
F_ac_ref = np.array([[  0.0000,  0.0000,   0.2129],
                     [  0.0000,  0.0000,  -0.2129]])
#
# tolerance
energy_tolerance = 0.00001
force_tolerance  = 0.001
niter_tolerance  = 0
#
atom     = Atoms('H', positions=[(0.0,0.0,0.0)], magmoms=[1.0])
molecule = Atoms('H2', positions=[(0.0,0.0,0.0), (0.0,0.0,0.737)],
                 magmoms=[+1.0,-1.0])
#
atom.center(vacuum=3.0)
molecule.center(vacuum=3.0)
#
calc = GPAW(xc=SIC(nspins=2),
            h=0.25,
            setups='hgh',
            convergence=dict(eigenstates=1e-9, density=1e-5, energy=1e-5),
            txt='-',
            spinpol=True)
#
atom.set_calculator(calc)
e = atom.get_potential_energy()
niter = calc.get_number_of_iterations()
#
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
niter2 = calc.get_number_of_iterations()
F_ac = molecule.get_forces()

print '   atomic energy (calc, reference, error):',e,eref,abs(e-eref)
print 'molecular energy (calc, reference, error):',e2,e2ref,abs(e2-e2ref)
print 'forces:'
print F_ac
print 'ref forces:'
print F_ac_ref
ferr = np.abs(F_ac - F_ac_ref).max()
print 'max force error', ferr

equal(e, eref, energy_tolerance) 
#equal(niter, niterref, niter_tolerance)
equal(e2, e2ref, energy_tolerance) 
#equal(niter2, niter2ref, niter_tolerance)
equal(ferr, 0.0, force_tolerance)


