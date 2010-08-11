import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.odd.sic import SIC
from gpaw.test import equal
#
# reference values for N
eref = 385.004749296
niterref = 47
#
# reference values for N2 (spin-polarized)
#e2ref = 754.774019381
#niter2ref = 63
#F_ac_ref = np.array([[  0.00010,  -0.00117,  -82.8312],
#                     [ -0.00013,  -0.00121,   82.8313]])
#
# reference values for N2 (spin-saturated)
e2ref = 754.775205926
niter2ref = 61
F_ac_ref = np.array([[  0.32375,  -0.00100,  -82.8350],
                     [  0.32380,  -0.00104,   82.8342]])
#
# tolerance
energy_tolerance = 0.00001
force_tolerance  = 0.001
niter_tolerance  = 2
#
atom     = Atoms('N', positions=[(0.0,0.0,0.0)], magmoms=[3.0])
#molecule = Atoms('N2', positions=[(0.0,0.0,0.0), (0.0,0.0,1.13)],
#                 magmoms=[+3.0,-3.0])
molecule = Atoms('N2', positions=[(0.0,0.0,0.0), (0.0,0.0,1.13)],
                 magmoms=[0.0,0.0])
#
# add vacuum
atom.center(vacuum=3.0)
molecule.center(vacuum=3.0)
#
#
# calculate N atom
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
print 'atomic energy (calc, reference, error):',e,eref,abs(e-eref)
#
equal(e, eref, energy_tolerance) 
equal(niter, niterref, niter_tolerance)
#
#
# calculate N2 molecule
calc2 = GPAW(xc=SIC(nspins=1),
            h=0.25,
            setups='hgh',
            convergence=dict(eigenstates=1e-9, density=1e-5, energy=1e-5),
            txt='-',
            spinpol=False)
#
molecule.set_calculator(calc2)
e2 = molecule.get_potential_energy()
#niter2 = calc2.get_number_of_iterations()
F_ac = molecule.get_forces()
#
print 'molecular energy (calc, reference, error):',e2,e2ref,abs(e2-e2ref)
print 'forces:'
print F_ac
print 'ref forces:'
print F_ac_ref
ferr = np.abs(F_ac - F_ac_ref).max()
print 'max force error', ferr
#
equal(e2, e2ref, energy_tolerance) 
#equal(niter2, niter2ref, niter_tolerance)
equal(ferr, 0.0, force_tolerance)
