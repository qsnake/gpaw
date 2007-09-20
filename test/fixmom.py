from gpaw import Calculator
from ASE import Crystal, Atom
from gpaw.utilities import equal

a = 2.87
bulk = Crystal([Atom('Fe', (0, 0, 0), magmom=2.20),
                Atom('Fe', (0.5, 0.5, 0.5), magmom=2.20)])
bulk.SetUnitCell((a, a, a))
mom0 = sum(bulk.GetMagneticMoments())
h = 0.20
calc = Calculator(h=h, nbands=11, kpts=(3, 3, 3),
                  convergence={'eigenstates': 0.02}, fixmom=True)
bulk.SetCalculator(calc)
e = bulk.GetPotentialEnergy()
mom = calc.GetMagneticMoment()
equal(mom, mom0, 1e-5)

