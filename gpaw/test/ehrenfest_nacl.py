from ase import *
from gpaw import *
from gpaw.tddft import *
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
import sys

d = 4.5
atoms = Atoms('NaCl', [(0,0,0),(0,0,d)])
atoms.center(vacuum=4.5)
d = 4.0
atoms.set_positions([(0,0,0),(0,0,d)])
atoms.center()

gs_calc = GPAW(nbands=4, gpts=(64,64,96), xc='LDA', setups='hgh')
atoms.set_calculator(gs_calc)
atoms.get_potential_energy()

gs_calc.write('nacl_hgh_gs.gpw', 'all')

td_calc = TDDFT('nacl_hgh_gs.gpw', propagator='ETRSCN')
evv = EhrenfestVelocityVerlet(td_calc, 0.001)

i=0
evv.get_energy()
r = evv.x[1][2] - evv.x[0][2]
print 'E = ', [i, r, evv.Etot, evv.Ekin, evv.Epot]
    
for i in range(10000):
    evv.propagate(1.0)
    evv.get_energy()
    r = evv.x[1][2] - evv.x[0][2]
    print 'E = ', [i+1, r, evv.Etot, evv.Ekin, evv.Epot]
