from gpaw import *
from ase import *
from gpaw.tddft import TDDFT


# Sodium dimer, Na2

# 10 fs run with 8.0 attosec time step
time_step = 8.0 # 1 as = 0.041341 autime
iters =  1250   # 10000 x 1 as => 10 fs
# Weak delta kick to z-direction
kick = [0,0,1e-3]

# TDDFT calculator
td_calc = TDDFT('na2_gs.gpw', propagator='SICN', solver='CSCG', tolerance=1e-8)
# Kick
td_calc.absorption_kick(kick)
# Propagate
td_calc.propagate(time_step, iters, 'na2_dmz.dat', 'na2_td.gpw')
# Linear absorption spectrum
TDDFT.photoabsorption_spectrum('na2_dmz.dat', 'na2_spectrum_z.dat', kick)
