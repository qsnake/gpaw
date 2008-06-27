from gpaw import *
from ase import *
from gpaw.tddft import *
import os

# Sodium dimer, Na2
d = 1.5
atoms = Atoms( symbols='Na2',
               positions=[( 0, 0, d),
                          ( 0, 0,-d)],
               pbc=False)


# Calculate ground state for TDDFT

# Larger box
atoms.center(vacuum=5.0)
# Larger grid spacing, LDA is ok
gs_calc = Calculator(nbands=1, h=0.35, xc='LDA')
atoms.set_calculator(gs_calc)
e = atoms.get_potential_energy()
gs_calc.write('na2_gs.gpw', 'all')

# 16 fs run with 8.0 attosec time step
time_step = 8.0 # 8.0 as (1 as = 0.041341 autime)5D
iters =  10     # 2000 x 8 as => 16 fs
# Weak delta kick to z-direction
kick = [0,0,1e-3]

# TDDFT calculator
td_calc = TDDFT('na2_gs.gpw')
# Kick
td_calc.absorption_kick(kick)
# Propagate
td_calc.propagate(time_step, iters, 'na2_dmz.dat', 'na2_td.gpw')
# Linear absorption spectrum
photoabsorption_spectrum('na2_dmz.dat', 'na2_spectrum_z.dat', width=0.3)


td_rest = TDDFT('na2_td.gpw')
td_rest.propagate(time_step, iters, 'na2_dmz2.dat', 'na2_td2.gpw')
photoabsorption_spectrum('na2_dmz2.dat', 'na2_spectrum_z2.dat', width=0.3)

td_calc.td_density.paw = None
td_rest.td_density.paw = None

os.remove('na2_gs.gpw')
os.remove('na2_td.gpw')
os.remove('na2_dmz.dat')
os.remove('na2_spectrum_z.dat')
os.remove('na2_td2.gpw')
os.remove('na2_dmz2.dat')
os.remove('na2_spectrum_z2.dat')

