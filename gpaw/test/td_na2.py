from ase import *
from gpaw import *
from gpaw.tddft import *
from gpaw.test import equal
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
gs_calc = GPAW(nbands=1, h=0.35, xc='LDA')
atoms.set_calculator(gs_calc)
e = atoms.get_potential_energy()
niter = gs_calc.get_number_of_iterations()
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


td_ipabs = TDDFT('na2_td2.gpw')
ip_abc = LinearAbsorbingBoundary(5.0, 0.01, atoms.positions)
td_ipabs.set_absorbing_boundary(ip_abc)
td_ipabs.propagate(time_step, iters, 'na2_dmz3.dat', 'na2_td3.gpw')

td_ip4abs = TDDFT('na2_td3.gpw')
ip4_abc = P4AbsorbingBoundary(5.0, 0.05, atoms.positions, 2.0)
td_ip4abs.set_absorbing_boundary(ip4_abc)
td_ip4abs.propagate(time_step, iters, 'na2_dmz4.dat', 'na2_td4.gpw')

td_pmlabs = TDDFT('na2_td4.gpw', solver='BiCGStab')
pml_abc = PML(5.0, 0.01)
td_pmlabs.set_absorbing_boundary(pml_abc)
td_pmlabs.propagate(time_step, iters, 'na2_dmz5.dat', 'na2_td5.gpw')


# photoabsorption_spectrum('na2_dmz2.dat', 'na2_spectrum_z2.dat', width=0.3)

#os.remove('na2_gs.gpw')
#os.remove('na2_td.gpw')
#os.remove('na2_dmz.dat')
#os.remove('na2_spectrum_z.dat')
#os.remove('na2_td2.gpw')
#os.remove('na2_dmz2.dat')
# os.remove('na2_spectrum_z2.dat')

energy_tolerance = 0.0001
niter_tolerance = 0
equal(e, -1.24941356939, energy_tolerance) # svnversion 5252
equal(niter, 21, niter_tolerance) # svnversion 5252
