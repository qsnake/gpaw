from gpaw import *
from ase import *
from gpaw.tddft import TDDFT


# Sodium dimer, Na2
d = 1.5
atoms = Atoms( symbols='Na2',
               positions=[( 0, 0, d),
                          ( 0, 0,-d)],
               pbc=False)


# Optimize ground state geometry

atoms.center(vacuum=4.0)
geom_calc = Calculator(nbands=1, h=0.25, xc='PBE')
atoms.set_calculator(geom_calc)
e = atoms.get_potential_energy()
geom_opt = QuasiNewton(atoms)
geom_opt.run(fmax=0.05)


# Calculate ground state for TDDFT

# Larger box
atoms.center(vacuum=6.0)
# Larger grid spacing, LDA is ok
gs_calc = Calculator(nbands=1, h=0.35, xc='LDA')
atoms.set_calculator(gs_calc)
e = atoms.get_potential_energy()
gs_calc.write('na2_gs.gpw', 'all')

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
