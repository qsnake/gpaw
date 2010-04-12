#!/usr/bin/env python

from ase import *
from gpaw import GPAW, FermiDirac
from ase.data.molecules import molecule

#----------------------------------
# Initialization
molname = 'benzene-mol'
dimername = 'benzene-dimer'
f = open('benzene-dimer-T-shape.dat','w')
h = 0.18
xc = 'vdW-DF'

#-------------------------------------
# relaxation of the benzene molecule
benz = molecule('C6H6')
benz.set_pbc(False)
benz.center(vacuum=4.0)

calc = GPAW(nbands=-1,
            h=h,
            xc=xc,
            occupations=FermiDirac(0.0),
            txt=molname+'_relax.txt')
benz.set_calculator(calc)

# qn constraint
for i in range(len(benz)):
        plane = FixedPlane(i, (0, 0, 1))
        benz.set_constraint(plane)

qn = QuasiNewton(benz,logfile=molname+'_relax.log',trajectory=molname+'_relax.traj')
qn.run(fmax=0.01)

e_mol = benz.get_potential_energy()
del calc

#-------------------------------------
# relaxation of benzene dimer (T-shaped) intermolecular distance
dimer = benz.copy()
benz.rotate('x', pi/2.,center='COM')
benz.translate([0,0,6.])
dimer.extend(benz)
dimer.center(vacuum=4.0) 

calc = GPAW(nbands=-2,
            h=h,
            xc=xc,
            occupations=FermiDirac(0.0),
            txt=dimername+'_relax.txt')
dimer.set_calculator(calc)

# qn constraint
for i in range(6):
        fix = FixBondLength(i, i+6)
        dimer.set_constraint(fix)
for i in range(6):
        fix = FixBondLength(i+12, i+18)
        dimer.set_constraint(fix)

qn = QuasiNewton(dimer,logfile=dimername+'_relax.log',trajectory=dimername+'_relax.traj')
qn.run(fmax=0.01)

calc.write(dimername+'_relax.gpw')
e_dimer = dimer.get_potential_energy()
del calc

#----------------------------------
# Eliminating eggbox-effect in the interaction energy 
# by considering each benzene molecule individually
benz_1 = dimer.copy()
benz_1.set_constraint()
for i in range(12):
        benz_1.pop(0)
 
calc = GPAW(nbands=-1,
            h=h,
            xc=xc,
            occupations=FermiDirac(0.0),
            txt='benz1.txt')
benz1.set_calculator(calc)
e_benz1 = benz1.get_potential_energy()
calc.write(benzname+'-1.gpw')
del calc

benz_2 = dimer.copy()
benz_2.set_constraint()
for i in range(12):
        benz_2.pop(-1)
 
calc = GPAW(nbands=-1,
            h=h,
            xc=xc,
            occupations=FermiDirac(0.0),
            txt='benz2.txt')
benz2.set_calculator(calc)
e_benz2 = benz2.get_potential_energy()
calc.write(benzname+'-2.gpw')
del calc

# interaction energy
e_int = e_dimer - (e_benz1 + e_benz2)

print >> f, e_int, e_dimer, e_benz1, e_benz2
