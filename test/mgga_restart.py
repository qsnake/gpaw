import os
import sys

from ase import *

from gpaw import GPAW
from gpaw.cluster import Cluster


fname='H2_PBE.gpw'
fwfname='H2_wf_PBE.gpw'

# write first if needed
try:
    c = GPAW(fname)
    c = GPAW(fwfname)
except:
    s = Cluster([Atom('H'), Atom('H', [0,0,1])])
    s.minimal_box(3.)
    c = GPAW(xc='PBE', h=.3, convergence={'density':1e-3, 'eigenstates':1e-3})
    c.calculate(s)
    c.write(fname)
    c.write(fwfname, 'all')

# full information
c = GPAW(fwfname)
E_PBE = c.get_potential_energy()
c.wfs.initialize_wave_functions_from_restart_file()
dE = c.get_xc_difference('TPSS')
E_1 = E_PBE + dE
print "E PBE, TPSS=", E_PBE, E_1

# no wfs
c = GPAW(fname)
E_PBE = c.get_potential_energy()
c.scf.reset()
c.set(fixdensity=True)
c.calculate()
dE = c.get_xc_difference('TPSS')
E_2 = E_PBE + dE
print "E PBE, TPSS=", E_PBE, E_2

print "diff=", E_1 - E_2
assert abs(E_1 - E_2) < 0.005
