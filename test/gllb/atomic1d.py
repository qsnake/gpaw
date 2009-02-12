#!/usr/bin/env python
from ase import *
from gpaw.utilities import equal
from gpaw.atom.all_electron import AllElectron

data = []
def out(a,b,c,d,e,f):
    data.append( (a,b,c,d,e,f) )

ETotal = {'Be': -14.572+0.012, 'Ne': -128.548 -0.029, 'Mg': -199.612 - 0.005 }
EX = {'Be': -2.666 - 0.010, 'Ne': -12.107 -0.122, 'Mg': -15.992 -0.092 }
EHOMO = {'Be': -0.309 + 0.008, 'Ne': -0.851 + 0.098, 'Mg': -0.253 + 0.006}
eignum = {'Be': 0, 'Ne':3, 'Mg':0 }

xcname = 'GLLB'
atoms = ['Be','Ne','Mg']
for atom in atoms:
    # Test AllElectron GLLB
    GLLB = AllElectron(atom, xcname = xcname, scalarrel = False, gpernode = 600)
    GLLB.run()
    
    out("Total energy", "1D", atom,  ETotal[atom] , GLLB.ETotal,"Ha")
    out("Exchange energy", "1D", atom, EX[atom], GLLB.Exc,"Ha")
    out("HOMO Eigenvalue", "1D", atom, EHOMO[atom], GLLB.e_j[-1],"Ha")

print "             Quanity        Method    Symbol     Ref[1]         GPAW      Unit  "
for a,b,c,d,e,f in data:
    print "%20s %10s %10s   %10.3f   %10.3f   %5s" % (a,b,c,d,e,f)
    
#        print """References:
#[1] Self-consistent approximation to the Kohn-Sham exchange potential
#Gritsenko, Oleg; Leeuwen, Robert van; Lenthe, Erik van; Baerends, Evert Jan
#Phys. Rev. A Vol. 51 p. 1944"""

