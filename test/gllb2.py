#!/usr/bin/env python
from ase import *
from gpaw import Calculator
from gpaw.utilities import equal
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.mpi import world

data = []
def out(a,b,c,d,e,f):
    data.append( (a,b,c,d,e,f) )

ETotal = {'Be': -14.572+0.012, 'Ne': -128.548 -0.029, 'Mg': -199.612 - 0.005 }
EX = {'Be': -2.666 - 0.010, 'Ne': -12.107 -0.122, 'Mg': -15.992 -0.092 }
EHOMO = {'Be': -0.309 + 0.008, 'Ne': -0.851 + 0.098, 'Mg': -0.253 + 0.006}
eignum = {'Be': 0, 'Ne':3, 'Mg':0 }

atoms = ['Be','Ne']
xcname = 'GLLB'
if world.rank == 0:
    for atom in atoms:
        # Test AllElectron GLLB
        GLLB = AllElectron(atom, xcname = xcname, scalarrel = False)
	GLLB.run()

	out("Total energy", "1D", atom,  ETotal[atom] , GLLB.ETotal,"Ha")
	out("Exchange energy", "1D", atom, EX[atom], GLLB.Exc,"Ha")
	out("HOMO Eigenvalue", "1D", atom, EHOMO[atom], GLLB.e_j[-1],"Ha")

setup_paths.insert(0, '.')

for atom in atoms:
    if world.rank == 0:
	# Generate non-scalar-relativistic setup for atom
	g = Generator(atom, xcname, scalarrel=False, nofiles=True)
	g.run(**parameters[atom])

    #SS = Atoms([Atom(atom)], cell=(12, 12, 12), pbc=False)
    SS = Atoms([Atom(atom)], cell=(8, 8, 8), pbc=False)
    SS.center()

    #h = 0.20
    h = 0.25
    world.barrier()
    calc = Calculator(h=h, nbands=5, xc=xcname)
    SS.set_calculator(calc)
    E = SS.get_potential_energy()
    eigs = calc.get_eigenvalues()
    if world.rank == 0:
        out("Total energy diff.", "3D", atom, 0.0, E/calc.Ha, "Ha")
        out("Homo eigenvalue", "3D", atom, EHOMO[atom], eigs[eignum[atom]]/calc.Ha, "Ha")

if world.rank == 0:
    print "             Quanity        Method    Symbol     Ref[1]         GPAW      Unit  "
    for a,b,c,d,e,f in data:
	print "%20s %10s %10s   %10.3f   %10.3f   %5s" % (a,b,c,d,e,f)
#        print """References:
#[1] Self-consistent approximation to the Kohn-Sham exchange potential
#Gritsenko, Oleg; Leeuwen, Robert van; Lenthe, Erik van; Baerends, Evert Jan
#Phys. Rev. A Vol. 51 p. 1944"""

