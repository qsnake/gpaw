from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal

def zincblende(symbol1, symbol2, a):
    """Zinc Blende - Zinc Sulfide"""
    atoms = Atoms(symbols='%s2%s2' % (symbol1, symbol2), pbc=True,
                  positions=[(.0, .0, .0),
                             (.5, .5, .5),
                             (.0, .5, .75),
                             (.5, .0, .25),])
    atoms.set_cell([a / sqrt(2), a / sqrt(2), a], scale_atoms=True)
    return atoms

xc = 'GLLB'
a = 5.404 # Si
if 1:
	bulk = zincblende('Si','Si', a)

	calc = GPAW(nbands=8*3,
	            width=0.01,
	            kpts=(5, 5, 7), h = 0.25, xc=xc)
	bulk.set_calculator(calc)
	bulk.get_potential_energy()
	eigs = calc.get_eigenvalues(kpt=0)
	#calc.write('temp.gpw')
        response = calc.hamiltonian.xc.xcfunc.xc.xcs['RESPONSE']
        response.calculate_delta_xc_perturbation()

	del bulk
	del calc

if 0:
    N = 15
    d = 0.5 / N
    N1 = N +1
    q = 1e-5
    kk1 = [ [ q,q,q+i*d ] for i in range(0,N1) ]
    kk2 = [ [ q,q+i*d, 0.5+q ] for i in range(0,N1) ]
    kk3 = [ [ q+i*0.05,q+0.5, q+0.5 ] for i in range(0,N1) ]
    kk4 = [ [ q+0.5-d*i,q+0.5-d*i, q+0.5-d*i ] for i in range(0,N1) ]
    bulk, calc = restart('temp.gpw', fixdensity=True, kpts=kk1+kk2+kk3+kk4)
    bulk.get_potential_energy()
    calc.write('si_bands.gpw')
    
