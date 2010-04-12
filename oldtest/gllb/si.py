from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.test import equal

def zincblende4(symbol1, symbol2, a):
    """Zinc Blende - Zinc Sulfide"""
    atoms = Atoms(symbols='%s2%s2' % (symbol1, symbol2), pbc=True,
                  positions=[(.0, .0, .0),
                             (.5, .5, .5),
                             (.0, .5, .75),
                             (.5, .0, .25),])
    atoms.set_cell([a / sqrt(2), a / sqrt(2), a], scale_atoms=True)
    return atoms

def zincblende8(symbol1, symbol2, a):
    atoms = Atoms(symbols='%s4%s4' % (symbol1, symbol2), pbc=True,
                  positions=[(0, 0, 0),
                             (0, 0.5, 0.5),
                             (0.5, 0, 0.5),
                             (0.5, 0.5, 0),
                             (0.25, 0.25, 0.25),
                             (0.25, 0.75, 0.75),
                             (0.75, 0.25, 0.75),
                             (0.75, 0.75, 0.25)])
    atoms.set_cell([a,a,a], scale_atoms=True)
    return atoms

xc = 'GLLB'
a = 5.404 # Si
if 1:
    # 8 atom unit cell
    bulk = zincblende8('Si','Si', a)
    calc = GPAW(nbands=8*3,
                width=0.01,
                kpts=(10, 10, 10), h = 0.25, xc=xc)
    bulk.set_calculator(calc)
    E8_tot = bulk.get_potential_energy()
    calc.write('Si8_GLLB.gpw')
    response = calc.hamiltonian.xc.xcfunc.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    EKs, Dxc = response.calculate_delta_xc_perturbation()

    # 4 atom unit cell
    bulk = zincblende4('Si','Si', a)
    calc = GPAW(nbands=8*3,
                width=0.01,
                kpts=(10, 10, 10), h = 0.25, xc=xc)
    bulk.set_calculator(calc)
    E4_tot = bulk.get_potential_energy()
    calc.write('Si4_GLLB.gpw')
    response = calc.hamiltonian.xc.xcfunc.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    EKs2, Dxc2 = response.calculate_delta_xc_perturbation()

    equals(E4_tot*2, E8_tot, 1e-2)
    equals(Eks, Eks2, 1e-2)
    equals(Dxc, Dxc2, 1e-2)
        
