from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.test import equal
from gpaw.test import gen
from ase.structure import bulk

gen('Si', xcname='GLLB')

def diamond2(symbol, a):
    return bulk(symbol, 'diamond', a=a)

def diamond4(symbol, a):
    """Zinc Blende - Zinc Sulfide"""
    atoms = Atoms(symbols='%s4' % (symbol), pbc=True,
                  positions=[(.0, .0, .0),
                             (.5, .5, .5),
                             (.0, .5, .75),
                             (.5, .0, .25),])
    atoms.set_cell([a / sqrt(2), a / sqrt(2), a], scale_atoms=True)
    return atoms


def diamond8(symbol, a):
    atoms = Atoms(symbols='%s8' % (symbol), pbc=True,
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
    # 2 atom unit cell
    bulk = diamond2('Si', a)
    calc = GPAW(nbands=2*3,
                width=0.01,
                kpts=(32, 32, 32), h = 0.3, xc=xc)
    bulk.set_calculator(calc)
    E2_tot = bulk.get_potential_energy()
    calc.write('Si2_GLLB.gpw')
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    Eks2, Dxc2 = response.calculate_delta_xc_perturbation()    

    """
    # 4 atom unit cell
    bulk = diamond4('Si', a)
    calc = GPAW(nbands=4*3,
                width=0.01,
                kpts=(24, 24, 20), h = 0.2, xc=xc)
    bulk.set_calculator(calc)
    E4_tot = bulk.get_potential_energy()
    calc.write('Si4_GLLB.gpw')
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    Eks4, Dxc4 = response.calculate_delta_xc_perturbation()
    """

    # 8 atom unit cell
    bulk = diamond8('Si', a)
    calc = GPAW(nbands=8*3,
                width=0.01,
                kpts=(10,10,10), h = 0.2, xc=xc)
    bulk.set_calculator(calc)
    E8_tot = bulk.get_potential_energy()
    calc.write('Si8_GLLB.gpw')
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    EKs8, Dxc8 = response.calculate_delta_xc_perturbation()

    equals(E2_tot*4, E8_tot, 1e-2)
    equals(Eks8, Eks2, 1e-2)
    equals(Dxc8, Dxc2, 1e-2)
        
