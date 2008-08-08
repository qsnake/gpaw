from ase import *
from ase.lattice.surface import fcc100, add_adsorbate
from gpaw import *

def aual100(site, height, calc=None):
    slab = fcc100('Al', size=(2, 2, 1))
    add_adsorbate(slab, 'Au', height, site)
    slab.center(axis=2, vacuum=3.0)
    mask = [atom.symbol == 'Al' for atom in slab]
    fixlayer = FixAtoms(mask=mask)
    slab.set_constraint(fixlayer)
    if calc is None:
        calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE', txt=site + '.txt')
    slab.set_calculator(calc)
    qn = QuasiNewton(slab, trajectory=site + '.traj')
    qn.run(fmax=0.05)
    calc.write(site + '.gpw')
    return slab.get_potential_energy()

e_hollow = aual100('hollow', 1.6)
e_bridge = aual100('bridge', 2.0)
e_ontop = aual100('ontop', 2.4)
calc = EMT()
e_hollow = aual100('hollow', 1.6, calc)
e_bridge = aual100('bridge', 2.0, calc)
e_ontop = aual100('ontop', 2.4, calc)
