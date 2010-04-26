from ase import *
from gpaw import GPAW

atoms =molecule('H2O')
atoms.pbc = False

h=0.2
l=h*8
cells = [4*l, l*6 , l*8, l*10, l*12]

for cell in cells:
    atoms.set_cell((cell,cell,cell))
    atoms.center()

    calc = GPAW(xc='PBE',
                h=h,
                nbands=-40,
                eigensolver='cg',
                setups={'O': 'hch1s'},
                stencils=(3,3) )

    atoms.set_calculator(calc)
    e1 = atoms.get_potential_energy()
    calc.write('h2o_hch_%s.gpw'%(cell))

