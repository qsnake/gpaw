from ase import *
from gpaw import *
from gpaw.elf import ELF
from gpaw.test import equal
from gpaw.mpi import rank

atoms = molecule('CO')
atoms.center(2.0)
calc = GPAW(h=0.24)
atoms.set_calculator(calc)
atoms.get_potential_energy()

elf = ELF(calc)
elf.initialize(calc)
elf.update(calc.wfs)
elf_G = elf.get_electronic_localization_function(spin=0,gridrefinement=1)
elf_g = elf.get_electronic_localization_function(spin=0,gridrefinement=2)
# for parallel calculations, collect arrays
elf_G = calc.gd.collect(elf_G)
elf_g = calc.finegd.collect(elf_g)
# integrate the CO bond
if rank == 0:
    # bond area
    x0 = (atoms.positions[0][0] - 1.0)/atoms.get_cell()[0,0]
    x1 = 1 - x0
    y0 = (atoms.positions[0][1] -1.0)/atoms.get_cell()[1,1]
    y1 = 1 - y0
    z0 = atoms.positions[1][2]/atoms.get_cell()[2,2]
    z1 = atoms.positions[0][2]/atoms.get_cell()[2,2]
    Gx0, Gx1 = calc.gd.N_c[0]*x0, calc.gd.N_c[0] * x1
    Gy0, Gy1 = calc.gd.N_c[1]*y0, calc.gd.N_c[1] * y1
    Gz0, Gz1 = calc.gd.N_c[2]*z0, calc.gd.N_c[2] * z1
    gx0, gx1 = calc.finegd.N_c[0]*x0, calc.finegd.N_c[0] * x1
    gy0, gy1 = calc.finegd.N_c[1]*y0, calc.finegd.N_c[1] * y1
    gz0, gz1 = calc.finegd.N_c[2]*z0, calc.finegd.N_c[2] * z1
    int1 = elf_G[Gx0:Gx1,Gy0:Gy1,Gz0:Gz1].sum() * calc.gd.dv
    int2 = elf_g[gx0:gx1,gy0:gy1,gz0:gz1].sum() * calc.finegd.dv
    equal(int1, 14.5144296205, 0.0001)
    equal(int2, 13.190140251, 0.0001)
