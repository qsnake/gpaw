from ase import Atoms
from gpaw import GPAW
from gpaw.transport.jstm import STM 
import numpy as np

# GPAW calculations
a = 0.75 # Bond length
cell = np.diag([5,5,12*a])

atoms = Atoms('H12', pbc=(1, 1, 1), cell=cell) 
atoms.positions[:, 2] = [i * a for i in range(12)]

calc = GPAW(h=0.2,
            width = 0.1,
            mode='lcao',
            basis='sz',
            usesymm = False)

# Lead calculation
lead = atoms.copy()
del(lead[4:])
cell_lead = cell.copy()
cell_lead[2,2]=a*len(lead)
lead.set_cell(cell_lead)
calc.set(kpts=(1,1,14))
lead.set_calculator(calc)
lead.get_potential_energy()
calc.write('lead.gpw')

# Tip calculation
tip = atoms.copy()
tip.positions[:] += (atoms.cell/2.0)[0,:]+(atoms.cell/2.0)[1,:]
del(tip[:4])
tip.translate([0,0,10])
tip.cell[2,2]+=10
calc.set(kpts=(1,1,1))
tip.set_calculator(calc)
tip.get_potential_energy()
calc.write('tip.gpw')

# Surface calculation
srf = atoms.copy()
del(srf[8:])
srf.cell[2,2]+=10
srf.set_calculator(calc)
srf.get_potential_energy()
calc.write('srf.gpw')

#STM simulation
tip = GPAW('tip')
srf = GPAW('srf')
lead = GPAW('lead')

stm = STM(tip, srf,
          lead1=lead,
          lead2=lead,
          align_bf=0,
          cvl1=2,
          cvl2=2)

stm.hs_from_paw()
stm.set(dmin=5)
stm.initialize()

stm.scan()
stm.linescan()

if 0:
    stm.plot(repeat = [3,3])


