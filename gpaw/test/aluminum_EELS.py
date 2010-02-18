import numpy as np
import sys
import time

from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.grid_chi import CHI
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

assert size <= 4**3

# Ground state calculation

t1 = time.time()

a = 4.043
atoms = bulk('Al', 'fcc', a=a, orthorhombic=True)

calc = GPAW(h=0.3,
            kpts=(4,4,4),
            usesymm=None,
            xc='LDA')

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Al.gpw','all')

t2 = time.time()



# Excited state calculation
calc = GPAW('Al.gpw',communicator=serial_comm)
            
dw = 0.1  
wmax = 24.

q = np.array([0.25, 0., 0.])
nkptxyz = np.array([4, 4, 4]) 
chi = CHI()
chi.initialize((calc,), q, wmax, dw, nkptxyz, eta=0.2, Ecut = 10)
chi.periodic()

chi.get_EELS_spectrum()

print 'For ground state calc, it took', (t2 - t1) / 60, 'minutes'

d = np.loadtxt('EELS')
wpeak = 16.2 # eV
Nw = 162
if d[Nw, 1] > d[Nw-1, 1] and d[Nw, 2] > d[Nw+1, 2]:
    pass
else:
    raise ValueError('Plasmon peak not correct ! ')

if (np.abs(d[Nw, 1] - 31.339994669) > 1e-5
    or np.abs(d[Nw, 2] -  31.2582002997) > 1e-5 ):
    raise ValueError('Please check spectrum strength ! ')
