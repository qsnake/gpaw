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

assert size <= 8 

print 'Ground state calculation started at:'
t1 = time.time()
print time.ctime()

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a, orthorhombic=True)

calc = GPAW(h=0.3,
            kpts=(2,2,2),
            xc='LDA',
            usesymm = None)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C_gs1.gpw','all')

# GS Calculation Two
q = np.array([0.0001, 0., 0.])
kpt = q +  calc.get_ibz_k_points()

calc = GPAW(h=0.3,
            kpts=kpt,
            xc='LDA',
            usesymm = None)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C_gs2.gpw','all')

del calc

print 'Dielectric function calculation started at:'
t2 = time.time()
print time.ctime()

# Dielectric function Calculation
calc1 = GPAW('C_gs1.gpw',communicator=serial_comm)
calc2 = GPAW('C_gs2.gpw',communicator=serial_comm)
            
dw = 0.1  
wmax = 24.

chi = CHI()
chi.initialize((calc1,calc2), q, wmax, dw, eta=0.27, Ecut = 10)
chi.periodic()

eM1, eM2 = chi.get_macroscopic_dielectric_constant()
print 
print 'Macroscopic dilectric constant: '
print 'Without local field correction:', eM1
print 'With    local field correction:', eM2

#chi.get_absorption_spectrum()
#chi.check_sum_rule()

print
print 'Dielectric function calculation ended at:'
t3 = time.time()
print time.ctime()

print
print 'For ground state calc, it took', (t2 - t1) / 60, 'minutes'
print 'For excited state calc, it took', (t3 - t2) / 60, 'minutes'


eM1_ = 3.49324502383
eM2_Ecut10 = 3.49324502383
if (np.abs(eM1 - eM1_) > 1e-5 or
    np.abs(eM2 - eM2_Ecut10) > 1e-5):
  raise ValueError('Macroscopic dielectric constant not correct ! ')
