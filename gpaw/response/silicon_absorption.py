import numpy as np
import sys
import time

from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull

# Ground state calculation
a = 10.16 * Bohr
atoms = bulk('Si', 'diamond', a=a)

calc = GPAW(h=0.20,
            kpts=(4,4,4),
            xc='LDA',
            basis='dzp',
            nbands=80,
            eigensolver='cg',
            convergence={'bands':70})
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_gs.gpw','all')


# Dielectric function Calculation
calc = GPAW('Si_gs.gpw',communicator=serial_comm)

dw = 0.05
wmax = 24.
q = np.array([0.0, 0.00001, 0.])

# getting macroscopic constant
df = DF(calc=calc, q=q, wmax=wmax, dw=dw, eta=0.0001,
        wlist=(0.,), HilbertTrans=False,
        sigma=1e-5, Ecut=150, OpticalLimit=True)

df.initialize()
df.calculate()

df1, df2 = df.get_dielectric_function()
eM1, eM2 = df.get_macroscopic_dielectric_constant(df1, df2)

print 'Macroscopic dilectric constant: '
print 'Without local field correction:', eM1
print 'With    local field correction:', eM2

# getting absorption spectrum
#df = DF(calc=calc, q=q, wmax=wmax, dw=dw, eta=0.1,
#        sigma=1e-5, Ecut=150, OpticalLimit=True)

#df.initialize()
#df.calculate()

#df1, df2 = df.get_dielectric_function()
#df.get_absorption_spectrum(df1, df2)
#df.check_sum_rule(df1, df2)
