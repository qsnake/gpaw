import numpy as np
import sys
import os
import time
from ase import Atom, Atoms
from ase.visualize import view
from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


# Ground state calculation
a = 4.043
atoms = bulk('Al', 'fcc', a=a)
atoms.center()
calc = GPAW(gpts=(12,12,12),
            kpts=(4,4,4),
            xc='LDA')

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Al1.gpw','all')

# Excited state calculation
q = np.array([1./4.,0.,0.])
w = np.linspace(0, 24, 241)

df = DF(calc='Al1.gpw', q=q, w=w, eta=0.2, ecut=50)
df1, df2 = df.get_dielectric_function()
#df.write('Al.pckl')
df.get_EELS_spectrum(df1, df2,filename='EELS_Al_1')

atoms = Atoms('Al8',scaled_positions=[(0,0,0),
                               (0.5,0,0),
                               (0,0.5,0),
                               (0,0,0.5),
                               (0.5,0.5,0),
                               (0.5,0,0.5),
                               (0.,0.5,0.5),
                               (0.5,0.5,0.5)],
              cell=[(0,a,a),(a,0,a),(a,a,0)],
              pbc=True)

calc = GPAW(gpts=(24,24,24),
            kpts=(2,2,2),
            xc='LDA')

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Al2.gpw','all')

# Excited state calculation
q = np.array([1./2.,0.,0.])
w = np.linspace(0, 24, 241)

df = DF(calc='Al2.gpw', q=q, w=w, eta=0.2, ecut=50)
df1, df2 = df.get_dielectric_function()
#df.write('Al.pckl')
df.get_EELS_spectrum(df1, df2,filename='EELS_Al_2')

d1 = np.loadtxt('EELS_Al_1')
d2 = np.loadtxt('EELS_Al_2')
error1 = (d1[1:,1] - d2[1:,1]) / d1[1:,1] * 100
error2 = (d1[1:,2] - d2[1:,2]) / d1[1:,2] * 100

if error1.max() > 0.2 or error2.max() > 0.2: # percent
    print error1.max(), error2.max()
    raise ValueError('Pls check spectrum !')

#if rank == 0:
#    os.remove('Al1.gpw')
#    os.remove('Al2.gpw')

