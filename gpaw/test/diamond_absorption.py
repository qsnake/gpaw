import numpy as np
import sys
import time

from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(h=0.2,
            kpts=(4,4,4),
            occupations=FermiDirac(0.001))

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C_gs.gpw','all')


# Macroscopic dielectric constant calculation
q = np.array([0.0, 0.00001, 0.])
w = np.linspace(0, 24., 241)

df = DF(calc='C_gs.gpw', q=q, w=(0.,), eta=0.001,
        ecut=50, hilbert_trans=False, optical_limit=True)
df1, df2 = df.get_dielectric_function()
eM1, eM2 = df.get_macroscopic_dielectric_constant(df1, df2)

eM1_ = 6.1518509552
eM2_ = 6.0525646447

if (np.abs(eM1 - eM1_) > 1e-5 or
    np.abs(eM2 - eM2_) > 1e-5):
    print eM1, eM2
    raise ValueError('Macroscopic dielectric constant not correct ! ')


# Absorption spectrum calculation
df = DF(calc='C_gs.gpw', q=q, w=w, eta=0.25,
        ecut=50, optical_limit=True)
df1, df2 = df.get_dielectric_function()
df.get_absorption_spectrum(df1, df2)
df.check_sum_rule(df1, df2)

