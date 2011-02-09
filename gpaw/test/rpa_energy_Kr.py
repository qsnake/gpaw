from ase import *
from ase.structure import bulk
import numpy as np
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation

calc = GPAW(h=0.18, xc='LDA', kpts=(4,4,4), #usesymm=None,
            nbands=15, eigensolver='cg', 
            convergence={'bands': -5},
            communicator=serial_comm)

V = 30.
a0 = (4.*V)**(1/3.)
Kr = bulk('Kr', 'fcc', a=a0)

Kr.set_calculator(calc)
Kr.get_potential_energy()

ecut = 30.
w = np.linspace(0.0, 50., 8)
rpa = RPACorrelation(calc)
E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, w=w)

equal(E_rpa, -3.2, 0.1)
