from ase import *
import numpy as np
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation

w = np.linspace(0.0, 200.0, 32)
ecut = 100.

N = data.molecules.molecule('N')
N.set_cell([5.0, 5.0, 5.0])
N.center()
N.set_pbc(True)
calc = GPAW(h=0.18,
            nbands=25,
            convergence={'bands':-5},
            communicator=serial_comm)
N.set_calculator(calc)
N.get_potential_energy()
rpa = RPACorrelation(calc, nbands=10)    
E1_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, w=w)

N2 = data.molecules.molecule('N2')
N2.set_cell([5.0, 5.0, 5.0])
N2.center()
N2.set_pbc(True)
calc = GPAW(h=0.18,
            nbands=25,
            convergence={'bands':-5},
            communicator=serial_comm)
N2.set_calculator(calc)
N2.get_potential_energy()
rpa = RPACorrelation(calc, nbands=10)    
E2_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, w=w)

equal(E2_rpa - 2*E1_rpa, -2.17, 0.01)
