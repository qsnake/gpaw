#!/usr/bin/env python
from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

w = np.linspace(0.0, 200.0, 32)

calc1 = GPAW('N_4000.gpw', communicator=serial_comm, txt=None)
calc2 = GPAW('N2_4000.gpw', communicator=serial_comm, txt=None)

rpa1 = RPACorrelation(calc1, txt='rpa_N.txt')    
rpa2 = RPACorrelation(calc2, txt='rpa_N2.txt')    

for ecut in [150, 200, 250, 300, 350, 400]:
    E1 = rpa1.get_rpa_correlation_energy(ecut=ecut, w=w)
    E2 = rpa2.get_rpa_correlation_energy(ecut=ecut, w=w)

    f = paropen('rpa_atomization.dat', 'a')
    print >> f, ecut, bands, E2 - 2*E1
    f.close()
