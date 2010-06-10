import numpy as np
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF

# Part 1: Ground state calculation
atoms = bulk('Si', 'diamond', a=5.431)   # Generate diamond crystal structure for silicon
calc = GPAW(h=0.20, kpts=(4,4,4))        # GPAW calculator initialization
 
atoms.set_calculator(calc)               
atoms.get_potential_energy()             # Ground state calculation is performed
calc.write('si.gpw','all')               # Use 'all' option to write wavefunction

# Part 2 : Spectrum calculation          # DF: dielectric function object
df = DF(calc='si.gpw',                   # Ground state gpw file (with wavefunction) as input
        q=np.array([0.0, 0.00001, 0.0]), # Momentum transfer, here excites in y-direction
        w=np.linspace(0,14,141),         # The Energies (eV) for spectrum: from 0-14 eV with 0.1 eV spacing
        optical_limit=True)              # Indicates that its a optical spectrum calculation

df.get_absorption_spectrum()             # By default, a file called 'Absorption.dat' is generated
