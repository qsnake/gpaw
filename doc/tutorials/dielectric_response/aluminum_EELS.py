import numpy as np
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF

# Part 1: Ground state calculation
atoms = bulk('Al', 'fcc', a=4.043)  # Generate fcc crystal structure for aluminum
calc = GPAW(h=0.2, kpts=(4,4,4))    # GPAW calculator initialization

atoms.set_calculator(calc)
atoms.get_potential_energy()        # Ground state calculation is performed
calc.write('Al.gpw','all')          # Use 'all' option to write wavefunctions

# Part 2: Spectrum calculation      # DF: dielectric function object
df = DF(calc='Al.gpw',              # Ground state gpw file as input
        q=np.array([1./4., 0, 0]),  # Momentum transfer, must be the difference between two kpoints !
        w=np.linspace(0, 24, 241))  # The Energies (eV) for spectrum: from 0-24 eV with 0.1 eV spacing

df.get_EELS_spectrum()              # By default, a file called 'EELS.dat' is generated
