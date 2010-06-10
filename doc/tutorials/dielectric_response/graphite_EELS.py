from math import sqrt
import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DF

# Part 1: Ground state calculation
a=1.42
c=3.355
atoms = Atoms('C4',[                  # Generate graphite AB-stack structure.
              (1/3.0,1/3.0,0),
              (2/3.0,2/3.0,0),
              (0.   ,0.   ,0.5),
              (1/3.0,1/3.0,0.5)
              ],
              pbc=(1,1,1))
atoms.set_cell([(sqrt(3)*a/2.0,3/2.0*a,0),
                (-sqrt(3)*a/2.0,3/2.0*a,0),
                (0.,0.,2*c)],
               scale_atoms=True)

calc = GPAW(xc='LDA',                        
            kpts=(20,20,7),           # The result should be converged with respect to kpoints.
            h=0.2,
            basis='dzp',              # Use LCAO basis to get good initialization for unoccupied states.
            nbands=70,                # The result should also be converged with respect to bands.
            convergence={'bands':60}, # It's better NOT to converge all bands. 
            eigensolver='cg',         # It's preferable to use 'cg' to calculate unoccupied states.
            occupations=FermiDirac(0.05),
            txt='out_gs.txt')

atoms.set_calculator(calc)       
atoms.get_potential_energy()          
calc.write('graphite.gpw','all')

# Part 2: Spectra calculations            
f = paropen('graphite_q_list', 'w')     # Write down q.

for i in range(1,8):                    # Loop over different q.   
    df = DF(calc='graphite.gpw',       
            nbands=60,                  # Use only bands that are converged in the gs calculation.
            q=np.array([i/20., 0., 0.]),      # Gamma - M excitation
            #q=np.array([i/20., -i/20., 0.])  # Gamma - K excitation
            w=np.linspace(0, 40, 401),  # Spectra from 0-40 eV with 0.1 eV spacing.
            eta=0.2,                    # Broadening parameter.
            ecut=40+(i-1)*10,           # In general, larger q requires larger planewave cutoff energy.       # 
            txt='out_df_%d.txt' %(i))   # Write differnt output for different q.

    df1, df2 = df.get_dielectric_function()
    df.get_EELS_spectrum(df1, df2, filename='graphite_EELS_%d' %(i)) # Use different filenames for different q
    df.check_sum_rule(df1, df2)         # Check f-sum rule.

    print >> f, sqrt(np.inner(df.qq_v / Bohr, df.qq_v / Bohr))



