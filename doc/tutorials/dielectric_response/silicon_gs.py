from ase.structure import bulk
from gpaw import GPAW, FermiDirac

a = 5.431 
atoms = bulk('Si', 'diamond', a=a)

calc = GPAW(h=0.20,
        kpts=(12,12,12),              # The result should be converged with respect to kpoints !
        xc='LDA',                      
        basis='dzp',                  # Use LCAO basis to get good intilialization for unoocupied states.
        txt='si_gs.txt',          
        nbands=80,                    # The result should also be converged with respect to bands!
        eigensolver='cg',             # It's preferable to use 'cg' to calculate unoccupied states.
        occupations=FermiDirac(0.001),# Refer to the note below. 
        convergence={'bands':70})     # It's better NOT to converge all bands. 

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('si.gpw','all')            
