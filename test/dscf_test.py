from ase import Atoms
from gpaw import Calculator
from gpaw.dscf import dscf_calculation,MolecularOrbitals
from gpaw.utilities import equal

atoms = Atoms(positions=[[0.,0.,5.],
                         [0.,0.,6.1],
                         [0.,0.,3.]],
              symbols='H2Al',
              cell=[3.,3.,9.1],
              pbc=[True,True,False])

calc = Calculator(h=0.24,
                  nbands=6,
                  xc='PBE',
                  spinpol = True,
                  kpts=[1,1,1],
                  width=0.1,
                  convergence={'energy': 0.01,
                               'density': 1.0e-2,
                               'eigenstates': 1.0e-6,
                               'bands': 14},
                  )

atoms.set_calculator(calc)
e_gs = atoms.get_potential_energy()

sigma_star=MolecularOrbitals(calc, molecule=[0,1],
                             w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]])
dscf_calculation(calc, [[1.0,sigma_star,1]], atoms)

e_exc=atoms.get_potential_energy()

equal(e_exc,e_gs+3.0,1.e-1)
