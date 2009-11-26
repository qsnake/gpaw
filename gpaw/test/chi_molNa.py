from ase import *
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.finite_sys import FiniteSys

a = 6.
c = a / 2.
mol = Atoms([Atom('Na', [c, c, c])], pbc=False, cell=(a, a, a))

basis = BasisMaker('Na').generate(1, 1)
calc = GPAW(h=.2, basis={'Na': basis}, mode='lcao')

mol.set_calculator(calc)
mol.get_potential_energy()

dw = 0.1  # eV 
q = 0.
w_cut = 24.
wmin = 0.
wmax = 20.

a = FiniteSys()

SNonInter, SRPA, SLDA, eCasidaRPA, eCasidaLDA, sCasidaRPA, sCasidaLDA = (
              a.get_dipole_strength(calc, q, w_cut, wmin, wmax, dw))

#f = open('Dipole_strength','w')
#for iw in range(SRPA.shape[0]):
#    print >> f,iw*dw, SNonInter[iw].sum(), SRPA[iw].sum(), SLDA[iw].sum()

#f = open('Casida', 'w')
#print >> f, 'RPA energy, strength, LDA energy, strength'
#for i in range(eCasidaRPA.shape[0]):
#    print >> f, eCasidaRPA[i], sCasidaRPA[i].sum(), eCasidaLDA[i], sCasidaLDA[i].sum()

for i in range(eCasidaRPA.shape[0]):
    if abs(eCasidaRPA[i] - 9.06846) > 1e-5 or abs(eCasidaLDA[i] - 8.27616) > 1e-5:
        raise ValueError('Excitation energy not correct ! ')
