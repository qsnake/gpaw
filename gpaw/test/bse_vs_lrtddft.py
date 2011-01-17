import numpy as np
from ase import Atom, Atoms
from ase.units import Hartree
from gpaw.mpi import size
from gpaw import GPAW
from gpaw.response.bse import BSE

assert size == 1
# 'to be parallelized'

GS = 1
bse = 1
casida = 1

if GS:
    d = 2.89
    cluster = Atoms([Atom('Na', (0, 0, 0)),
                     Atom('Na', (0, 0, d)),
                     ], pbc=True)
    cluster.set_cell((15.,15.,18.), scale_atoms=False)
    cluster.center()
    calc=GPAW(h=0.3,nbands=8)

    
    cluster.set_calculator(calc)
    cluster.get_potential_energy()
    calc.write('Na2.gpw','all')

if bse:
    
    bse = BSE('Na2.gpw',w=np.linspace(0,15,151),
              q=np.array([0,0,0.0001]),optical_limit=True,ecut=1.,
              nbands=8)
    bse.initialize()

    e_w, chi_w = bse.calculate()


if casida:

    from gpaw.lrtddft import LrTDDFT
    from gpaw.lrtddft import photoabsorption_spectrum

    calc = GPAW('Na2.gpw',txt=None)

    istart=0 # band index of the first occ. band to consider
    jend=7  # band index of the last unocc. band to consider
    lr = LrTDDFT(calc, xc=None, istart=istart, jend=jend,
             nspins=1) # force the calculation of triplet excitations also
    lr.diagonalize()

    print 'lrTDDFT:'
    print lr.get_energies() * Hartree



# result:
#Solve BSE (without Tamm-Dancoff appx.):
#[ 2.66087588 -2.45270587e-22j  3.82295895 +1.64672024e-20j
# -2.66087588 +3.97235097e-20j  3.60083311 +2.96258277e-20j
# -3.82295895 -4.66191366e-20j  3.48319991 +7.18864632e-21j
#  3.76191847 +2.61602985e-20j  3.76191785 +2.00843880e-20j
#  3.48320700 -2.19460293e-22j -3.60083311 +5.12448655e-20j
# -3.48319991 +2.50890886e-20j -3.48320700 +1.55898683e-20j
# -3.76191847 +7.76288591e-17j -3.76191785 -3.93728241e-16j]
#Solve BSE (with Tamm-Dancoff appx.):
#[ 3.06516228  3.73990267  3.78495714  3.78495783  3.82329711  3.82330809
#  4.09109334]
#Solve Casida equation:
#[ 2.66086082  3.48319991  3.483207    3.60083311  3.76191785  3.76191847
#  3.82296943]
#'lrTDDFT:'
#[ 2.66039225  3.48278946  3.48278946  3.60057758  3.76170362  3.76170362
#  3.82283999]
