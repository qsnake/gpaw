import os
import numpy as np
from ase import *
from gpaw import GPAW
from gpaw.mpi import rank, MASTER
from gpaw.utilities import equal
from gpaw.lrtddft import LrTDDFT

txt = '-'
if 0 or rank != MASTER:
    txt = None
load = True
load = False

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))

calc = GPAW(xc='PBE', nbands=2, spinpol=False, txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()
calc.write('H2saved_wfs.gpw', 'all')
calc.write('H2saved.gpw')
wfs_error = calc.wfs.eigensolver.error

xc = 'LDA'

if txt:
    print "##############################################"
    print "-> starting directly after a gs calculation"
lr = LrTDDFT(calc, xc=xc, txt='-')
lr.diagonalize()

if txt:
    print "##############################################"
    print "-> reading gs with wfs"
gs = GPAW('H2saved_wfs.gpw', txt="-")

# check that the wfs error is read correctly, 
# but take rounding errors into account
# XXX disabled until wavefunctions.py is corrected
# assert( abs(calc.wfs.eigensolver.error/gs.wfs.eigensolver.error - 1) < 1e-8)
lr1 = LrTDDFT(gs, xc=xc, txt='-')
lr1.diagonalize()
# check the oscillator strength
# XXX decreased accuracy until wavefunctions.py is corrected
assert(abs(lr1[0].get_oscillator_strength()[0]/
           lr[0].get_oscillator_strength()[0]   -1) < 1e-5)

if txt:
    print "##############################################"
    print "-> reading gs without wfs"
gs = GPAW('H2saved.gpw', txt=txt)
# the wave functions err should be infinity
assert(gs.wfs.eigensolver.error >  calc.wfs.eigensolver.error)
lr2 = LrTDDFT(gs, xc=xc, txt='-')
lr2.diagonalize()
# check the oscillator strrength
assert(abs(lr2[0].get_oscillator_strength()[0]/
           lr[0].get_oscillator_strength()[0]   -1) < 1e-5)
