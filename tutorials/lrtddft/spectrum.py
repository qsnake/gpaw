from gpaw import *
from gpaw.lrtddft import LrTDDFT, absorption_spectrum

atoms, calc = restart('na2_gs.gpw')

# Calculate the omega matrix
lr = LrTDDFT(calc, xc='LDA', jend=5)
# Save the omega matrix
lr.write('Omega_Na2.gz')
# Diagonalize the matrix
lr.diagonalize()
# Analyse 5 lowest excitations
lr.analyse(range(5))
absorption_spectrum(lr, 'Na2_spectrum.dat', e_min=0.0, e_max=10)
