from gpaw import GPAW
from gpaw.xas import XAS
import pylab as p

dks_energy = 532.774614261  #from dks calcualtion

calc = GPAW('h2o_xas.gpw')
calc.set_positions()

xas = XAS(calc, mode='xas')
x, y = xas.get_spectra(fwhm=0.5, linbroad=[4.5, -1.0, 5.0])
x_s, y_s = xas.get_spectra(stick=True)

shift = dks_energy - x_s[0] # shift the first transition 

y_tot = y[0] + y[1] + y[2]
y_tot_s = y_s[0] + y_s[1] + y_s[2]

p.plot(x +shift, y_tot)
p.bar(x_s +shift, y_tot_s, width=0.001)
p.savefig('xas_spectrum.png')
p.show()
