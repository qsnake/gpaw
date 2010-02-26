from gpaw.xas import XAS
import pylab as plt

calc = GPAW('h2o_xas.gpw')
calc.set_positions()

xas = XAS(calc, mode='xas')
x, y = xas.get_spectra(fwhm=0.5, linbroad=[4.5, -1.0, 5.0])
x_s, y_s = xas.get_spectra(stick=True)

y_tot = y[0] + y[1] + y[2]
y_tot_s = y_s[0] + y_s[1] + y_s[2]

plt.plot(x, y_tot)
plt.bar(x_s, y_tot_s, width=0.001)
plt.show()
