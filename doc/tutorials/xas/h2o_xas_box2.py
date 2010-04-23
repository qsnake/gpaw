from ase import *
from gpaw import GPAW
from gpaw.xas import XAS
import pylab as plt

h=0.2
l=h*8
cells = [4*l, l*6 , l*8, l*10, l*12]

offset=0.005
for cell in cells:

    calc =GPAW('h2o_hch_%s.gpw'%(cell))
    xas = XAS(calc)
    x, y = xas.get_spectra(fwhm=0.4)
    plt.plot(x,sum(y)+offset, label=str(cell))
    offset += 0.01

plt.legend()
plt.xlim(-6, 6)
plt.show()
plt.savefig('h2o_xas_box.png')


