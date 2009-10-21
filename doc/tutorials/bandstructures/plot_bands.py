import pylab as pl
from gpaw.io.array import load_array
k, e = load_array('Na_bands.txt', transpose=True)
fig = pl.figure(1, dpi=80, figsize=(4.2, 6))
fig.subplots_adjust(left=.15, right=.97, top=.97, bottom=.05)
pl.plot(k, e, 'ro')
pl.axis('tight')
pl.axis(xmin=0, xmax=.5, ymax=8)
pl.xticks([0, .25, .5], [r'$\Gamma$', r'$\Delta$', r'$X$'], size=16)
pl.ylabel(r'$E - E_F\ \rm{[eV]}$', size=16)
pl.savefig('Na_bands.png', dpi=80)
pl.show()
