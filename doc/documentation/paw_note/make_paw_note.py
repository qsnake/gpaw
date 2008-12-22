# creates: Pt.png paw_note.pdf

import os
import numpy as npy
import pylab as plt
from gpaw.atom.all_electron import AllElectron

ae = AllElectron('Pt')
ae.run()

fig = plt.figure(1, figsize=(5, 4), dpi=80)
fig.subplots_adjust(left=.05, bottom=.11, right=.85, top=.95)
for n, l, u in zip(ae.n_j, ae.l_j, ae.u_j):
    plt.plot(ae.r, u, label='%i%s' % (n, 'spdf'[l]))

rcut = 2.5
lim = [0, 3.5, -2, 3]
plt.plot([rcut, rcut], lim[2:], 'k--', label='_nolegend_')
plt.axis(lim)
plt.legend(loc=(1.02, .04), pad=.05, markerscale=1)
plt.xlabel(r'$r$ [Bohr]')
plt.text(rcut + .05, lim[2] + .05, '$r_c$', ha='left', va='bottom')
plt.text(.6, 2, '[Pt] = [Xe]4f$^{14}$5d$^9$6s$^1$')
plt.savefig('Pt.png', dpi=80)
#plt.show()


error = 0
error += os.system('pdflatex paw_note > /dev/null')
error += os.system('bibtex paw_note > /dev/null')
error += os.system('pdflatex paw_note > /dev/null')
error += os.system('pdflatex paw_note > /dev/null')
error += os.system('cp paw_note.pdf ../../_build')

assert error == 0
