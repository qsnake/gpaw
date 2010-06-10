import numpy as np
from gpaw.test import wrap_pylab
wrap_pylab()
import pylab as pl

def plot_EELS(head):
# plot EELS spectra
    pl.figure(figsize=(4,7))

    d = np.loadtxt(head + '_q_list')
    q = d[:,0]
    ndata = q.shape[0] + 1
    w = np.zeros(ndata)
    w2 = np.zeros(ndata)
    
    for i in range(1,ndata):
        filename = head + '_EELS_' + str(i) 

        d = np.loadtxt(filename)
        pl.plot(d[:,0], d[:,2] + 0.4*(ndata-i-1),'-k', label=str(q[i-1])[:4])
        
    fsize = 14
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('Loss Function', fontsize=fsize)
    pl.title('EELS spectra of ' + head +': $\Gamma-\mathrm{M}$', fontsize=fsize)
    pl.ylim(0,)
    pl.legend(loc=2)
    
plot_EELS('graphite')

pl.show()
