import numpy as np
import pylab as pl

def plot_EELS(head):
# plot EELS spectra
    pl.figure(figsize=(4,7))

    d = np.loadtxt(head + '_q_list')
    q = d[:]
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
    
def plot_ABS(head):    
    # plot optical absorption specctrum
    pl.figure()
    d = np.loadtxt(head+'_absorption')
    pl.plot(d[:,0], d[:,1], '-k', label='$\mathrm{Re}\epsilon(\omega)$')
    pl.plot(d[:,0], d[:,2], '-r', label='$\mathrm{Im}\epsilon(\omega)$')
    pl.title('Dielectric function of ' + head)
    pl.legend()
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('$\epsilon$', fontsize=fsize)
    pl.ylim([-2,20])
    
plot_EELS('graphite')
#plot_EELS('graphene')

pl.show()
