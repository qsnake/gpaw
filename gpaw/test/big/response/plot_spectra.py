import numpy as np
import pylab as pl

def plot_spectra(head):
# plot EELS spectra
    d = np.loadtxt(head + 'q_list')
    q = d[:]
    ndata = q.shape[0] + 1
    w = np.zeros(ndata)
    w2 = np.zeros(ndata)
    
    pl.subplot(1,2,1)
    for i in range(1,ndata):
        filename = head + 'EELS_' + str(i) 
    
        d = np.loadtxt(filename)
        pl.plot(d[:,0], d[:,2],'-k')
        
    fsize = 14
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('EELS', fontsize=fsize)
    #pl.ylim([0, 6])
    pl.title('EELS spectra of ' + head, fontsize=fsize)
    
    
    # plot optical absorption specctrum
    d = np.loadtxt('Absorption')
    pl.plot(d[:,0], d[:,1], '-k', label='$\mathrm{Re}\epsilon(\omega)$')
    pl.plot(d[:,0], d[:,2], '-r', label='$\mathrm{Im}\epsilon(\omega)$')
    pl.title('Dielectric function of ' + head)
    pl.legend()
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('$\epsilon$', fontsize=fsize)
    
plot_spectra('graphite')
plot_spectra('graphene')

pl.show()
