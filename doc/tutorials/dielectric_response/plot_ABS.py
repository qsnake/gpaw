import numpy as np
import pylab as pl
    
def plot_ABS(head):    
    # plot optical absorption specctrum
    pl.figure()
    d = np.loadtxt(head+'_abs.dat')
    pl.plot(d[:,0], d[:,1], '-k', label='$\mathrm{Re}\epsilon(\omega)$')
    pl.plot(d[:,0], d[:,2], '-r', label='$\mathrm{Im}\epsilon(\omega)$')

    fsize = 14
    pl.title('Dielectric function of ' + head)
    pl.legend()
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('$\epsilon$', fontsize=fsize)
    
plot_ABS('si')

pl.show()
