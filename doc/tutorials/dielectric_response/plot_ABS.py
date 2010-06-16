import numpy as np
from gpaw.test import wrap_pylab
wrap_pylab()
import pylab as pl
    
def plot_ABS(head):    
    # plot optical absorption specctrum
    pl.figure(figsize=(7,5))
    d = np.loadtxt(head+'_abs.dat')
    pl.plot(d[:,0], d[:,3], '-k', label='$\mathrm{Re}\epsilon(\omega)$')
    pl.plot(d[:,0], d[:,4], '-r', label='$\mathrm{Im}\epsilon(\omega)$')

    fsize = 14
    pl.title('Dielectric function of ' + head)
    pl.legend()
    pl.xlabel('Energy (eV)', fontsize=fsize)
    pl.ylabel('$\epsilon$', fontsize=18)
    
plot_ABS('si')

# data from G.Kresse, PRB 73, 045112 (2006) 
x = np.array([2.53, 2.71, 3.08, 3.72, 4.50])

arr1 = pl.arrow(2.53, -3, 0, 3.,width=0.01,head_width=0.1,head_length=1)
arr2 = pl.arrow(2.71, 20, 0, -3.,width=0.01,head_width=0.1,head_length=1)
arr3 = pl.arrow(3.08, 13, 0, 3.,width=0.01,head_width=0.1,head_length=1)
arr4 = pl.arrow(3.72, 32, 0, 3.,width=0.01,head_width=0.1,head_length=1)
arr5 = pl.arrow(4.50, 30, 0, -3.,width=0.01,head_width=0.1,head_length=1)

ax = pl.gca()
ax.add_patch(arr1)
ax.add_patch(arr2)
ax.add_patch(arr3)
ax.add_patch(arr4)
ax.add_patch(arr5)

pl.xlim(0,10)
pl.show()
