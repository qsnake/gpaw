from gpaw import Calculator
import Numeric as num
from os.path import isfile
from gpaw.io.array import load_array as load
from gpaw.io.array import save_array as save
gridrefinement = 2

# Load calculation parameters
calc = Calculator('H2O.gpw')
H2O = calc.get_atoms()
a_c = num.diagonal(H2O.GetUnitCell())
N_c = gridrefinement * calc.GetNumberOfGridPoints()
h_c = calc.GetGridSpacings() / gridrefinement

# Get AE-density
if isfile('H2O-AEdensity.gz'):
    n_xy = load('H2O-AEdensity.gz', typecode=num.Float)
else:
    if calc.GetSpinPolarized():
        n_g = num.sum(calc.GetAllElectronDensity(gridrefinement))
    else:
        n_g = calc.GetAllElectronDensity(gridrefinement)

    # Check that density integrates properly
    dv = num.product(calc.GetGridSpacings()) / gridrefinement**3
    print 'Number of electrons:', num.sum(n_g.flat) * dv

    # Dump xy crossection to file
    n_xy = n_g[:, :, N_c[2] / 2]
    save('H2O-AEdensity.gz', n_xy)

# Construct bader volumes
if not isfile('H2O-bader-volumes.dat'):
    from os import system
    system('bader H2O.cube 2')
    system('rm BvAt*.dat BCF.dat AtomVolumes.dat dipole.dat')
    system('mv bader_rho.dat H2O-bader-volumes.dat')
    system('mv ACF.dat H2O-bader-charges.dat')
    
# Load bader volumes
bader = load('H2O-bader-volumes.dat')
charge = tuple(load('H2O-bader-charges.dat', skiprows=[0,1,2])[:, -3])
bader.shape = N_c[::-1]
bader = num.transpose(bader)

# Clip domain
n_xy = num.clip(n_xy, 0, .01 * max(n_xy.flat))
c = num.transpose(num.array([N_c[:2] / 4, 3 * N_c[:2] / 4]))
extent = tuple((c * h_c[:2]).flat)
n_xy = n_xy[c[0,0]:c[0,1], c[1,0]:c[1,1]]
bader_xy = bader[c[0,0]:c[0,1],c[1,0]:c[1,1], N_c[2] / 2]


from pylab import *

# Plot density
f = figure(1, figsize=(14, 7))
f.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=.1, hspace=.1)

subplot(121)
imshow(n_xy, extent=extent, interpolation='bicubic',
       origin='lower', cmap=cm.jet)
xlabel('x'); ylabel('y'); title('H2O density crossection')

subplot(122)
contour(n_xy, extent=extent,
       interpolation='bicubic',
       origin='lower', cmap=cm.jet)
contour(bader_xy, [1.5], extent=extent,
       interpolation='bicubic',
       origin='lower', cmap=cm.jet)
xlabel('x'); ylabel('y')
title('Bader partitions. Partial charges: %4.2f, %4.2f, and %4.2f'%charge)
savefig('H2O.png', dpi=60)
