from ase import *
from gpaw import GPAW, restart, extra_parameters

usenewxc = extra_parameters.get('usenewxc')
extra_parameters['usenewxc'] = True
from gpaw.utilities.kspot import AllElectronPotential
try:
    if 1:
        be = Atoms(symbols='Be',positions=[(0,0,0)])
        be.center(vacuum=5)
        calc = GPAW(h=0.17, xc='LDA', nbands=1) #0.1 required for accuracy
        be.set_calculator(calc)
        be.get_potential_energy()
        #calc.write("be.gpw")

    #be, calc = restart("be.gpw")
    AllElectronPotential(calc).write_spherical_ks_potentials('bepot.txt')
    f = open('bepot.txt')
    lines = f.readlines()
    f.close()
    mmax = 0
    for l in lines:
        mmax = max(abs(eval(l.split(' ')[3])), mmax)

    print "Max error: ", mmax
    assert mmax<0.008
except:
    extra_parameters['usenewxc'] = usenewxc
    raise
else:
    extra_parameters['usenewxc'] = usenewxc
    
