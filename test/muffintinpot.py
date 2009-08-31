from ase import *
from gpaw import GPAW, restart, extra_parameters

usenewxc = extra_parameters.get('usenewxc')
extra_parameters['usenewxc'] = True
from gpaw.utilities.kspot import AllElectronPotential
try:
    be = Atoms(symbols='Be',positions=[(0,0,0)])
    be.center(vacuum=5)
    calc = GPAW(h=0.17, xc='LDA', nbands=1) #0.1 required for accuracy
    be.set_calculator(calc)
    be.get_potential_energy()
    AllElectronPotential(calc).write_spherical_ks_potentials('bepot.txt')
    f = open('bepot.txt')
    lines = f.readlines()
    f.close()
    for l in lines[2:]:
        # TODO: Improve accuracy
        assert eval(l.split(' ')[3])<0.01
except:
    extra_parameters['usenewxc'] = usenewxc
    raise
else:
    extra_parameters['usenewxc'] = usenewxc
    
