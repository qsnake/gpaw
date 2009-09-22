#!/usr/bin/env python
from ase import *
from gpaw import GPAW, extra_parameters

usenewxc = extra_parameters.get('usenewxc')
extra_parameters['usenewxc'] = True

from gpaw.utilities.kspot import CoreEigenvalues
try:
    a = 7.0
    calc = GPAW(h=0.10)
    system = Atoms([Atom('Ne',[a/2,a/2,a/2])], pbc=False, cell=(a, a, a), calculator=calc)
    e0 = system.get_potential_energy()
    e_j = CoreEigenvalues(calc).get_core_eigenvalues(0)
    assert abs(e_j[0]-(-30.344066))*27.21<0.1 # Error smaller than 0.1 eV
except:
    extra_parameters['usenewxc'] = usenewxc
    raise
else:
    extra_parameters['usenewxc'] = usenewxc
    
