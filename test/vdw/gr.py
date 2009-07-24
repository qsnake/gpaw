import os
from ase import *
from gpaw import GPAW, setup_paths
from gpaw.vdw import FFTVDWFunctional
from ase.parallel import rank, barrier
from gpaw.atom.generator import Generator, parameters

def test():
   
    
     h=4.02
    
     vdw = FFTVDWFunctional(verbose=1)
        
     L=20
     a=3.52
    
     atoms = Atoms(pbc=(True, True, True), cell=(a/sqrt(2), sqrt(6)*a/2.0, L))
     atoms.append(Atom('C',[1.5*a/sqrt(2)*1.0/3,a/sqrt(2)*sqrt(3)/2*1.0/3, L/2+h]))
     atoms.append(Atom('C',[1.5*a/sqrt(2)*2.0/3,a/sqrt(2)*sqrt(3)/2*2.0/3, L/2+h]))
     atoms.append(Atom('C',[a/sqrt(2)/2.0+1.5*a/sqrt(2)*1.0/3,sqrt(6)*a/4.0+a/sqrt(2)*sqrt(3)/2*1.0/3, L/2+h]))
     atoms.append(Atom('C',[a/sqrt(2)/2.0+1.5*a/sqrt(2)*2.0/3-a/sqrt(2),sqrt(6)*a/4.0+a/sqrt(2)*sqrt(3)/2*2.0/3, L/2+h]))
 
 
     atoms.append(Atom('C',[1.5*a/sqrt(2)*1.0/3,a/sqrt(2)*sqrt(3)/2*1.0/3+2.874/2, L/2]))
     atoms.append(Atom('C',[1.5*a/sqrt(2)*2.0/3,a/sqrt(2)*sqrt(3)/2*2.0/3+2.874/2, L/2]))
     atoms.append(Atom('C',[a/sqrt(2)/2.0+1.5*a/sqrt(2)*1.0/3,sqrt(6)*a/4.0+a/sqrt(2)*sqrt(3)/2*1.0/3+2.874/2, L/2]))
     atoms.append(Atom('C',[a/sqrt(2)/2.0+1.5*a/sqrt(2)*2.0/3-a/sqrt(2),sqrt(6)*a/4.0+a/sqrt(2)*sqrt(3)/2*2.0/3+2.874/2, L/2]))
     calc = GPAW(h=0.18, xc='revPBE',kpts=(8,8,1),txt=str(h)+'.txt')
     atoms.set_calculator(calc)
     e2 = atoms.get_potential_energy()
     calc.write('gr_dilayer.gpw')
     e2vdw = calc.get_xc_difference(vdw)
     del atoms[4:]
     e = atoms.get_potential_energy()
     calc.write('gr_dilayer.gpw')
     evdw = calc.get_xc_difference(vdw)
    
     E = 2 * e - e2
     Evdw = E + 2 * evdw - e2vdw
     print E, Evdw
     assert abs(E +0.032131069056 ) < 1e-4
     assert abs(Evdw- 0.144516773316) < 1e-4

test()
