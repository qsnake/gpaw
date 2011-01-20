import numpy as np
from ase import Atom, Atoms
from ase.structure import bulk
from ase.units import Hartree, Bohr
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE


GS = 1
bse = 1
df = 1
check_spectrum = 1

if GS:

    a = 6.75 * Bohr
    atoms = bulk('C', 'diamond', a=a)

    calc = GPAW(h=0.2,
                kpts=(2,2,2),
                occupations=FermiDirac(0.001),
                nbands=8,
                convergence={'band':'all'})

    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('C_kpt8.gpw','all')

if bse:
    
    bse = BSE('C_kpt8.gpw',w=np.linspace(0,20,201),
              q=np.array([0,0,0.5]),optical_limit=True,ecut=50.,
              nbands=8)

    bse.get_dielectric_function('C_bse.dat')

if df:
    from gpaw.response.df import DF
    df = DF('C_kpt8.gpw',w=np.linspace(0,20,201),q=np.array([0,0,0.5]),
            optical_limit=True,ecut=50., hilbert_trans=False)
    df.get_absorption_spectrum(filename='C.dat')


if check_spectrum:

    d = np.loadtxt('C_bse.dat')[:,2]
    Nw1 = 97
    Nw2 = 109
    if d[Nw1] > d[Nw1-1] and d[Nw1] > d[Nw1+1] and \
       d[Nw2] > d[Nw2-1] and d[Nw2] > d[Nw2+1] :
        pass
    else:
        raise ValueError('Absorption peak not correct ! ')
    
    if np.abs(d[Nw1] - 68.8295454438) > 1e-5 or \
       np.abs(d[Nw2] - 90.2424318491) > 1e-5 :
        print d[Nw1], d[Nw2]
        raise ValueError('Please check spectrum strength ! ')

    d2 = np.loadtxt('C.dat')
    if np.abs((d - d2[:200, 4]).sum()) > 1e-5:
        raise ValueError('Please compare two spectrum')

  
