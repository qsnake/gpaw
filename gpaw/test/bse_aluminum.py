import numpy as np
from ase.units import Bohr, Hartree
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.response.bse import BSE

GS = 1
df = 1
bse = 1
check_spectrum = 1

if GS:
    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    atoms.center()
    calc = GPAW(h=0.2,
                kpts=(4,2,2),
                xc='LDA',
                nbands=4,
                convergence={'band':'all'})
    
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Al.gpw','all')

if bse:
    
    bse = BSE('Al.gpw',w=np.linspace(0,24,241),
              q=np.array([0.25, 0, 0]),ecut=50., eta=0.2)

    bse.get_dielectric_function('Al_bse.dat')
    
if df:

    # Excited state calculation
    q = np.array([1/4.,0.,0.])
    w = np.linspace(0, 24, 241)
    
    df = DF(calc='Al.gpw', q=q, w=w, eta=0.2, ecut=50,hilbert_trans=False)
    df1, df2 = df.get_dielectric_function()
    df.get_EELS_spectrum(df1, df2,filename='Al_df.dat')
    df.write('Al.pckl')
    df.check_sum_rule()


if check_spectrum:

    d = np.loadtxt('Al_bse.dat')[:,2] 
    wpeak = 16.4 
    Nw = 164
    if d[Nw] > d[Nw-1] and d[Nw] > d[Nw+1]:
        pass
    else:
        raise ValueError('Plasmon peak not correct ! ')
    
    if np.abs(d[Nw] - 27.5317730322) > 1e-5:
        print d[Nw]
        raise ValueError('Please check spectrum strength ! ')

    d2 = np.loadtxt('Al_df.dat')
    if np.abs((d[:240] - d2[:240, 2]).sum()) > 1e-4:
        raise ValueError('Please compare two spectrum')

    
