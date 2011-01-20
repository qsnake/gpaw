import numpy as np
from ase.units import Bohr, Hartree
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.response.bse import BSE

GS = 1
df = 1
bse = 1
if GS:
    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    atoms.center()
    calc = GPAW(h=0.2,
                kpts=(4,2,2),
                xc='LDA',
                nbands=8,
                convergence={'band':'all'})
    
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Al.gpw','all')

if df:

    # Excited state calculation
    q = np.array([1/4.,0.,0.])
    w = np.linspace(0, 24, 241)
    
    df = DF(calc='Al.gpw', q=q, w=w, eta=0.2, ecut=50,hilbert_trans=True,nbands=8)
    df1, df2 = df.get_dielectric_function()
    df.get_EELS_spectrum(df1, df2,filename='EELS_Al')
    df.write('Al.pckl')
    df.check_sum_rule()

if bse:
    
    bse = BSE('Al.gpw',w=np.linspace(0,24,241),
              q=np.array([0.25, 0, 0]),ecut=200., eta=0.2, nbands=8
              )
    bse.initialize()

    e_w, chi_w = bse.calculate()

    f = open('Al_bse.dat','w')
    for iw in range(bse.Nw):
        print >> f, bse.dw*iw*Hartree, np.real(chi_w[iw]), np.imag(chi_w[iw])
