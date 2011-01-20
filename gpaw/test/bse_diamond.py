import numpy as np
from ase import Atom, Atoms
from ase.structure import bulk
from ase.units import Hartree, Bohr
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE


GS = 1
bse = 1
df = 1

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
              q=np.array([0,0,0.5]),optical_limit=True,ecut=250.,
              nbands=8)
    bse.initialize()

    e_w, chi_w = bse.calculate()

    f = open('C_bse.dat','w')
    for iw in range(bse.Nw):
        print >> f, bse.dw*iw*Hartree, np.imag(chi_w[iw])


if df:
    from gpaw.response.df import DF
    df = DF('C_kpt8.gpw',w=np.linspace(0,20,201),q=np.array([0,0,0.5]),optical_limit=True,ecut=150.)
    df.get_absorption_spectrum(filename='C.dat')

