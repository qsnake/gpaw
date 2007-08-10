from gpaw import Calculator
from ASE import Atom,ListOfAtoms
from ASE.Utilities.Wannier.Wannier import Wannier
from ASE.IO.Cube import WriteCube
from ASE.Utilities.MonkhorstPack import MonkhorstPack


if 1:
    hhbondlenght = 0.9
    natoms = 1
    L = hhbondlenght*(natoms)
    bands = 2

    posx = 0.0
    dx = hhbondlenght
    atomslst = []
    for n in range(natoms):
        atom = Atom('H', (posx, 4.0, 4.0))
        atomslst.append(atom)
        posx = posx + dx
    atoms = ListOfAtoms(atomslst, cell=(L, 10, 10), periodic=True)


    # GPAW calculator:
    kpts = MonkhorstPack((21, 1, 1)) + 2e-5
    calc = Calculator(h=0.18, nbands=bands, xc='PBE', txt='wire.txt',
                      kpts=kpts)
    atoms.SetCalculator(calc)
    # Displace kpoints sligthly, so that the symmetry program 
    # not use inversion symmetry to reduce kpoints.
    energy = atoms.GetPotentialEnergy()
    calc.write('wire.gpw', 'all')
    
try:
    import Scientific.IO.NetCDF
except ImportError:
    print 'This test needs Scientific.IO.NetCDF'
else:
    nwannier = 1
    atoms = Calculator('wire.gpw').get_atoms()
    calc = atoms.GetCalculator()
    wannier = Wannier(numberofwannier=nwannier, calculator=calc,
                      occupationenergy=30.0)

    wannier.SaveZIBlochMatrix('zibloch.pickle')
    wannier.Localize()
    wannier.TranslateAllWannierFunctionsToCell([10,0,0])

    centers = wannier.GetCenters()
    for center in centers: 
       print center

    for n in range(1): 
       wannier.WriteCube(n,'test'+str(n)+'.cube')

