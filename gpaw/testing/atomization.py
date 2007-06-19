#!/usr/bin/python

"""
This file contains utility functions used by the GPAW setup optimizer.

The GPAW setup optimizer performs two different calculations in
molecule tests:

  1) The absolute energy of a molecule E[molecule]
  2) The atomization energy Ea = E[molecule] - 2 * E[single atom]

Aside from these functions, this file contains a number of minor
helper functions and some molecule data.
"""

from ASE import Atom, ListOfAtoms
from gpaw import Calculator

"""
A dictionary which associates element symbols with MoleculeInfo-objects
pertaining to those elements.
"""
elements = {}

class MoleculeInfo:
    """
    Utility class wrapping molecule informations nbands1 and 2 are the
    numbers of bands to be used with single-atom and molecular
    calculations, respectively
    """
    
    def __init__(self, letter, d, magmom1, magmom2, nbands1, nbands2,
                 Ea_PBE=None):
        self.d = d
        self.letter = letter
        self.magmom1 = magmom1
        self.magmom2 = magmom2
        # Figure out how the magnetic moment is modified under
        # hybridization and forming of molecules
        self.nbands1 = nbands1
        self.nbands2 = nbands2
        self.Ea_PBE = Ea_PBE
        elements[letter] = self

#Also test H2 (4.5 eV). magmom=1, nbands=1 in Calculator for H as well as H2

"""
Reference values are taken from the GPAW molecule tests page, reference 2.
"""
molN = MoleculeInfo('N', 1.103, 3, 0, 4, 5, -10.546)
molH = MoleculeInfo('H', 0.751, 1, 0, 1, 1, -4.535)
molO = MoleculeInfo('O', 1.218, 2, 2, 6, 6, -6.231)

"""
Creates an atom. If a separation greater than 0 is specified, creates
two atoms correspondingly spaced along the x axis.

Returns a ListOfAtoms containing whatever was created in this way
"""
def getListOfAtoms(molecule=molN, separation=0, a=5., dislocation=(0.,0.,0.),
                   periodic=False):
    atoms = None
    (dx, dy, dz) = dislocation
    (cx, cy, cz) = (a/2. + dx, a/2. + dy, a/2. + dz)
    d = separation/2.
    if separation==0:
        #One atom only
        atoms = ListOfAtoms([Atom(molecule.letter, (cx, cy, cz),
                                  magmom=molecule.magmom1)],
                            periodic=periodic,
                            cell=(a,a,a))
    else:
        #Create two atoms separated along x axis
        #No magnetic moment then! At least not for the molecules we are
        #considering presently

        atoms = ListOfAtoms([Atom(molecule.letter, (cx+d,cy,cz)),
                             Atom(molecule.letter, (cx-d,cy,cz))],
                            periodic=periodic,                            
                            cell=(a,a,a))
    return atoms

"""
Calculates the atomization energy, i.e. E[molecule] - 2*E[single atom].
"""
def calcEnergy(calc1=None, calc2=None, a=4., molecule=molN,
               dislocation=(0,0,0), periodic=False, setup='paw'):

    oneAtom = getListOfAtoms(molecule, a=a, dislocation=dislocation,
                          periodic=periodic)

    if calc1 == None:
        calc1 = MakeCalculator(nbands=molecule.nbands1, setup=setup)
    if calc2 == None:
        calc2 = MakeCalculator(nbands=molecule.nbands2, setup=setup)

    #bands: 2s and 2p yield a total of 4 bands; 1s is ignored
    #setups='A1' => will search for /home/ask/progs/gpaw/setups/N.A1.PBE.gz
    oneAtom.SetCalculator(calc1)
    e1 = oneAtom.GetPotentialEnergy()


    #gpts=(n,n,n) - to be varied in multiples of 4
    d = molecule.d

    twoAtoms = getListOfAtoms(molecule, a=a, dislocation=dislocation,
                              periodic=periodic, separation=molecule.d)

    #10 electrons in total from 2s and 2p.
    #Thus it is necessary only to include 5 bands
    twoAtoms.SetCalculator(calc2)
    e2 = twoAtoms.GetPotentialEnergy()

    return e2-2*e1

"""
Using a particular resolution h, test whether energies deviate considerably
if the system is translated in intervals smaller than h.
"""
def displacementTest(a=5., molecule=molN, h=.2):
    print 'Displacement test:', molecule

    h += 0. #floating point

    testcount = 3
    dislocations = []
    
    #Initialise test coordinates
    for value in range(testcount):
        coordinate = h * value/testcount #linear distribution
        dislocations.append((coordinate, 0., 0.))

    print dislocations

    energies = []
    for dislocation in dislocations:
        #e = calcEnergy(a, molecule, dislocation, h)
        e = energyAtDistance(molecule.d, dislocation, h)
        energies.append(e)

    print 'Energies:'
    print energies

    print 'Max',max(energies)
    print 'Min',min(energies)
    print 'Diff',max(energies) - min(energies)

"""
Creates two calculators for the given molecule with appropriate band counts
"""
def atomizationCalculators(molecule=molN, out='-', h=.2, lmax=2, setup='paw'):
    calc1 = MakeCalculator(molecule.nbands1, out, h, lmax, setup=setup)
    calc2 = MakeCalculator(molecule.nbands2, out, h, lmax, setup=setup)
    return (calc1, calc2)


"""
Default calculator setup, however complicated it might become someday
This method allows you to forget about lmax and PBE and such
"""
def MakeCalculator(nbands, out='-', h=.2, lmax=2, setup='paw'):
    return Calculator(nbands=nbands, out=out, h=h, lmax=lmax, xc='PBE',
                      setups=setup)

"""
Calculates the ground-state energy of the given molecule when the atoms
are spaced by the given distance
"""
def energyAtDistance(distance, calc=None, dislocation=(0,0,0),
                     molecule=molN, a=5., periodic=False):
    c = a/2.
    (dx, dy, dz) = dislocation

    coord1 = (c-distance/2. + dx, c + dy, c + dz)
    coord2 = (c+distance/2. + dx, c + dy, c + dz)

    twoAtoms = getListOfAtoms(molecule, distance, a, dislocation,
                              periodic)

    if calc == None:
        calc = MakeCalculator(nbands=molecule.nbands2)

    twoAtoms.SetCalculator(calc)

    energy = twoAtoms.GetPotentialEnergy()
    return energy

"""
Write lists x and y to specified file
"""
def writeResults(x, y, fileName, header=[]):
    if len(x) != len(y):
            raise Exception('Result list length mismatch')
    length = len(x)
    f = open(fileName, 'w')
    lines = [''.join([str(x[i]),'\t',str(y[i]),'\n']) for i in range(length)]

    for line in header:
        line = '# '+line

    f.writelines(header)
    
    f.writelines(lines)
    f.close()

"""
Read list of (x,y) entries from datafiles, return as two lists
"""
def readResults(fileName):
    f = open(fileName, 'r')
    lines = filter(stringFilter, f.readlines())
    length = len(lines)
    pairs = [s.split() for s in lines]
    x = [float(pair[0]) for pair in pairs]
    y = [float(pair[1]) for pair in pairs]
    return (x,y)

"""
Allow comments and empty lines in data files
"""
def stringFilter(s):
    return not (s.startswith('#') or s.isspace())

"""
The gbar doesn't have pylab so use this function
"""
def linspace(start, end, count):
    return [start + float(i)/(count-1)*(end-start) for i in range(count)]

