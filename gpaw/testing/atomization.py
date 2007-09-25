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
from gpaw.utilities import molecule, singleatom
import atomization_data

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
    
    def __init__(self, symbol, nbands1, nbands2):
        self.symbol = symbol
        formula = symbol + str(2)
        
        atom = singleatom.SingleAtom(symbol, parameters={'txt':None})
        diatomic = molecule.molecules[formula]

        self.magmom1 = sum(atom.atom.GetMagneticMoments())
        self.magmom2 = sum(diatomic.GetMagneticMoments())
        
        self.nbands1 = nbands1
        self.nbands2 = nbands2

        energy_kcal_mol = atomization_data.atomization[formula][2]
        self.energy_pbe = - energy_kcal_mol * 43.364e-3

        # This is obviously going to break hard if someone ever
        # changes molecule.py.  The file contains lists of atoms where
        # the first one is at (0,0,0) and the second, in a diatomic
        # molecule, at (d, 0, 0)
        self.d = diatomic[1].GetCartesianPosition()[0]
        
        elements[symbol] = self

"""
Reference values are taken from the GPAW molecule tests page, reference 2.
"""
for arg in [('H', 1, 1),
            ('Li', 1, 1),
            ('Be', 2, 2),
            ('N', 4, 5),
            ('O', 4, None),
            ('F', None, None),
            ('Cl', None, None)]:
    MoleculeInfo(*arg)



def get_list_of_atoms(symbol='N', separation=0, a=6.,
                      dislocation=(0.,0.,0.),
                      periodic=False):
    """
    Creates an atom. If a separation greater than 0 is specified, creates
    two atoms correspondingly spaced along the x axis.
    
    Returns a ListOfAtoms containing whatever was created in this way
    """
    molecule = elements[symbol]
    
    atoms = None
    (dx, dy, dz) = dislocation
    (cx, cy, cz) = (a/2. + dx, a/2. + dy, a/2. + dz)
    d = separation/2.
    if separation==0:
        #One atom only
        atoms = ListOfAtoms([Atom(symbol, (cx, cy, cz),
                                  magmom=molecule.magmom1)],
                            periodic=periodic,
                            cell=(a*0.9,a,a*1.2))
    else:
        #Create two atoms separated along x axis
        mag = molecule.magmom2/2. # magmom per atom
        atoms = ListOfAtoms([Atom(symbol, (cx+d,cy,cz), magmom=mag),
                             Atom(symbol, (cx-d,cy,cz), magmom=mag)],
                            periodic=periodic,                            
                            cell=(a,a,a))
    return atoms

def calc_energy(calc1=None, calc2=None, a=6., symbol = 'N',
               dislocation=(0,0,0), periodic=False, setup='paw'):
    """
    Calculates the atomization energy, i.e. E[molecule] - 2*E[single atom].
    """

    molecule = elements[symbol]

    one_atom = get_list_of_atoms(symbol, a=a, dislocation=dislocation,
                                 periodic=periodic)

    if calc1 == None:
        calc1 = makecalculator(nbands=molecule.nbands1, setup=setup)
    if calc2 == None:
        calc2 = makecalculator(nbands=molecule.nbands2, setup=setup)

    #bands: 2s and 2p yield a total of 4 bands; 1s is ignored
    #setups='A1' => will search for /home/ask/progs/gpaw/setups/N.A1.PBE.gz

    one_atom.SetCalculator(calc1)
    e1 = one_atom.GetPotentialEnergy()


    #gpts=(n,n,n) - to be varied in multiples of 4
    d = molecule.d

    two_atoms = get_list_of_atoms(symbol, a=a, dislocation=dislocation,
                                  periodic=periodic, separation=molecule.d)

    #10 electrons in total from 2s and 2p.
    #Thus it is necessary only to include 5 bands
    two_atoms.SetCalculator(calc2)
    e2 = two_atoms.GetPotentialEnergy()

    return e2-2*e1

def displacement_test(a=5., symbol='N', h=.2):
    """
    Using a particular resolution h, test whether energies deviate considerably
    if the system is translated in intervals smaller than h.

    This function is deprecated.
    """
    print 'Displacement test:', symbol

    molecule = elements[symbol]

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

def atomization_calculators(symbol='N', out='-', h=.2, setup='paw'):
    """
    Creates two calculators for the given molecule with appropriate band counts
    """
    molecule = elements[symbol]
    calc1 = makecalculator(molecule.nbands1, out, h, setup=setup)
    calc2 = makecalculator(molecule.nbands2, out, h, setup=setup)
    return calc1, calc2


def makecalculator(nbands=None, out='-', h=.2, setup='paw'):
    """
    Default calculator setup, however complicated it might become someday
    This method allows you to forget about lmax and PBE and such
    """
    return Calculator(nbands=nbands, txt=out, h=h, xc='PBE',
                      setups=setup)

def energy_at_distance(distance, calc=None, dislocation=(0,0,0),
                       symbol='N', a=5., periodic=False):
    """
    Calculates the ground-state energy of the given molecule when the atoms
    are spaced by the given distance
    """

    molecule = elements[symbol]

    c = a/2.
    (dx, dy, dz) = dislocation

    coord1 = (c-distance/2. + dx, c + dy, c + dz)
    coord2 = (c+distance/2. + dx, c + dy, c + dz)

    two_atoms = get_list_of_atoms(symbol, distance, a, dislocation,
                                  periodic)

    if calc == None:
        calc = makecalculator(nbands=molecule.nbands2)

    two_atoms.SetCalculator(calc)

    energy = two_atoms.GetPotentialEnergy()
    return energy

def writeresults(x, y, filename, header=[]):
    """
    Write lists x and y to specified file
    """
    if len(x) != len(y):
        raise Exception('Result list length mismatch')
    length = len(x)
    f = open(filename, 'w')
    lines = [''.join([str(x[i]),'\t',str(y[i]),'\n']) for i in range(length)]

    for line in header:
        line = '# '+line

    f.writelines(header)
    
    f.writelines(lines)
    f.close()

def readresults(filename):
    """
    Read list of (x,y) entries from datafiles, return as two lists
    """
    f = open(filename, 'r')
    lines = filter(stringfilter, f.readlines())
    length = len(lines)
    pairs = [s.split() for s in lines]
    x = [float(pair[0]) for pair in pairs]
    y = [float(pair[1]) for pair in pairs]
    return (x,y)

def stringfilter(s):
    """
    Allow comments and empty lines in data files
    """
    return not (s.startswith('#') or s.isspace())

def linspace(start, end, count):
    """
    The gbar doesn't have pylab so use this function
    """
    return [start + float(i)/(count-1)*(end-start) for i in range(count)]

