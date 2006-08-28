from math import sqrt, pi, cos, sin

from ASE import  ListOfAtoms, Atom

from gpaw import Calculator
from gpaw.utilities import center
from gpaw.utilities.singleatom import SingleAtom

class Molecule:
    def __init__(self, formula, a=None, b=None, c=None, h=None, parameters={}):
        self.formula = formula
        self.parameters = parameters
        if a is None:
            a = 7.0  # Angstrom
        if b is None:
            b = a
        if c is None:
            c = a
        self.a = a
        self.b = b
        self.c = c
        
        self.atoms = molecules[formula].Copy()
        self.atoms.SetUnitCell([a, b, c], fix=True)
        
        calc = Calculator(h=h, **parameters)
        self.atoms.SetCalculator(calc)
        center(self.atoms)

    def energy(self):
        return self.atoms.GetPotentialEnergy()
        
    def non_self_xc(self, xcs=[]):
        return [self.atoms.GetCalculator().GetXCDifference(xc) for xc in xcs]

    def relax(self, fmax=0.05, verbose=False):
        from ASE.Dynamics.QuasiNewton import QuasiNewton
        old_pos = self.atoms.GetCartesianPositions()
        qn = QuasiNewton(self.atoms, fmax=fmax)
        qn.Converge()
        if verbose:
            new_pos = self.atoms.GetCartesianPositions()
            print 'Relaxed coordinates for %s from\n%s\nto\n%s\n'\
                  'in %s function evaluations.'%(
                self.formula, old_pos, new_pos, qn.GetNumberFuncEvals())
            
    def atomize(self, verbose=False, atom_energies=None, xcs=[]):
        ea = [-self.energy()] + [-xcd for xcd in self.non_self_xc(xcs)]
        if verbose:
            print '%s: %.3f eV' % (self.formula, -ea[0])
        if atom_energies is None:
            atom_energies = {}
        h = self.atoms.GetCalculator().GetGridSpacings()[0]
        self.atoms.SetCalculator(None)
        for atom in self.atoms:
            symbol = atom.GetChemicalSymbol()
            if symbol not in atom_energies:
                if verbose:
                    print '%s:' % symbol,
                atom = SingleAtom(symbol, a=self.a, b=self.b, c=self.c, h=h,
                                  parameters=self.parameters)
                atom_energies[symbol] = [atom.energy()] + atom.non_self_xc(xcs)
                if verbose:
                    print '%.3f eV' % atom_energies[symbol][0]
            for i in range(len(ea)):
                ea[i] += atom_energies[symbol][i]
        if xcs == []: return ea[0]
        else: return ea
                
# Diatomic molecules: (Angstrom units)
H2 = ListOfAtoms([Atom('H', (0, 0, 0)),
                  Atom('H', (0.7414, 0, 0))])
HF = ListOfAtoms([Atom('H', (0, 0, 0)),
                  Atom('F', (0.9169, 0, 0))])
OH = ListOfAtoms([Atom('O', (0, 0, 0), magmom=0.5),
                  Atom('H', (1.0, 0, 0), magmom=0.5)]) # ??????????
CO = ListOfAtoms([Atom('C', (0, 0, 0)),
                  Atom('O', (1.1283, 0, 0))])
C2 = ListOfAtoms([Atom('C', (0, 0, 0)),
                  Atom('C', (1.0977, 0, 0))])
N2 = ListOfAtoms([Atom('N', (0, 0, 0)),
                  Atom('N', (1.0977, 0, 0))])
NO = ListOfAtoms([Atom('N', (0, 0, 0), magmom=0.5),
                  Atom('O', (1.1506, 0, 0), magmom=0.5)])
O2 = ListOfAtoms([Atom('O', (0, 0, 0), magmom=1),
                  Atom('O', (1.2074, 0, 0), magmom=1)])
F2 = ListOfAtoms([Atom('F', (0, 0, 0)),
                  Atom('F', (1.4119, 0, 0))])
P2 = ListOfAtoms([Atom('P', (0, 0, 0)),
                  Atom('P', (1.8931, 0, 0))])
LiH = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('H', (1.5949, 0, 0))])
Li2 = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('Li', (2.6729, 0, 0))])
Be2 = ListOfAtoms([Atom('Be', (0, 0, 0)),
                   Atom('Be', (2.45, 0, 0))])
Cl2 = ListOfAtoms([Atom('Cl', (0, 0, 0)),
                   Atom('Cl', (1.9878, 0, 0))])
LiF = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('F', (1.5639, 0, 0))])
Cl2 = ListOfAtoms([Atom('Cl', (0, 0, 0)),
                   Atom('Cl', (1.9878, 0, 0))])

# Methane molecule
x = 1.087 / sqrt(3)
CH4 = ListOfAtoms([Atom('C', (0, 0, 0)),
                   Atom('H', (x, x, x)),
                   Atom('H', (-x, -x, x)),
                   Atom('H', (x, -x, -x)),
                   Atom('H', (-x, x, -x))])

# Ammonia molecule
d = 1.012
t = pi / 180 * 106.7
x = 2 * d / sqrt(3) * sin(t / 2)
z = sqrt(d**2 - x**2)
NH3 = ListOfAtoms([Atom('N', (0, 0, 0)),
                   Atom('H', (x, 0, z)),
                   Atom('H', (-x / 2, sqrt(3) * x / 2, z)),
                   Atom('H', (-x / 2, -sqrt(3) * x / 2, z))])

# Water molecule
d = 0.9575
t = pi / 180 * 104.51
H2O = ListOfAtoms([Atom('O', (0, 0, 0)),
                   Atom('H', (d, 0, 0)),
                   Atom('H', (d * cos(t), d * sin(t), 0))])

# Acetylene molecule
d1 = 1.203
d2 = 1.060
C2H2 = ListOfAtoms([Atom('C', (0, 0, 0)),
                    Atom('C', (d1, 0, 0)),
                    Atom('H', (-d2, 0, 0)),
                    Atom('H', (d1 + d2, 0, 0))])

# Ethylene molecule
d1 = 1.339
d2 = 1.087
t = pi / 180 * 121.3
C2H4 = ListOfAtoms([Atom('C', (0, 0, 0)),
                    Atom('C', (d1, 0, 0)),
                    Atom('H', (d2 * cos(t), d2 * sin(t), 0)),
                    Atom('H', (d2 * cos(t), -d2 * sin(t), 0)),
                    Atom('H', (d1 - d2 * cos(t), d2 * sin(t), 0)),
                    Atom('H', (d1 - d2 * cos(t), -d2 * sin(t), 0))])

# HCN molecule
d1 = 1.0655
d2 = 1.1532
HCN = ListOfAtoms([Atom('H', (-d1, 0, 0)),
                   Atom('C', (0, 0, 0)),
                   Atom('N', (d2, 0, 0))])

molecules = {}
for m in ['H2', 'LiH', 'CH4', 'NH3', 'OH',
          'H2O', 'HF', 'Li2', 'LiF', 'Be2',
          'C2H2', 'C2H4', 'HCN', 'CO', 'N2',
          'NO', 'O2', 'F2', 'P2', 'Cl2']:
    molecules[m] = eval(m)


