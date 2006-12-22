import sys

import Numeric as num
from ASE.Units import units
from ASE import Atom, ListOfAtoms

from gpaw.atom.configurations import configurations
from gpaw import Calculator


assert units.GetLengthUnit() == 'Ang' and units.GetEnergyUnit() == 'eV'

# Special cases:
magmoms = {}


class SingleAtom:
    def __init__(self, symbol, a=None, b=None, c=None,
                 h=None, gpts=None, spinpaired=False,
                 eggboxtest=False, parameters={}, forcesymm=True):
        if a is None:
            a = 7.0  # Angstrom

        if b is None:
            b = a
        if c is None:
            c = a

        if eggboxtest:
            spinpaired = True

        periodic = eggboxtest
        pos = (a / 2, b / 2, c / 2)

        parameters = parameters.copy()
        
        if spinpaired:
            magmom = 0
            width = 0.1  # 0.1 eV
            hund = False
        else:
            width = 0
            hund = True
            if symbol in ['C', 'O', 'F', 'Cl']:
                if forcesymm:
                    parameters['kpts'] = (2, 2, 2)
                    periodic = True
                    if symbol == 'O':
                        pos = (0, 0.1, 0)
                    else:
                        pos = (0.1, 0, 0)
                else:
                    tol = parameters.get('tolerance', 1e-99)
                    parameters['tolerance'] = max(1e-6, tol)

            # Is this a special case?
            magmom = magmoms.get(symbol)
            if magmom is None:
                # No.  Use Hund's rule:
                magmom = 0
                for n, l, f, e in configurations[symbol][1]:
                    magmom += min(f, 2 * (2 * l + 1) - f)
                    
        self.atom = ListOfAtoms([Atom(symbol, pos, magmom=magmom)],
                                periodic=periodic,
                                cell=[a, b, c])
        
        calc = Calculator(h=h, gpts=gpts, width=width, hund=hund, **parameters)
        self.atom.SetCalculator(calc)
        
    def energy(self):
        return self.atom.GetPotentialEnergy()

    def non_self_xc(self, xcs=[]):
        return [self.atom.GetCalculator().GetXCDifference(xc) for xc in xcs]

    def eggboxtest(self, N=30, verbose=False):
        X = num.zeros(N + 1, num.Float)
        e = num.zeros(N + 1, num.Float)
        dedx = num.zeros(N + 1, num.Float)
        self.atom[0].SetCartesianPosition([0, 0, 0])
        self.energy()
        h = self.atom.GetCalculator().GetGridSpacings()[0]
        for g in range(-2, N + 1):
            if verbose:
                sys.stderr.write('.')
            x = g * h / 2 / N
            self.atom[0].SetCartesianPosition([x, 0, 0])
            energy = self.energy()
            forces = self.atom.GetCartesianForces()
            # The two first points are only for warm up.
            if g >= 0:
                X[g] = x
                e[g] = energy
                dedx[g] = -forces[0, 0]
        if verbose:
            sys.stderr.write('\n')
            
        return X, e, dedx
