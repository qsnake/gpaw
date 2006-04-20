from math import sqrt, pi, cos, sin

from ASE import  ListOfAtoms, Atom

from gridpaw import Calculator
from gridpaw.utilities import center
from gridpaw.setuptests.singleatom import SingleAtom


class D(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)

    def __getattr__(self, key):
        return self[key]
        
data = {'Fe': D(structure='bcc', magmom=2.2, volume=2.89**3/2),
        }

class Bulk:
    
    def __init__(self, symbol, structure=None, a=None, c=None,
                 h=None, kpts=None, magmom=None,
                 parameters={}):
        self.symbol = symbol
        
        d = data.get(symbol, {})
        
        if structure is None:
            structure = d.structure

        if magmom is None:
            magmom = d.get('magmom', 0)

        spos_ac = {'sc':      [(0, 0, 0)],
                   'bcc':     [(0, 0, 0), (.5, .5, .5)],
                   'fcc':     [(0, 0, 0), (0, .5, .5),
                               (.5, 0, .5), (.5, .5, 0)],
                   'diamond': [(0, 0, 0), (0, .5, .5),
                               (.5, 0, .5), (.5, .5, 0),
                               (.25, .25, .25), (.25, .75, .75),
                               (.75, .25, .75), (.75, .75, .25)]}[structure]
        
        self.atoms = ListOfAtoms([Atom(symbol, spos_c, magmom=magmom)
                                  for spos_c in spos_ac],
                                 periodic=True)

        V = d.volume

        if structure == 'hcp':
            if c is None:
                coa = d.get('coa', sqrt(8.0 / 3))
                if a is None:
                    a = (V / coa / sqrt(3))**(1.0 / 3)
                c = coa * a
            else:
                if a is None:
                    a = sqrt(V / c / sqrt(3))
            b = sqrt(3) * a
        elif structure == 'hex':
            raise NotImplementedError
        else:
            if a is None:
                a = (V * len(spos_ac))**(1.0 / 3)
            b = a
            c = a
            
        self.atoms.SetUnitCell([a, b, c])

        if kpts is None:
            kpts = [2 * int(8.0 / L) for L in (a, b, c)]
            
        calc = Calculator(h=h, kpts=kpts, **parameters)
        self.atoms.SetCalculator(calc)
        
    def energy(self):
        return self.atoms.GetPotentialEnergy()

    def atomize(self, L=7.0, verbose=False):
        ec = -self.energy()
        if verbose:
            print '%s: %.3f eV' % (self.symbol, -ec)
        energy = {}
        h = num.sum(self.atoms.GetCalculator().GetGridSpacings()) / 3
        a = 4 * int(L / h / 4 + 0.5)
        self.atoms.SetCalculator(None)
        for atom in self.atoms:
            symbol = atom.GetChemicalSymbol()
            if symbol not in energy:
                if verbose:
                    print '%s:' % symbol,
                atom = SingleAtom(symbol, a=a, h=h,
                                  parameters=self.parameters)
                energy[symbol] = atom.energy()
                if verbose:
                    print '%.3f eV' % energy[symbol]
            ec += energy[symbol]
        return ec
