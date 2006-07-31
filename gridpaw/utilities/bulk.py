from math import sqrt

from ASE import ListOfAtoms, Atom
from ASE.ChemicalElements import numbers
from ASE.ChemicalElements.crystal_structure import crystal_structures

from gridpaw import Calculator
from gridpaw.utilities import center
from gridpaw.utilities.singleatom import SingleAtom
from gridpaw.atom.generator import parameters as setup_parameters

data = {}
for symbol in setup_parameters:
    Z = numbers[symbol]
    X = crystal_structures[Z]
    structure = X['symmetry'].lower()
    if structure == 'cubic':
        structure = 'sc'
    s = {'sc': 1, 'bcc': 0.5, 'fcc': 0.25, 'hcp': sqrt(3) / 2,
         'diamond': 0.128}.get(structure)
    if s is not None and symbol is not 'P':
        a = X['a']
        coa = X.get('c/a', 1)
        V = a**3 * coa * s
        data[symbol] = {'structure': structure, 'volume': V}
        if coa != 1:
            data[symbol]['c/a'] = coa
        
data['Fe']['magmom'] = 2.2
#data['Co']['magmom'] = 1.5
data['Ni']['magmom'] = 0.5

class Bulk:
    
    def __init__(self, symbol, structure=None, a=None, c=None, magmom=None):
        self.symbol = symbol
        
        d = data.get(symbol, {})
        
        if structure is None:
            structure = d['structure']

        if magmom is None:
            magmom = d.get('magmom', 0)

        spos_ac = {'sc':       [(0, 0, 0)],
                   'bcc':      [(0, 0, 0), (.5, .5, .5)],
                   'fcc':      [(0, 0, 0), (.5, .5, .5)],
                   'hcp':      [(0, 0, 0), (.5, .5, 0),
                                (.5, 1/6., .5), (0, 2/3., .5)],
                   'diamond':  [(0, 0, 0), (0, .5, .5),
                                (.5, 0, .5), (.5, .5, 0),
                                (.25, .25, .25), (.25, .75, .75),
                                (.75, .25, .75), (.75, .75, .25)]}[structure]
        
        self.atoms = ListOfAtoms([Atom(symbol, spos_c, magmom=magmom)
                                  for spos_c in spos_ac],
                                 periodic=True)

        V = d.get('volume', 20.0)

        if structure == 'hcp':
            if c is None:
                coa = d.get('c/a', sqrt(8.0 / 3))
                if a is None:
                    a = (2 * V / coa / sqrt(3))**(1.0 / 3)
                c = coa * a
            else:
                if a is None:
                    a = sqrt(2 * V / c / sqrt(3))
            b = sqrt(3) * a
        elif structure == 'fcc':
            if a is None:
                a = (V * 4)**(1.0 / 3) / sqrt(2)
            b = a
            c = a * sqrt(2)
        else:
            if a is None:
                a = (V * len(spos_ac))**(1.0 / 3)
            b = a
            c = a
            
        self.atoms.SetUnitCell([a, b, c])

    def energy(self, h=None, gpts=None, kpts=None, parameters={}):
        cell = self.atoms.GetUnitCell()
        if kpts is None:
            kpts = [2 * int(8.0 / cell[c, c]) for c in range(3)]

        calc = Calculator(h=h, gpts=gpts, kpts=kpts, **parameters)
        self.atoms.SetCalculator(calc)
        
        e = self.atoms.GetPotentialEnergy()
        m = self.atoms.GetCalculator().GetMagneticMoment()
        return e, m

