from math import sqrt, sin, cos, pi

from ase.atoms import Atom, Atoms
from ase.data import atomic_numbers as numbers
from ase.data import reference_states as crystal_structures
from ase.data import covalent_radii

from gpaw import Calculator
from gpaw.atom.generator import parameters as setup_parameters

data = {}
for symbol in setup_parameters:
    Z = numbers[symbol]
    X = crystal_structures[Z]
    structure = X['symmetry'].lower()
    if structure == 'cubic':
        structure = 'sc'
    s = {'sc': 1, 'bcc': 0.5, 'fcc': 0.25, 'hcp': sqrt(3) / 4,
         'diamond': 0.125}.get(structure)
    if s is not None and symbol != 'P':
        a = X['a']
        coa = X.get('c/a', 1)
        V = a**3 * coa * s
        data[symbol] = {'structure': structure, 'volume': V}
        if coa != 1:
            data[symbol]['c/a'] = coa
    elif structure == 'diatom':
        d = max(1.5, X['d'])
        data[symbol] = {'structure': 'sc', 'volume': d**3}
    elif structure in ['atom', 'orthorhombic', 'cubic']:
        d = 2.5 * covalent_radii[Z]
        data[symbol] = {'structure': 'sc', 'volume': d**3}
    elif symbol in ['As', 'Sb', 'Bi']:
        a = X['a']
        t = X['alpha'] * pi / 180
        V = a**3 * sin(t) * sqrt(1 - (cos(t) / cos(t / 2))**2) / 2
        data[symbol] = {'structure': 'sc', 'volume': V}

data['Fe']['magmom'] = 2.2
#data['Co']['magmom'] = 1.5
data['Ni']['magmom'] = 0.6

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
                   'diamond':  [(0, 0, 0), (.25, .5, 0),
                                (.5, .5, .5), (.75, 0, .5)]}[structure]

        self.atoms = Atoms([Atom(symbol, spos_c, magmom=magmom)
                            for spos_c in spos_ac],
                           pbc=True)

        V = d.get('volume', 20.0)

        natoms = len(spos_ac)

        if structure == 'hcp':
            if c is None:
                coa = d.get('c/a', sqrt(8.0 / 3))
                if a is None:
                    a = (4 * V / coa / sqrt(3))**(1.0 / 3)
                c = coa * a
            else:
                if a is None:
                    a = sqrt(4 * V / c / sqrt(3))
            b = sqrt(3) * a
        elif structure in ['fcc', 'diamond']:
            if a is None:
                a = (V * 2 * natoms)**(1.0 / 3)
            b = a / sqrt(2)
            c = b
        else:
            if a is None:
                a = (V * natoms)**(1.0 / 3)
            b = a
            c = a

        self.atoms.set_cell([a, b, c], scale_atoms=True)

    def energy(self, h=None, gpts=None, kpts=None, parameters={}):
        cell = self.atoms.get_cell()
        if kpts is None:
            kpts = [2 * int(8.0 / cell[c, c]) for c in range(3)]

        calc = Calculator(h=h, gpts=gpts, kpts=kpts, **parameters)
        self.atoms.set_calculator(calc)

        e = self.atoms.get_potential_energy()
        m = self.atoms.get_calculator().get_magnetic_moments()
        return e, m
