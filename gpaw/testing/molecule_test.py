#!/usr/bin/env python

"""Module for making tests on small molecules in GPAW.

One molecule test to rule them all
One molecule test to run them
One molecule test to save them all
And on the webpage plot them (implementation pending)
"""

from gpaw import GPAW, ConvergenceError
from ase.data.molecules import g1, atoms, molecule
from ase.utils.molecule_test import MoleculeTest, EnergyTest, BondLengthTest,\
     BatchTest

class GPAWMoleculeTest(MoleculeTest):
    def __init__(self, name='gpaw', vacuum=6.0, h=0.17, xc='LDA',
                 setups='paw', eigensolver='rmm-diis', basis=None,
                 exceptions=(RuntimeError, ConvergenceError)):
        MoleculeTest.__init__(self, name=name, vacuum=vacuum,
                              exceptions=exceptions)
        if basis is None:
            basis = {}
        self.basis = basis
        self.eigensolver=eigensolver
        self.setups = setups
        self.h = h
        self.xc = xc
        self.bad_formulas = ['NO', 'ClO', 'CH']

    def setup_calculator(self, system, formula):
        hund = (len(system) == 1)
        cell = system.get_cell()
        h = self.h
        system.set_cell((cell / (4 * h)).round() * 4 * h)
        system.center()
        calc = GPAW(xc=self.xc,
                    h=h,
                    hund=hund,
                    fixmom=True,
                    setups=self.setups,
                    txt=self.get_filename(formula, extension='txt'),
                    eigensolver=self.eigensolver,
                    basis=self.basis
                    )

        # Special cases
        if formula == 'BeH':
            calc.set(idiotproof=False)
            #calc.initialize(system)
            #calc.nuclei[0].f_si = [(1, 0, 0.5, 0, 0),
            #                       (0.5, 0, 0, 0, 0)]

        if formula in self.bad_formulas:
            system.positions[:, 1] += h * 1.5
        
        return calc


class GPAWEnergyTest(EnergyTest, GPAWMoleculeTest):
    pass


class GPAWBondLengthTest(BondLengthTest, GPAWMoleculeTest):
    pass


def main():
    formulas = g1 + atoms
    dimers = [formula for formula in g1 if len(molecule(formula)) == 2]

    kwargs = dict(vacuum=3.0,
                  eigensolver='lcao',
                  basis='dzp')
    etest = BatchTest(GPAWEnergyTest('test/energy', **kwargs))
    btest = BatchTest(GPAWBondLengthTest('test/bonds', **kwargs))

    etest.run(formulas)
    btest.run(dimers)

if __name__ == '__main__':
    main()
