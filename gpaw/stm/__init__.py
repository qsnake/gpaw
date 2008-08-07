import numpy as np
from ase.units import Bohr

from gpaw import Calculator
from gpaw.localized_functions import create_localized_functions as clf


class STM:
    def __init__(self, surfacecalc, tipcalc):
        self.scalc = surfacecalc
        self.tcalc = tipcalc
        self.surface = surfacecalc.get_atoms()
        self.tip = tipcalc.get_atoms()

    def initialize(self):
        tip = self.tip.copy()
        htip = tip.get_cell()[2, 2]
        #tip.translate((0, 0, htip))
        self.combined = self.surface + tip
        self.combined.cell[2, 2] += htip
        
        self.calc = Calculator(h=0.2, eigensolver='lcao', basis='sz',
                               txt=None)
        #self.combined.set_calculator(self.calc)
        self.calc.initialize(self.combined)
        self.calc.hamiltonian.initialize(self.calc)
        self.calc.density.initialize()

        self.vtip_G = self.tcalc.hamiltonian.vt_sG[0]
        self.vsurface_G = self.scalc.hamiltonian.vt_sG[0]

        self.get_basis_functions()
        
        self.tgd = self.tcalc.gd
        self.sgd = self.scalc.gd

    def get_basis_functions(self):
        gd = self.calc.gd
        self.functions = []
        for nucleus in self.calc.nuclei:
            spos0_c = np.round(nucleus.spos_c * gd.N_c) / gd.N_c
            f_iG = clf(nucleus.setup.phit_j, gd,
                       nucleus.spos_c + 0.5 - spos0_c,
                       dtype=float, cut=False,
                       forces=False, lfbc=None).box_b[0].get_functions()
            self.functions.append((spos0_c, f_iG))
            
    def set_position(self, dG):
        positions = self.combined.get_positions()
        tippositions = self.tip.get_positions()
        tippositions += dG * self.calc.gd.h_c / Bohr
        positions[-len(self.tip):] = tippositions
        self.combined.set_positions(positions)

        self.calc.set_positions(self.combined)
        self.calc.hamiltonian.initialize_lcao()

    
