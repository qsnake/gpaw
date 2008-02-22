from math import pi
from pickle import load, dump

import numpy as npy
from ase.units import Bohr

from _gpaw import localize


class Wannier:
    def __init__(self, calc=None, spin=0):
        self.spin = spin
        if calc is not None:
            self.cell_c = calc.domain.cell_c * Bohr
            n = calc.get_number_of_bands()
            self.Z_nnc = npy.empty((n, n, 3), complex)
            for c in range(3):
                self.Z_nnc[:, :, c] = calc.get_wannier_integrals(c, spin,
                                                                #k, k1, G
                                                                 0, 0, 1.0)
            self.value = 0.0
            self.U_nn = npy.identity(n)

    def load(self, filename):
        self.cell_c, self.Z_nnc, self.value, self.U_nn = load(open(filename))

    def dump(self, filename):
        dump((self.cell_c, self.Z_nnc, self.value, self.U_nn), filename)
        
    def localize(self, eps=1e-5, iterations=-1):
        i = 0
        while i != iterations:
            value = localize(self.Z_nnc, self.U_nn)
            print i, value
            if value - self.value < eps:
                break
            i += 1
            self.value = value
        return value

    def get_centers(self):
        scaled_c = -npy.angle(self.Z_nnc.diagonal()).T / (2 * pi)
        return (scaled_c % 1.0) * self.cell_c

    def get_function(self, calc, n):
        psit_nG = calc.kpt_u[self.spin].psit_nG[:].reshape((calc.nbands, -1))
        return npy.dot(self.U_nn[:, n],
                       psit_nG).reshape(calc.gd.n_c) / Bohr**1.5

