from math import pi
from pickle import load, dump

import numpy as npy
from ase.units import Bohr

from _gpaw import localize


class Wannier:
    def __init__(self, calc=None):
        if calc is not None:
            self.cell = calc.domain.cell_c * Bohr
            n = calc.get_number_of_bands()
            self.Z = npy.empty((n, n, 3), complex)
            for c in range(3):
                self.Z[:, :, c] = calc.get_wannier_integrals(c, 0, 0, 0, 1.0)
            self.value = 0.0
            self.U = npy.identity(n)

    def load(self, filename):
        self.cell, self.Z, self.value, self.U = load(open(filename))

    def dump(self, filename):
        dump((self.cell, self.Z, self.value, self.U), filename)
        
    def localize(self, eps=1e-5, iterations=-1):
        i = 0
        while i != iterations:
            value = localize(self.Z, self.U)
            print i, value
            if value - self.value < eps:
                break
            i += 1
            self.value = value
        return value

    def get_centers(self):
        scaled = -npy.angle(self.Z.diagonal()).T / (2 * pi)
        return (scaled % 1.0) * self.cell

    def get_function(self, calc, n):
        psit_nG = calc.kpt_u[0].psit_nG[:].reshape((calc.nbands, -1))
        return npy.dot(self.U[:, n],  psit_nG).reshape(calc.gd.n_c) / Bohr**1.5
