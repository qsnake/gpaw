import Numeric as num
from RandomArray import random

from gridpaw.transformers import Interpolator


class RandomWaveFunctionGenerator:
    def __init__(self, gd0, typecode):
        self.typecode = typecode
        gd1 = gd0.coarsen()
        gd2 = gd1.coarsen()
        self.n2_c = gd2.n_c
        self.r1 = gd1.new_array(typecode=typecode)
        self.r2 = gd2.new_array(typecode=typecode)
        self.interpolate2 = Interpolator(gd2, 1, typecode).apply
        self.interpolate1 = Interpolator(gd1, 1, typecode).apply
        
    def generate(self, psit_G, phase_cd):
        if self.typecode == num.Float:
            self.r2[:] = random(self.n2_c) - 0.5
        else:
            self.r2.real = random(self.n2_c) - 0.5
            self.r2.imag = random(self.n2_c) - 0.5
        self.interpolate2(self.r2, self.r1, phase_cd)
        self.interpolate1(self.r1, psit_G, phase_cd)
