from gpaw.pes import BasePES
from ase import Hartree as Ha
from ase import *
from gpaw import *
import numpy as np

class DOSPES(BasePES):
    def __init__(self, Mother, Daughter):
        self.c_m=Mother
        self.c_d=Daughter
        self.f=None
        self.be=None


    def _calculate(self):

        e_m=self.c_m.get_potential_energy()
        e_d=self.c_d.get_potential_energy()

        ex_m=[]
        occ_m=0
    
        for kpt in self.c_m.wfs.kpt_u:
            ex_m+=list(kpt.eps_n * Ha)
            occ_m+=kpt.f_n.sum()

        if len(self.c_m.wfs.kpt_u)==1:
            ex_m+=ex_m

        ex_m.sort()
        occ_m=int(occ_m)

        self.be=[]

        for j in range(occ_m):
            self.be.append(-ex_m[j]+ex_m[occ_m-1]+(e_d-e_m))

        self.f = [1] * len(self.be)
