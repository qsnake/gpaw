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
	self.first_peak_energy=None


    def _calculate(self):
        
        e_m=self.c_m.get_potential_energy()
        e_d=self.c_d.get_potential_energy()

        if self.first_peak_energy==None:
            self.first_peak_energy=e_d-e_m

        ex_m=[]
    
        for kpt in self.c_m.wfs.kpt_u:
            for j in range(len(kpt.f_n)):
                ex_m+=[kpt.eps_n[j]* Ha]*int(kpt.f_n[j])

        ex_m.sort()
        self.be=[]

        for j in range(len(ex_m)):
            self.be.append(-ex_m[j]+ex_m[-1]+(self.first_peak_energy))

        self.f = [1] * len(self.be)
