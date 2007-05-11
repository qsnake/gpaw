import sys
from math import pi, sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
from gpaw import debug
from gpaw.utilities import pack,packed_index

from gpaw.io.plt import write_plt

# ..............................................................
# general excitation classes

class ExcitationList(list):
    """
    General Excitation List class
    """
    def __init__(self,calculator=None):

        # initialise empty list
        list.__init__(self)

        self.calculator = calculator
        if calculator is not None:
            self.out = calculator.out
            self.Ha = calculator.Ha
        else:
            self.out = sys.stdout

    def GetEnergies(self):
        el = []
        for ex in self:
            el.append(ex.GetEnergy()*self.Ha)
        return el

    def GetTRK(self):
        """Evaluate the Thonmas Reiche Kuhn sum rule"""
        trkm = num.zeros((3),num.Float)
        for ex in self:
            trkm += ex.GetEnergy()*ex.GetDipolME()**2
        return 2.*trkm # scale to get the number of electrons
    
    def GetPolarizabilities(self,lmax=7):
        """Calculate the Polarisabilities
        see Jamorski et al. J. Chem. Phys. 104 (1996) 5134"""
        S=num.zeros((lmax+1),num.Float)
        for ex in self:
            e = ex.GetEnergy()
            f = ex.GetOscillatorStrength()[0]
            for l in range(lmax+1):
                S[l] += e**(-2*l) * f
        return S

    def __str__(self):
        string= '# ' + str(type(self))
        if len(self) != 0:
            string+=', %d excitations:' % len(self)
        string += '\n'
        for ex in self:
            string += '#  '+ex.__str__()+"\n"
        return string
        
class Excitation:
    def GetEnergy(self):
        """return the excitations energy relative to the ground state energy"""
        return self.energy
    
    def GetDipolME(self):
        """return the excitations dipole matrix element
        including the occupation factor"""
        return self.me / sqrt(self.energy)
    
    def GetOscillatorStrength(self):
        """return the excitations oscillator strength"""
        me=self.me
        osz=[0.]
        for i in range(3):
            val=2.*me[i]**2
            osz.append( val )
            osz[0]+=val/3.
        return osz

    def SetEnergy(self,E):
        """return the excitations energy relative to the ground state energy"""
        self.energy = E
    
