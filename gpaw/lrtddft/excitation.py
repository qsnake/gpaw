import sys
from math import pi, sqrt
import numpy as npy
import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER
from gpaw import debug
from gpaw.utilities import pack,packed_index

#from gpaw.io.plt import write_plt

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
            self.out = calculator.txt
            self.Ha = calculator.Ha
            # initialize the nuclei if not ready
            if not calculator.nuclei[0].ready:
                calculator.set_positions()
        else:
            if mpi.rank != MASTER: self.out = DownTheDrain()
            else: self.out = sys.stdout

    def get_energies(self):
        """Get excitation energies in Hartrees"""
        el = []
        for ex in self:
            el.append(ex.get_energy())
        return npy.array(el)

    def GetEnergies(self):
        """Get excitation energies in units of the calculators energy unit"""
        return self.get_energies()*self.Ha

    def GetTRK(self):
        """Evaluate the Thonmas Reiche Kuhn sum rule"""
        trkm = npy.zeros((3))
        for ex in self:
            trkm += ex.get_energy()*ex.GetDipolME()**2
        return 2.*trkm # scale to get the number of electrons
    
    def GetPolarizabilities(self,lmax=7):
        """Calculate the Polarisabilities
        see Jamorski et al. J. Chem. Phys. 104 (1996) 5134"""
        S=npy.zeros((lmax+1))
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
    def get_energy(self):
        """Get the excitations energy relative to the ground state energy
        in Hartrees.
        """
        return self.energy
    
    def GetDipolME(self):
        """return the excitations dipole matrix element
        including the occupation factor"""
        return self.me / sqrt(self.energy)
    
    def GetOscillatorStrength(self):
        return self.get_oscillator_strength()

    def get_oscillator_strength(self):
        """Return the excitations dipole oscillator strength.

        self.me is assumed to be::

          sqrt(f*E) * <I|r|J>,

        for f = multiplicity, E = transition energy and initial and
        final states::
        
          |I>, |J>
          
        """
        
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
    
