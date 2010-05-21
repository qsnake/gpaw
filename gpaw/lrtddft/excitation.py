"""Excitation lists base classes

"""
from math import sqrt

import numpy as np

import gpaw.mpi as mpi
from gpaw.output import initialize_text_stream

class ExcitationList(list):
    """General Excitation List class.

    """
    def __init__(self, calculator=None, txt=None):

        # initialise empty list
        list.__init__(self)

        self.calculator = calculator
        if not txt and calculator:
            txt = calculator.txt
        self.txt, firsttime = initialize_text_stream(txt, mpi.rank)

    def get_calculator(self):
        return self.calculator

    def get_energies(self):
        """Get excitation energies in Hartrees"""
        el = []
        for ex in self:
            el.append(ex.get_energy())
        return np.array(el)

    def get_trk(self):
        """Evaluate the Thonmas Reiche Kuhn sum rule"""
        trkm = np.zeros((3))
        for ex in self:
            trkm += ex.get_energy()*ex.get_dipol_me()**2
        return 2.*trkm # scale to get the number of electrons
    
    def get_polarizabilities(self, lmax=7):
        """Calculate the Polarisabilities
        see Jamorski et al. J. Chem. Phys. 104 (1996) 5134"""
        S = np.zeros((lmax+1))
        for ex in self:
            e = ex.get_energy()
            f = ex.get_oscillator_strength()[0]
            for l in range(lmax+1):
                S[l] += e**(-2 * l) * f
        return S

    def set_calculator(self, calculator):
        self.calculator = calculator

    def __str__(self):
        string = '# ' + str(type(self))
        if len(self) != 0:
            string += ', %d excitations:' % len(self)
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
    
    def get_dipol_me(self):
        """return the excitations dipole matrix element
        including the occupation factor"""
        return self.me / sqrt(self.energy)
    
    def get_oscillator_strength(self, form='r'):
        """Return the excitations dipole oscillator strength.


        self.me is assumed to be::

          form='r': sqrt(f * E) * <I|r|J>,
          form='v': sqrt(f / E) * <I|d/(dr)|J>

        for f = multiplicity, E = transition energy and initial and
        final states::
        
          |I>, |J>
          
        """
        
        if form == 'r':
            # length form
            me = self.me
        elif form == 'v':
            raise NotImplemented
            # velocity form
            me = self.muv

        osz = [0.]
        for c in range(3):
            val = 2. * me[c]**2
            osz.append(val)
            osz[0] += val / 3.
        
        return osz

    def set_energy(self, E):
        """Set the excitations energy relative to the ground state energy"""
        self.energy = E
    
