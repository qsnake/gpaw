import Numeric as num
import _gridpaw
from gridpaw import debug

"""This module defines a linear response TDDFT-class."""

class LrTDDFT:
    """Linear Response TDDFT excitation class
    """
    
    def __init__(self,calculator=None):
        """
        Initialise Linear Response TDDFT excitations with it's
        calculator.
        """
        if calculator==None:
            raise RuntimeError('You have to set a calculator for Linear ' +
                               'Response TDDFT excitations')
        self.calculator = calculator
        print "<LrTDDFT::init> number of states: ",self.calculator.nbands
        print "<LrTDDFT::init> number of electrons: ",self.calculator.nbands

    def get_excitations(self):
        """Calcuate excitations from wave-functions and density"""

        # check if this is periodic
        if self.calculator.GetKPoints():
            print 'Linear Response TDDFT does not work for more that ' + \
                  'one k-point'
            raise NotImplementedError

        self.paw = self.calculator.paw
        self.wf = self.paw.wf

        self.lrt = LrTransitions(self.wf)

        print "<LrTDDFT::init> lrt=",self.lrt
        
#        self.density = self.calculator.GetDensityArray()
#        self.wf = self.calculator.GetWaveFunctionArray()

    #####################
    ## User interface: ##
    #####################
    def GetEnergies(self):
        """Calculate the linear response TDDFT excitation energies"""
        self.get_excitations()
        
    def GetOscillatorStrengths(self):
        """Calculate the linear response TDDFT oscillator strengths"""
        raise NotImplementedError
        
    def GetOpticalSpectrum(self):
        """Calculate the optical spectrum from the given excitations"""
        raise NotImplementedError

# ---------------------------------------------------------------------
# Helping classes

class LrTransitions:
    """Linear Response Transition objects

    Input parameters:
    wf     = the ground wave function object, needed for number of spins
             and occupations
    nspins = number of spins considered in the calculation
             Note: Valid only for unpolarised ground state calculation
    eps    = Minimal occupation difference for a transition
    """

    def __init__(self,
                 wf=None,
                 nspins=None,
                 eps=0.001):
        """
        Initialise Linear Response Transitions from occupation list.
        """
        if wf==None:
            raise RuntimeError('WaveFunction object is needed')
        self.wf = wf

        # vspin is the virtual spin of the wave functions
        # pspin is the physical spin in the Omega-matrix
        self.nvspins = self.wf.nspins
        self.npspins = self.wf.nspins
        if self.nvspins < 2:
            if nspins>self.nvspins:
                self.npspins = nspins

        # set transition array
        self.tr = []
        for ispin in range(self.npspins):
            vspin=ispin
            print "vspin=",vspin,"ispin=",ispin
            if self.nvspins<2:
                vspin=0
            f=wf.kpt_u[vspin].f_n
            print "f=",f
            for i in range(len(f)):
                for j in range(len(f)):
                    if (f[i]-f[j])>eps:
                        self.tr.append(LrTransition(i,j,ispin,vspin))
                                       
    def __str__(self):
        str = "<LrTransitions> npspins=%d, nvspins=%d" % \
              (self.npspins, self.nvspins)
        str = str + "\n ntrans=%d" % len(self.tr)
        return str

class LrTransition:
    """Single transition containing all it's indicees"""
    def __init__(self,iidx=None,jidx=None,pspin=None,vspin=None):
        self.i=iidx
        self.j=jidx
        self.pspin=pspin
        self.vspin=vspin


def d2Excdnsdnt(dup,ddn,ispin,kspin):
    """Second derivative of Exc polarised"""
    res=num.zeros(dup.shape,num.Float)
    _gridpaw.d2Excdnsdnt(dup, ddn, ispin, kspin, res)
    return res

def d2Excdn2(den):
    """Second derivative of Exc unpolarised"""
    res=num.zeros(den.shape,num.Float)
    _gridpaw.d2Excdn2(den, res)
    return res

