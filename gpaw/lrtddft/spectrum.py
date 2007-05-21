import sys
import Numeric as num
from math import exp, pi, sqrt

from ASE.Units import Convert

def spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             energyunit='eV',
             folding='Gauss',
             width=0.08 # Gauss/Lorentz width
             ):
    """Print the optical spectrum of an excitation list

    Parameters:
    =============== ===================================================
    ``exlist``      ExcitationList
    ``filename``    File name for the output file, STDOUT if not given
    ``emin``        min. energy, set to cover all energies if not given
    ``emax``        max. energy, set to cover all energies if not given
    ``de``          energy spacing
    ``energyunit``  Energy unit, default 'eV'
    ``folding``     Gauss (default) or Lorentz
    ``width``       folding width in terms of the chosen energyunit
    =============== ===================================================
    all energies in [eV]
    """

    # initialise the folding function
    func=Gauss(width)
    if folding == 'Lorentz':
        func=Lorentz(width)

    # output
    out = sys.stdout
    if filename != None:
        out = open( filename, 'w' )

    # energy unit
    Ha = Convert(1., 'Hartree', energyunit)

    # minimal and maximal energies
    if emin == None:
        emin=exlist.get_energies()[0]*Ha
        emax=emin
    for e in exlist.get_energies():
        e*=Ha
        if e<emin: emin=e
        if e>emax: emax=e
    emin -= 4*width
    emax += 4*width

    # set de to sample 4 points in the width
    if de == None:
        de = width/4.

    print >> out, '# %s folded, width=%g [%s]' % (folding,width,energyunit)
    print >> out, '# om [%s]     osz          osz x       osz y       osz z'\
          % energyunit

    # loop over energies
    emax=emax+.5*de
    e=emin
    while e<emax:
        val=num.zeros((4),num.Float)
        for ex in exlist:
            wght=func.Get(ex.get_energy()*Ha-e)
            osz=num.array(ex.GetOscillatorStrength())
            val += wght*osz
        print >> out, "%10.5f %12.7e %12.7e %11.7e %11.7e" % \
              (e,val[0],val[1],val[2],val[3])
        e+=de
        
    if filename != None: out.close()

class Gauss:
    """Normalised Gauss distribution"""
    def __init__(self,width=0.08):
        self.SetWidth(width)
        
    def Get(self,x):
        return self.norm*exp(-(x*self.wm1)**2)
    
    def SetWidth(self,width=0.08):
        self.norm=1./width/sqrt(pi)
        self.wm1=sqrt(.5)/width
    

class Lorentz:
    """Normalised Lorentz distribution"""
    def __init__(self,width=0.08):
        self.SetWidth(width)
        
    def Get(self,x):
        return self.norm/(x**2+self.width2)
    
    def SetWidth(self,width=0.08):
        self.norm=width/pi
        self.width2=width**2

