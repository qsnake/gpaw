import sys
from math import exp, pi, sqrt

from ASE.Units import Convert
from gpaw.gauss import Gauss

def spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             folding='Gauss',
             width=0.08 # Gauss/Lorentz width in eV
             ):
    """Print the optical spectrum of an excitation list

    Parameters:
    =============== ===================================================
    ``exlist``      ExcitationList
    ``filename``    File name for the output file, STDOUT if not given
    ``emin``        min. energy, set to cover all energies if not given
    ``emax``        max. energy, set to cover all energies if not given
    ``de``          energy spacing
    ``folding``     Gauss (default) or Lorentz
    ``width``       folding width
    =============== ===================================================
    all energies in [eV]
    """

    func=Gauss(width)
    if folding == 'Lorentz':
        func=Lorentz(width)

    # output
    out = sys.stdout
    if filename != None:
        out = open( filename, 'w' )

    au_eV = Convert(1., 'Hartree', 'eV')

 ##   print "exlist=",exlist.GetEnergies()

    # minimal and maximal energies
    if emin == None:
        emin=exlist.GetEnergies()[0]*au_eV
        for e in exlist.GetEnergies():
            e*=au_eV
            if e<emin: emin=e
        emin -= 4*width
    if emax == None:
        emax=exlist.GetEnergies()[0]*au_eV
        for e in exlist.GetEnergies():
            e*=au_eV
            if e>emax: emax=e
        emax += 4*width

    # set de to sample 4 points in the width
    if de == None:
        de = width/4.

    print >> out, '# %s folded, width=%g [eV]' % (folding,width)
    print >> out, '# om [eV]     osz          osz x       osz y       osz z'

    # loop over energies
    emax=emax+.5*de
    e=emin
    while e<emax:
        val=[0.,0.,0.,0.]
        for ex in exlist:
            wght=func.Get(ex.GetEnergy()*au_eV-e)
            osz=ex.GetOszillatorStrength()
            for i in range(4):
                val[i]+=wght*osz[i]
        print >> out, "%10.5f %12.7f %12.7f %11.7f %11.7f" % \
              (e,val[0],val[1],val[2],val[3])
        e+=de
        
    if filename != None: out.close()

class Lorentz:
    """Normalised Lorentz distribution"""
    def __init__(self,width=0.08):
        self.SetWidth(width)
        
    def Get(self,x):
        return self.norm/(x**2+self.width2)
    
    def SetWidth(self,width=0.08):
        self.norm=width/pi
        self.width2=width**2

