import sys
import numpy as npy
from math import exp, pi, sqrt

from ase import Hartree as Ha
from gpaw.gauss import Gauss, Lorentz

def spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             energyunit='eV',
             folding='Gauss',
             width=0.08, # Gauss/Lorentz width
             comment=None
             ):
    """spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             energyunit='eV',
             folding='Gauss',
             width=0.08 # Gauss/Lorentz width
             )
    Print the optical spectrum of an excitation list

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
    if folding == 'Gauss':
        func=Gauss(width)
    elif folding == 'Lorentz':
        func=Lorentz(width)
    elif folding is None:
        func=None
    else:
        raise RuntimeError('unknown folding "'+folding+'"')
    
    if energyunit != 'eV':
        raise RuntimeError('currently only eV supported')
 
    # output
    out = sys.stdout
    if filename != None:
        out = open( filename, 'w' )
    if comment:
        print >> out, '#', comment

    # energy unit

    # minimal and maximal energies
    if emin == None:
        emin = min(exlist.get_energies().tolist()) *  Ha
        emin -= 4*width
    if emax == None:
        emax = max(exlist.get_energies().tolist()) *  Ha
        emax += 4*width

    # set de to sample 4 points in the width
    if de == None:
        de = width/4.

    if func is not None: # fold the spectrum
        
        print >> out, '# %s folded, width=%g [%s]' % (folding,width,energyunit)
        print >> out,\
              '# om [%s]     osz          osz x       osz y       osz z'\
              % energyunit

        # loop over energies
        emax=emax+.5*de
        e=emin
        while e<emax:
            val=npy.zeros((4))
            for ex in exlist:
                wght=func.get(ex.get_energy()*Ha-e)
                osz=npy.array(ex.GetOscillatorStrength())
                val += wght*osz
            print >> out, "%10.5f %12.7e %12.7e %11.7e %11.7e" % \
                  (e,val[0],val[1],val[2],val[3])
            e+=de

    else: # just list energies and oszillator strengths

        print >> out,\
              '# om [%s]     osz          osz x       osz y       osz z'\
              % energyunit
        for ex in exlist:
            e=ex.get_energy()*Ha
            val=ex.GetOscillatorStrength()
            print >> out, "%10.5f %12.7e %12.7e %11.7e %11.7e" % \
                  (e,val[0],val[1],val[2],val[3])
        
    if filename != None: out.close()


