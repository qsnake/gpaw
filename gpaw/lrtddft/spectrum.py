import sys
import numpy as np
from math import exp, pi, sqrt

from ase import Hartree as Ha
from gpaw.gauss import Gauss, Lorentz

from gpaw.version import version

def spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             energyunit='eV',
             folding='Gauss',
             width=0.08, # Gauss/Lorentz width
             comment=None,
             title='Photoabsorption spectrum from linear response TD-DFT'
             ):
    """Write out a folded spectrum.

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

    print >> out, '# Photoabsorption spectrum from linear response TD-DFT'
    print >> out, '# GPAW version:', version

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
            val=np.zeros((4))
            for ex in exlist:
                wght=func.get(ex.get_energy()*Ha-e)
                osz=np.array(ex.GetOscillatorStrength())
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


from gpaw.version import version
from gpaw.gauss import Gauss, Lorentz

def fold(energies, values, 
         emin=None, emax=None, de=None,
         folding='Gauss', width=0.08, # Gauss/Lorentz width
         ):

    if folding == 'Gauss':
        func = Gauss(width)
    elif folding == 'Lorentz':
        func = Lorentz(width)
    else:
        raise RuntimeError('unknown folding "'+folding+'"')

    # minimal and maximal energies
    if emin == None:
        emin = min(energies)
        emin -= 4 * width
    if emax == None:
        emax = max(energies)
        emax += 4 * width
    if de == None:
        # set de to sample 4 points in the width
        de = width / 4.
    print "de=", de

    print values.shape, len(values.shape)
    if len(values.shape) < 2:
        vallists = values.reshape((len(values),1))
    else:
        vallists = values
 
    el = []
    vl = []

    e = emin
    while e < emax:
        folded = np.zeros(len(vallists[0]))
        
        for ene, val in zip(energies, vallists):
            wght = func.get(ene - e)
            folded += wght * np.array(val)
            
        el.append(e)
        vl.append(folded)
        e += de
            
    return np.array(el), np.array(vl)
     
class Writer:
    def __init__(self, folding=None, width=0.08, # Gauss/Lorentz width
                 ):

        self.folding = folding
        if folding == 'Gauss':
            self.func = Gauss(width)
        elif folding == 'Lorentz':
            self.func = Lorentz(width)
        elif folding is None:
            self.func=None
        else:
            raise RuntimeError('unknown folding "'+folding+'"')

        self.width = width

    def write(self, filename=None, emin=None, emax=None, de=None,
              comment=None):

        out = sys.stdout
        if filename != None:
            out = open( filename, 'w' )
 
        print >> out, '#', self.title
        print >> out, '# GPAW version:', version
        if comment:
            print >> out, '#', comment

        if self.func is not None: # fold the spectrum
        
             # minimal and maximal energies
            if emin == None:
                emin = min(self.energies)
                emin -= 4 * self.width
            if emax == None:
                emax = max(self.energies)
                emax += 4 * self.width
            if de == None:
                # set de to sample 4 points in the width
                de = self.width / 4.

            print >> out, '# %s folded, width=%g [eV]' % (self.folding,
                                                          self.width)
            print >> out, '#', self.fields

            # loop over energies
            e = emin
            while e < emax:
                folded = np.zeros((len(self.values[0])))
                               
                for ene, val in zip(self.energies, self.values):
                    wght = self.func.get(ene - e)
                    folded += wght * np.array(val)

                str = '%10.5f' % e
                for vf in folded:
                    str += ' %12.7e' % vf
                print >> out, str

                e += de

        else: # just list energies and oszillator strengths

            print >> out, '#', self.fields

            for ene, val in zip(self.energies, self.values):
                str = '%10.5f' % ene
                for vf in val:
                    str += ' %12.7e' % vf
                print >> out, str
                
        if filename != None: 
            out.close()

class AbsoptionSpectrum(Writer):    
    def __init__(self,
                 exlist,
                 folding=None, 
                 width=0.08 # Gauss/Lorentz width
                 ):
        Writer.__init__(self, folding, width)
        self.title = 'Photoabsorption spectrum from linear response TD-DFT'
        self.fields = 'om [eV]     osz          osz x       osz y       osz z'

        self.energies = []
        self.values = []
        for ex in exlist:
            self.energies.append(ex.get_energy() * Ha)
            self.values.append(ex.get_oscillator_strength())
