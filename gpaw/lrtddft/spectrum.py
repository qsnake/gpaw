import sys
import numpy as np
from math import exp, pi, sqrt

from ase import Hartree as Ha
from gpaw.gauss import Gauss, Lorentz
from gpaw.version import version
from gpaw.utilities.folder import Folder

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
    if folding is not None: # fold the spectrum
        print >> out, '# %s folded, width=%g [%s]' % (folding,width,energyunit)
    print >> out,\
        '# om [%s]     osz          osz x       osz y       osz z'\
        % energyunit

    x = []
    y = []
    for ex in exlist:
        x.append(ex.get_energy() * Ha)
        y.append(ex.get_oscillator_strength())
        
    el, vl = Folder(width, folding).fold(x, y, de, emin, emax)
    for e, val in zip(el, vl):
        print >> out, "%10.5f %12.7e %12.7e %11.7e %11.7e" % \
            (e,val[0],val[1],val[2],val[3])

    if filename != None: out.close()

class Writer(Folder):
    def __init__(self, folding=None, width=0.08, # Gauss/Lorentz width
                 ):
        self.folding = folding
        Folder.__init__(self, width, folding)

    def write(self, filename=None, 
              emin=None, emax=None, de=None,
              comment=None):
        
        out = sys.stdout
        if filename != None:
            out = open( filename, 'w' )
 
        print >> out, '#', self.title
        print >> out, '# GPAW version:', version
        if comment:
            print >> out, '#', comment
        if self.folding is not None:
            print >> out, '# %s folded, width=%g [eV]' % (self.folding,
                                                          self.width)
        print >> out, '#', self.fields

        el, vl = self.fold(self.energies, self.values,
                           de, emin, emax)
        for e, val in zip(el, vl):
            str = '%10.5f' % e
            for vf in val:
                str += ' %12.7e' % vf
            print >> out, str
            
        if filename != None: 
            out.close()

