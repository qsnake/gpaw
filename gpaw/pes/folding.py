from gpaw.gauss import Gauss, Lorentz
import sys

def folding_routine(enlist,width,filename=None,emin=None,emax=None,
                     de=None,folding='Gauss',comment=None):
    """
    Print the Photo emission spectrum

    Parameters:
    =============== ===================================================
    ``enlist``      [energies, weights]
    ``filename``    File name for the output file, STDOUT if not given
    ``emin``        min. energy, set to cover all energies if not given
    ``emax``        max. energy, set to cover all energies if not given
    ``de``          energy spacing
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
 
    # output
    out = sys.stdout
    if filename != None:
        out = open( filename, 'w' )
    if comment:
        print >> out, '#', comment

    print >> out, '# Photo emission spectrum'

    # minimal and maximal energies
    if emin == None:
        emin = min(enlist[0])
        emin -= 4*width
    if emax == None:
        emax = max(enlist[0])
        emax += 4*width

    # set de to sample 4 points in the width
    if de == None:
        de = width/4.

    if func is not None: # fold the spectrum
        
        print >> out, '# %s folded, width=%g [eV]' % (folding,width)
        print >> out,\
              '# Binding energy [eV]     Folded spectroscopic factor'\

        # loop over energies
        emax=emax+.5*de
        e=emin
        while e<emax:
            val=0
            for index in range(len(enlist[0])):
                wght=func.get(enlist[0][index]-e)
                osz=enlist[1][index]
                val += wght*osz
            print >> out, "%10.5f %12.7e" % \
                  (e,val)
            e+=de

    else: # just list energies and oszillator strengths

        print >> out,\
              '# Binding energy [eV]     Spectroscopic factor'\

        for index in range(len(enlist[0])):
            e=enlist[0][index]
            val=enlist[1][index]
            print >> out, "%10.5f %12.7e" % \
                  (e,val)
        
    if filename != None: out.close()
