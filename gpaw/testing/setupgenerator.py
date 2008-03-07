#!/usr/bin/env python

import os
from gpaw.atom import generator
from gpaw import mpi
#from generator import Generator
Generator = generator.Generator

maxparcount = 5 # This is the maximum number of parameters supported
# by the generator function

comm = mpi.world

class SetupGenerator:
    """
    A SetupGenerator generates setups for a particular element during a
    setup optimization run with an Optimizer. The generated setup files
    will be named according to the name of this SetupGenerator.

    Any existing setup with the given element/name combination will be
    overwritten. Each concurrently active generator should thus use a
    different name or element.

    A SetupGenerator must have a standard_parameters() method which is
    used to initialize the optimizer.

    A SetupGenerator should possess a method which generates a setup given
    some number of parameters in a way compatible with the
    standard_parameters() method.
    """
    
    def __init__(self, symbol='H', name='test',
                 whichparms=tuple(range(maxparcount)),
                 std_parm_values=None):
        """
        Creates a SetupOptimizer for the element with the given symbol
        (string), where setup files will be generated with the name
        <symbol>.<name>.PBE . The name parameter should be a non-empty
        string.

        This documentation is incomplete. See the gpaw wiki for further
        details.
        """

        # We don't want anything to mess up with existing files
        # so make sure a proper name is entered with a couple of chars
        # (it should be enough to test for len==0, but what the heck)
        if len(name) < 1:
            raise ValueError('Please supply a non-empty name.')
        self.symbol = symbol
        self.name = name

        if callable(whichparms):
            self.parmfilter = whichparms
            self.stdparms = std_parm_values
            self.descriptor = str(whichparms)
        else:
            parmfilter = DefaultParmFilter(symbol, whichparms, std_parm_values)
            self.descriptor = 'custom parameter spec'
            self.parmfilter = parmfilter.filter
            self.get_standard_parameters = parmfilter.get_standard_parameters

    def get_standard_parameters(self):
        if self.stdparms is None:
            raise ValueError('Standard parameters not defined!')
        else:
            return self.stdparms
        
    def new_setup(self, r=None, rvbar=None, rcomp=None, rfilter=None,
                  hfilter=None):
        """Generate new molecule setup.

        The new setup depends on five parameters (Bohr units):

        * 0.6 < r < 1.9: cutoff radius for projector functions
        * 0.6 < rvbar < 1.9: cutoff radius zero potential (vbar)
        * 0.6 < rcomp < 1.9: cutoff radius for compensation charges
        * 0.6 < rfilter < 1.9: cutoff radius for Fourier-filtered
          projector functions
        * 0.2 < hfilter < 0.6: target grid spacing

        Use the setup like this::

          calc = Calculator(setups={symbol: name}, ...)
        """

        (r, rvbar, rcomp, rfilter, hfilter) = \
            standard_setup_parameters(self.symbol, r, rvbar, rcomp, rfilter,
                                      hfilter)

        param = generator.parameters[self.symbol]

        core=''
        if param.has_key('core'):
            core=param['core']
        
        g = Generator(self.symbol, 'PBE', scalarrel=True, nofiles=True)
        g.run(core=core,
              rcut=r,
              vbar=('poly', rvbar),
              filter=(hfilter, rfilter / r),
              rcutcomp=rcomp,
              logderiv=False,
              name=self.name)
        path = os.environ['GPAW_SETUP_PATH'].split(':')[0]
        name = self.symbol+'.'+self.name+'.PBE'
        os.rename(name, path + '/'+name)

    def generate_setup(self, parms):
        """
        Calls new_setup with after unpacking a parameter list. This method
        can be overridden to change the parameters that should be
        optimized.
        """
        if comm.rank == 0:
            newparms = self.parmfilter(parms)
            self.new_setup(*newparms)
        comm.barrier()

def standard_setup_parameters(symbol, r=None, rvbar=None, rcomp=None,
                              rfilter=None, hfilter=None):
    """
    Given up to five argument variables, selects any remaining parameters
    and returns them in a quintuple.
    
    The rcut-parameter is selected per default from the dictionary
    gpaw.atom.generator.parameters. Any remaining unspecified parameters
    are set set in terms of rcut except hfilter which deafults to 0.4.
    """

    param = generator.parameters[symbol]
    if r is None:
        r = param['rcut']
    if rvbar is None:
        rvbar = r
    if rcomp is None:
        rcomp = r
    if rfilter is None:
        rfilter = 2 * r
    if hfilter is None:
        hfilter = .4

    return (r, rvbar, rcomp, rfilter, hfilter)

class DefaultParmFilter:
    def __init__(self, symbol, whichparms, std_parm_values):
        self.whichparms = whichparms
        parcount = len(whichparms)
        self.parfilter = [False]*maxparcount
        self.symbol = symbol

        for par in whichparms:
            self.parfilter[par] = True

        self.stdparms = std_parm_values

    def get_standard_parameters(self):
        if self.stdparms is None:
            all_std_parms = standard_setup_parameters(self.symbol)

            return [par for (filter, par) in zip(self.parfilter, all_std_parms)
                    if filter]
        else:
            return self.stdparms

    def filter(self, parms):
        allparms = [None]*maxparcount

        for i,parindex in enumerate(self.whichparms):
            allparms[parindex] = parms[i]

        return allparms

