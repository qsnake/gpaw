from math import sqrt
import sys

import numpy as npy
from ase.units import Hartree

import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER
from gpaw import debug
from gpaw.poisson import PoissonSolver
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.lrtddft.apmb import ApmB
##from gpaw.lrtddft.transition_density import TransitionDensity
from gpaw.utilities import packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XC3DGrid, XCFunctional

"""This module defines a linear response TDDFT-class."""

class LrTDDFT(ExcitationList):
    """Linear Response TDDFT excitation class
    
    Input parameters:

    calculator:
    the calculator object after a ground state calculation
      
    nspins:
    number of spins considered in the calculation
    Note: Valid only for unpolarised ground state calculation

    eps:
    Minimal occupation difference for a transition (default 0.001)

    istart:
    First occupied state to consider
    jend:
    Last unoccupied state to consider
      
    xc:
    Exchange-Correlation approximation in the Kernel
    derivativeLevel:
    0: use Exc, 1: use vxc, 2: use fxc  if available

    filename:
    read from a file
    """
    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 xc=None,
                 derivativeLevel=1,
                 numscale=0.00001,
                 filename=None,
                 finegrid=2,
                 force_ApmB=False # for tests
                 ):

        if filename is None:

            ExcitationList.__init__(self,calculator)

            self.filename=None
            self.calculator=None
            self.nspins=None
            self.eps=None
            self.istart=None
            self.jend=None
            self.xc=None
            self.derivativeLevel=None
            self.numscale=numscale
            self.finegrid=finegrid
            self.force_ApmB=force_ApmB
            
            self.update(calculator,nspins,eps,istart,jend,
                        xc,derivativeLevel,numscale)

        else:
            self.read(filename)

    def analyse(self, what=None, out=None, min=0.1):
        """Print info about the transitions.
        
        Parameters:
          1. what: I list of excitation indicees, None means all
          2. out : I where to send the output, None means sys.stdout
          3. min : I minimal contribution to list (0<min<1)
        """
        if what is None:
            what = range(len(self))
        elif isinstance(what, int):
            what = [what]

        if out is None:
            out = sys.stdout
            
        for i in what:
            print >> out, str(i) + ':', self[i].analyse(min=min)
            
    def update(self,
               calculator=None,
               nspins=None,
               eps=0.001,
               istart=0,
               jend=None,
               xc=None,
               derivativeLevel=None,
               numscale=0.001):

        changed=False
        if self.calculator!=calculator or \
           self.nspins != nspins or \
           self.eps != eps or \
           self.istart != istart or \
           self.jend != jend :
            changed=True

        if not changed: return

        self.calculator = calculator
        self.out = calculator.txt
        self.nspins = nspins
        self.eps = eps
        self.istart = istart
        self.jend = jend
        self.xc = xc
        self.derivativeLevel=derivativeLevel
        self.numscale=numscale
        self.kss = KSSingles(calculator=calculator,
                             nspins=nspins,
                             eps=eps,
                             istart=istart,
                             jend=jend)
        if not self.force_ApmB:
            Om = OmegaMatrix
            name = 'LrTDDFT'
            if self.xc:
                xc = XCFunctional(self.xc)
                if xc.hybrid > 0.0:
                    Om = ApmB
                    name = 'LrTDDFThyb'
        else:
            Om = ApmB
            name = 'LrTDDFThyb'
        self.Om = Om(self.calculator, self.kss,
                     self.xc, self.derivativeLevel, self.numscale,
                     finegrid=self.finegrid)
        self.name = name
##        self.diagonalize()

    def diagonalize(self, istart=None, jend=None):
        self.istart = istart
        self.jend = jend
        self.Om.diagonalize(istart,jend)
        
        # remove old stuff
        while len(self): self.pop()

        for j in range(len(self.Om.kss)):
            self.append(LrTDDFTExcitation(self.Om,j))

    def get_Om(self):
        return self.Om

    def Read(self, filename=None, fh=None):
        return self.read(filename,fh)
        
    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            self.Ha = Hartree
            
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename)
                else:
                    f = open(filename, 'r')
                self.filename = filename
            else:
                f = fh
                self.filename = None

            # get my name
            s = f.readline().replace('\n','')
            hash, self.name = s.split()
            
            self.xc = f.readline().replace('\n','')
            values = f.readline().split()
            self.eps = float(values[0])
            if len(values) > 1:
                self.derivativeLevel = int(values[1])
                self.numscale = float(values[2])
                self.finegrid = int(values[3])
            else:
                # old writing style, use old defaults
                self.numscale = 0.001
                pass
                
            self.kss = KSSingles(filehandle=f)
            if self.name == 'LrTDDFT':
                self.Om = OmegaMatrix(kss=self.kss, filehandle=f)
            else:
                self.Om = ApmB(kss=self.kss, filehandle=f)
            self.Om.Kss(self.kss)

            # check if already diagonalized
            p = f.tell()
            s = f.readline()
            if s != '# Eigenvalues\n':
                # go back to previous position
                f.seek(p)
            else:
                # load the eigenvalues
                n = int(f.readline().split()[0])
                for i in range(n):
                    l = f.readline().split()
                    E = float(l[0])
                    me = [float(l[1]),float(l[2]),float(l[3])]
                    self.append(LrTDDFTExcitation(e=E,m=me))
                    
                # load the eigenvectors
                pass

            if fh is None:
                f.close()

    def SPA(self):
        """Return the excitation list according to the
        single pole approximation. See e.g.:
        Grabo et al, Theochem 501 (2000) 353-367
        """
        spa = self.kss
        for i in range(len(spa)):
            E = sqrt(self.Om.full[i][i])
            print "<SPA> E was",spa[i].GetEnergy()*27.211," and is ",E*27.211
            spa[i].SetEnergy(sqrt(self.Om.full[i][i]))
        return spa

    def __str__(self):
        string = ExcitationList.__str__(self)
        string += '# derived from:\n'
        string += self.kss.__str__()
        return string

    def Write(self, filename=None, fh=None):
        return self.write(filename,fh)
    
    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files. 
        """
        if mpi.rank == mpi.MASTER:
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename,'wb')
                else:
                    f = open(filename, 'w')
            else:
                f = fh

            f.write('# ' + self.name + '\n')
            xc = self.xc
            if xc is None: xc = 'RPA'
            f.write(xc+'\n')
            f.write('%g %d %g %d' % (self.eps, int(self.derivativeLevel),
                                     self.numscale, int(self.finegrid)) + '\n')
            self.kss.write(fh=f)
            self.Om.write(fh=f)

            if len(self):
                f.write('# Eigenvalues\n')
                istart = self.istart
                if istart is None: istart = self.kss.istart
                jend = self.jend
                if jend is None: jend = self.kss.jend
                f.write('%d %d %d'%(len(self),istart,jend)+'\n')
                for ex in self:
                    f.write(ex.outstring())
                f.write('# Eigenvectors\n')
                for ex in self:
                    for w in ex.f:
                        f.write('%g '%w)
                    f.write('\n')

            if fh is None:
                f.close()

def d2Excdnsdnt(dup,ddn):
    """Second derivative of Exc polarised"""
    res=[[0, 0], [0, 0]]
    for ispin in range(2):
        for jspin in range(2):
            res[ispin][jspin]=npy.zeros(dup.shape)
            _gpaw.d2Excdnsdnt(dup, ddn, ispin, jspin, res[ispin][jspin])
    return res

def d2Excdn2(den):
    """Second derivative of Exc unpolarised"""
    res=npy.zeros(den.shape)
    _gpaw.d2Excdn2(den, res)
    return res

class LrTDDFTExcitation(Excitation):
    def __init__(self,Om=None,i=None,
                 e=None,m=None):
        # define from the diagonalized Omega matrix
        if Om is not None:
            if i is None:
                raise RuntimeError

            ev = Om.eigenvalues[i]
            if ev < 0:
                # we reached an instability, mark it with a negative value
                self.energy = -sqrt(-ev)
            else:
                self.energy = sqrt(ev)
            self.f = Om.eigenvectors[i]
            self.kss = Om.kss
            self.me = 0.
            for f,k in zip(self.f,self.kss):
                self.me += f * k.me

            return

        # define from energy and matrix element
        if e is not None:
            if m is None:
                raise RuntimeError
            self.energy = e
            self.me = m
            return

        raise RuntimeError

    def density_change(self,paw):
        """get the density change associated with this transition"""
        raise NotImplementedError

    def outstring(self):
        str = '%g ' % self.energy
        str += '  '
        for m in self.me:
            str += ' %g' % m
        str += '\n'
        return str
        
    def __str__(self):
        m2 = npy.sum(self.me*self.me)
        m = sqrt(m2)
        if m>0: me = self.me/m
        else:   me = self.me
        str = "<LrTDDFTExcitation> om=%g[eV] |me|=%g (%.2f,%.2f,%.2f)" % \
              (self.energy*27.211,m,me[0],me[1],me[2])
        return str

    def analyse(self,min=.1):
        """Return an analysis string of the excitation"""
        s='E=%.3f'%(self.energy*27.211)+' eV, f=%.3g'\
           %(self.GetOscillatorStrength()[0])+'\n'

        def sqr(x): return x*x
        spin = ['u','d'] 
        min2 = sqr(min)
        rest = npy.sum(self.f**2)
        for f,k in zip(self.f,self.kss):
            f2 = sqr(f)
            if f2>min2:
                s += '  %d->%d ' % (k.i,k.j) + spin[k.pspin] + ' ' 
                s += '%.3g \n'%f2
                rest -= f2
        s+='  rest=%.3g'%rest
        return s
        




