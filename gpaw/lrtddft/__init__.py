from math import sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER

from gpaw import debug
from gpaw.poisson_solver import PoissonSolver
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.lrtddft.omega_matrix import OmegaMatrix
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
                 derivativeLevel=None,
                 numscale=0.001,
                 filename=None):

        if filename is None:

            ExcitationList.__init__(self,calculator)

            self.calculator=None
            self.nspins=None
            self.eps=None
            self.istart=None
            self.jend=None
            self.xc=None
            self.derivativeLevel=None
            self.numscale=numscale
            self.update(calculator,nspins,eps,istart,jend,
                        xc,derivativeLevel,numscale)

        else:
            self.read(filename)

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
        self.out = calculator.out
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
        self.Om = OmegaMatrix(self.calculator,self.kss,
                              self.xc,self.derivativeLevel,self.numscale)
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

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'r')
            else:
                f = fh

            f.readline()
            self.xc = f.readline().replace('\n','')
            self.eps = float(f.readline())
            self.kss = KSSingles(filehandle=f)
            self.Om = OmegaMatrix(filehandle=f)
            self.Om.Kss(self.kss)

            # check if already diagonalized
            p = f.tell()
            s = f.readline()
            if s != '# Eigenvalues\n':
                # go back to previous position
                f.seek(p)
            else:
                # load the eigenvalues/vectors
                pass

            if fh is None:
                f.close()

    def __str__(self):
        string = ExcitationList.__str__(self)
        string += '# derived from:\n'
        string += self.kss.__str__()
        return string

    def write(self, filename=None, fh=None):
        """Write current state to a file."""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'w')
            else:
                f = fh

            f.write('# LrTDDFT\n')
            xc = self.xc
            if xc is None: xc = 'RPA'
            f.write(xc+'\n')
            f.write('%g' % self.eps + '\n')
            self.kss.write(fh=f)
            self.Om.write(fh=f)

##            print "<write> ",self.istart,self.jend
##            print "<write> 2:",self.kss.istart,self.kss.jend
            if len(self):
                f.write('# Eigenvalues\n')
                istart = self.istart
                if istart is None: istart = self.kss.istart
                jend = self.jend
                if jend is None: jend = self.kss.jend
                f.write('%d %d %d'%(len(self),istart,jend)+'\n')
                for ex in self:
                    print 'ex=',ex
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
            res[ispin][jspin]=num.zeros(dup.shape,num.Float)
            _gpaw.d2Excdnsdnt(dup, ddn, ispin, jspin, res[ispin][jspin])
    return res

def d2Excdn2(den):
    """Second derivative of Exc unpolarised"""
    res=num.zeros(den.shape,num.Float)
    _gpaw.d2Excdn2(den, res)
    return res

class LrTDDFTExcitation(Excitation):
    def __init__(self,Om=None,i=None,
                 e=None,m=None):
        # define from the diagonalized Omega matrix
        if Om is not None:
            if i is None:
                raise RuntimeError
        
            self.energy=sqrt(Om.eigenvalues[i])
            f = Om.eigenvectors[i]
            self.f = f
            kss = Om.kss
            for j in range(len(kss)):
                # ?????? ist weight noch nötig ?????
                weight = f[j]*sqrt(kss[j].GetEnergy()*kss[j].GetWeight())
                if j==0:
                    self.me = kss[j].GetDipolME()*weight
                else:
                    self.me += kss[j].GetDipolME()*weight
            return

        # define from energy and matrix element
        if e is not None:
            if m is None:
                raise RuntimeError
            self.enegy = e
            self.me = m
            return

        raise RuntimeError

    def outstring(self):
        str = '%g ' % self.energy
        str += '  '
        for m in self.me:
            str += ' %g' % m
        str += '\n'
        return str
        
    def __str__(self):
        str = "<LrTDDFTExcitation> om=%g[eV] me=(%g,%g,%g)" % \
              (self.energy*27.211,self.me[0],self.me[1],self.me[2])
        return str


        




