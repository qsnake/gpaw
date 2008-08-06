from math import pi, sqrt

import numpy as npy

import gpaw.mpi as mpi
from gpaw import debug
from gpaw.utilities import pack,packed_index
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.localized_functions import create_localized_functions
from gpaw.pair_density import PairDensity
from gpaw.operators import Gradient


# KS excitation classes

class KSSingles(ExcitationList):
    """Kohn-Sham single particle excitations

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
    """

    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 filehandle=None):

        if filehandle is not None:
            self.read(fh=filehandle)
            return None

        ExcitationList.__init__(self,calculator)
        
        if calculator is None:
            return # leave the list empty

        paw = self.calculator
        self.kpt_u = paw.kpt_u
        if self.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions in calculator!')

        # here, we need to take care of the spins also for
        # closed shell systems (Sz=0)
        # vspin is the virtual spin of the wave functions,
        #       i.e. the spin used in the ground state calculation
        # pspin is the physical spin of the wave functions
        #       i.e. the spin of the excited states
        self.nvspins = paw.nspins
        self.npspins = paw.nspins
        fijscale=1
        if self.nvspins < 2:
            if nspins>self.nvspins:
                self.npspins = nspins
                fijscale = 0.5
                
        # get possible transitions
        for ispin in range(self.npspins):
            vspin=ispin
            if self.nvspins<2:
                vspin=0
            f=self.kpt_u[vspin].f_n
            if jend==None: jend=len(f)-1
            else         : jend=min(jend,len(f)-1)
            
            for i in range(istart,jend+1):
                for j in range(istart,jend+1):
                    fij=f[i]-f[j]
                    if fij>eps:
                        # this is an accepted transition
                        ks=KSSingle(i,j,ispin,vspin,paw,fijscale=fijscale)
                        self.append(ks)

        self.istart = istart
        self.jend = jend

        trkm = self.GetTRK()
        print >> self.out, 'KSS TRK sum %g (%g,%g,%g)' % \
              (npy.sum(trkm)/3.,trkm[0],trkm[1],trkm[2])
        pol = self.GetPolarizabilities(lmax=3)
        print >> self.out, \
              'KSS polarisabilities(l=0-3) %g, %g, %g, %g' % \
              tuple(pol.tolist())

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename)
                else:
                    f = open(filename, 'r')
            else:
                f = fh

            f.readline()
            n = int(f.readline())
            for i in range(n):
                kss = KSSingle(string=f.readline())
                self.append(kss)
            self.update()
                
            if fh is None:
                f.close()

    def update(self):
        istart = self[0].i
        jend = 0
        npspins = 1
        nvspins = 1
        for kss in self:
            istart = min(kss.i, istart)
            jend = max(kss.j, jend)
            if kss.pspin == 1:
                npspins = 2
            if kss.spin == 1:
                nvspins = 2
        self.istart = istart
        self.jend = jend
        self.npspins = npspins
        self.nvspins = nvspins

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

            f.write('# KSSingles\n')
            f.write('%d\n' % len(self))
            for kss in self:
                f.write(kss.outstring())
            
            if fh is None:
                f.close()

 
class KSSingle(Excitation, PairDensity):
    """Single Kohn-Sham transition containing all it's indicees

    ::

      pspin=physical spin
      spin=virtual  spin, i.e. spin in the ground state calc.
      fijscale=weight for the occupation difference::
      me  = sqrt(fij*epsij) * <i|r|j>
      mur = - <i|r|a>
      muv = - <i|nabla|a>/omega_ia with omega_ia>0
      m   = <i|[r x nabla]|a> / (2c)
    """
    def __init__(self, iidx=None, jidx=None, pspin=None, spin=None,
                 paw=None, string=None, fijscale=1):
        
        if string is not None: 
            self.fromstring(string)
            return None

        # normal entry
        
        PairDensity.__init__(self, paw)
        PairDensity.initialize(self, paw.kpt_u[spin], iidx, jidx)

        self.pspin=pspin
        
        f=paw.kpt_u[spin].f_n
        self.fij=(f[iidx]-f[jidx])*fijscale
        e=paw.kpt_u[spin].eps_n
        self.energy=e[jidx]-e[iidx]

        # calculate matrix elements -----------

        gd = paw.kpt_u[spin].gd
        self.gd = gd

        # length form ..........................

        # course grid contribution
        # <i|r|j> is the negative of the dipole moment (because of negative
        # e- charge)
        me = -gd.calculate_dipole_moment(self.get())

        # augmentation contributions
        ma = npy.zeros(me.shape)
        for nucleus in paw.my_nuclei:
            Ra = nucleus.spos_c*paw.domain.cell_c
            Pi_i = nucleus.P_uni[self.u,self.i]
            Pj_i = nucleus.P_uni[self.u,self.j]
            Delta_pL = nucleus.setup.Delta_pL
            ni=len(Pi_i)
            ma0 = 0
            ma1 = npy.zeros(me.shape)
            for i in range(ni):
                for j in range(ni):
                    pij = Pi_i[i]*Pj_i[j]
                    ij = packed_index(i, j, ni)
                    # L=0 term
                    ma0 += Delta_pL[ij,0]*pij
                    # L=1 terms
                    if nucleus.setup.lmax>=1:
                        # see spherical_harmonics.py for
                        # L=1:y L=2:z; L=3:x
                        ma1 += npy.array([ Delta_pL[ij,3], Delta_pL[ij,1], \
                                           Delta_pL[ij,2] ])*pij
            ma += sqrt(4*pi/3)*ma1 + Ra*sqrt(4*pi)*ma0
        gd.comm.sum(ma)

#        print '<KSSingle> i,j,me,ma,fac=',self.i,self.j,\
#            me, ma,sqrt(self.energy*self.fij)
        self.me = sqrt(self.energy*self.fij) * ( me + ma )

        self.mur = - ( me + ma )
##        print '<KSSingle> mur=',self.mur,-self.fij *me

        # velocity form .............................

        # smooth contribution
        dwfdr_G = gd.empty()
        if not hasattr(gd,'ddr'):
            gd.ddr = [Gradient(gd, c).apply for c in range(3)]
        for c in range(3):
            gd.ddr[c](self.wfj, dwfdr_G)
            me[c] = gd.integrate(self.wfi*dwfdr_G)
            
        self.muv = me / self.energy
##        print '<KSSingle> muv=',self.muv

        # magnetic transition dipole ................
        
        # m depends on how the origin is set, so we need th centre of mass
        # of the structure
#        cm = paw

#        for i in range(3):
#            me[i] = gd.integrate(self.wfi*dwfdr_cg[i])
        
    def __add__(self, other):
        """Add two KSSingles"""
        result = self.copy()
        result.me = self.me + other.me
        result.mur = self.mur + other.mur
        result.muv = self.muv + other.muv
        return result

    def __sub__(self, other):
        """Subtract two KSSingles"""
        result = self.copy()
        result.me = self.me - other.me
        result.mur = self.mur - other.mur
        result.muv = self.muv - other.muv
        return result

    def __mul__(self, x):
        """Multiply a KSSingle with a number"""
        if type(x) == type(0.) or type(x) == type(0):
            result = self.copy()
            result.me = self.me * x
            result.mur = self.mur * x
            result.muv = self.muv * x
            return result
        else:
            return RuntimeError('not a number')
        
    def __div__(self, x):
        return self.__mul__(1. / x)

    def copy(self):
        return KSSingle(string=self.outstring())

    def fromstring(self,string):
        l = string.split()
        self.i = int(l.pop(0))
        self.j = int(l.pop(0))
        self.pspin = int(l.pop(0))
        self.spin = int(l.pop(0))
        self.energy = float(l.pop(0))
        self.fij = float(l.pop(0))
        if len(l) == 3: # old writing style
            self.me = npy.array([float(l.pop(0)) for i in range(3)])
        else:
            self.mur = npy.array([float(l.pop(0)) for i in range(3)])
            self.me = - self.mur * sqrt(self.energy*self.fij)
            self.muv = npy.array([float(l.pop(0)) for i in range(3)])
        return None

    def outstring(self):
        str = '%d %d   %d %d   %g %g' % \
               (self.i,self.j, self.pspin,self.spin, self.energy, self.fij)
        str += '  '
        for m in self.mur: str += '%12.4e' % m
        str += '  '
        for m in self.muv: str += '%12.4e' % m
        str += '\n'
        return str
        
    def __str__(self):
        str = "# <KSSingle> %d->%d %d(%d) eji=%g[eV]" % \
              (self.i, self.j, self.pspin, self.spin,
               self.energy*27.211)
        str += " (%g,%g,%g)" % (self.me[0],self.me[1],self.me[2])
        return str
    
    #####################
    ## User interface: ##
    #####################

    def GetEnergy(self):
        return self.energy

    def GetWeight(self):
        return self.fij

