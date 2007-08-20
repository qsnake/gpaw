from math import pi, sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
from gpaw import debug
from gpaw.utilities import pack,packed_index
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.localized_functions import create_localized_functions
from gpaw.pair_density import PairDensity

from gpaw.io.plt import write_plt

# ..............................................................
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
        if not self.kpt_u[0].psit_nG:
            raise RuntimeError('No wave functions in calculator')

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

        self.istart=istart
        self.jend=jend

        trkm = self.GetTRK()
        print >> self.out, 'KSS TRK sum %g (%g,%g,%g)' % \
              (num.sum(trkm)/3.,trkm[0],trkm[1],trkm[2])
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
        istart = len(self)
        jend = 0
        for kss in self:
            istart = min(kss.i,istart)
            jend = max(kss.j,jend)
        self.istart = istart
        self.jend = jend

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

 
class KSSingle(Excitation,PairDensity):
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
    def __init__(self,iidx=None,jidx=None,pspin=None,spin=None,
                 paw=None,string=None,fijscale=1):
        
        if string is not None: 
            self.fromstring(string)
            return None

        # normal entry
        
        PairDensity.__init__(self,paw.density,paw.kpt_u[spin],iidx,jidx)
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
        me = -gd.calculate_dipole_moment(self.GetPairDensity())

        # augmentation contributions
        ma = num.zeros(me.shape,num.Float)
        for nucleus in paw.my_nuclei:
            Ra = nucleus.spos_c*paw.domain.cell_c
            Pi_i = nucleus.P_uni[self.spin,self.i]
            Pj_i = nucleus.P_uni[self.spin,self.j]
            Delta_pL = nucleus.setup.Delta_pL
            ni=len(Pi_i)
            ma0 = 0
            ma1 = num.zeros(me.shape,num.Float)
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
                        ma1 += num.array([ Delta_pL[ij,3], Delta_pL[ij,1], \
                                           Delta_pL[ij,2] ])*pij
            ma += sqrt(4*pi/3)*ma1 + Ra*sqrt(4*pi)*ma0

##         print '<KSSingle> me,ma=',me,ma
##         print '<KSSingle> i,j,m,fac=',self.i,self.j,\
##               me+ma,sqrt(self.energy*self.fij)
        self.me = sqrt(self.energy*self.fij) * ( me + ma )

        self.mur = - self.fij * ( me + ma )

        # velocity form .............................

        # course grid contribution
#        dwfdr_cg = gd.empty(3)
#        gd.derivative(self.wfj, dwfdr_cg)
#        for i in range(3):
#            me[i] = gd.integrate(self.wfi*dwfdr_cg[i])

        self.muv = self.fij * me

        # magnetic transition dipole ................
        
        # m depends on how the origin is set, so we need th centre of mass
        # of the structure
#        cm = paw

#        for i in range(3):
#            me[i] = gd.integrate(self.wfi*dwfdr_cg[i])
        
         
         

    def fromstring(self,string):
        l = string.split()
        self.i = int(l[0])
        self.j = int(l[1])
        self.pspin = int(l[2])
        self.spin = int(l[3])
        self.energy = float(l[4])
        self.fij = float(l[5])
        self.me = num.array([float(l[6]),float(l[7]),float(l[8])])
        return None

    def outstring(self):
        str = '%d %d   %d %d   %g %g' % \
               (self.i,self.j, self.pspin,self.spin, self.energy, self.fij)
        str += '  '
        for m in self.me:
            str += '%12.4e' % m
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

    def GetPairDensity(self,finegrid=False):
        """Get pair density"""
        return self.get(finegrid) # inherited from PairDensity

    def GetPairDensityAndCompensationCharges(self,finegrid=False):
        """Get pair densisty including the compensation charges"""
        # inherited from PairDensity
        return self.width_compensation_charges(finegrid)

