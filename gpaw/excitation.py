from math import pi, sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
from gpaw import debug
from gpaw.utilities import packed_index

# ..............................................................
# general excitation classes

class ExcitationList(list):
    """
    General Excitation List class
    """
    def __init__(self,calculator=None):

        # initialise empty list
        list.__init__(self)

        self.calculator = calculator

    def GetEnergies(self):
        el = []
        for ex in self:
            el.append(ex.GetEnergy())
        return el
    
    def __str__(self):
        string= '# ' + str(type(self))
        string+=', %d excitations:\n' % len(self)
        for ex in self:
            string += '  '+ex.__str__()+"\n"
        return string
        
class Excitation:
    def GetEnergy(self):
        """return the excitations energy relative to the ground state energy"""
        return self.energy
    
    def GetDipolME(self):
        """return the excitations dipole matrix element"""
        return self.me
    
    def GetOszillatorStrength(self):
        """return the excitations oszillator strength"""
        me=self.GetDipolME()
        osz=[0.]
        for i in range(3):
            val=2.*me[i]**2
            osz.append( val )
            osz[0]+=val/3.
        return osz

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

        self.calculator=calculator
        paw = self.calculator.paw
        self.kpt_u = paw.kpt_u
        
        self.istart=istart
        self.jend=jend

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
            if self.jend==None: jend=len(f)
            else              : jend=min(self.jend+1,len(f))

            for i in range(istart,jend):
                for j in range(istart,jend):
                    fij=f[i]-f[j]
                    if fij>eps:
                        # this is an accepted transition
                        ks=KSSingle(i,j,ispin,vspin,paw,fijscale=fijscale)
                        self.append(ks)

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            if fh is None:
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
        """Write current state to a file."""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'w')
            else:
                f = fh

            f.write('# KSSingles\n')
            f.write('%d\n' % len(self))
            for kss in self:
                f.write(kss.outstring())
            
            if fh is None:
                f.close()
 
class KSSingle(Excitation):
    """Single Kohn-Sham transition containing all it's indicees
    pspin=physical spin
    vspin=virtual  spin, i.e. spin in the ground state calc.
    fijscale=weight for the occupation difference
    """
    def __init__(self,iidx=None,jidx=None,pspin=None,vspin=None,
                 paw=None,string=None,fijscale=1):
        
        self.i=iidx
        self.j=jidx
        self.pspin=pspin
        self.vspin=vspin

        if string is not None: 
            self.fromstring(string)
            return None

        # normal entry
        
        self.paw=paw

        f=paw.kpt_u[vspin].f_n
        self.fij=(f[iidx]-f[jidx])*fijscale
        e=paw.kpt_u[vspin].eps_n
        self.energy=e[jidx]-e[iidx]

        # calculate matrix elements
        
        # course grid contribution
        gd = paw.kpt_u[vspin].gd
        self.wfi = paw.kpt_u[vspin].psit_nG[iidx]
        self.wfj = paw.kpt_u[vspin].psit_nG[jidx]
        me = gd.calculate_dipole_moment(self.GetPairDensity())
##        print '<KSSingle> pseudo mij=',self.i,self.j,me
        
        # augmentation contributions
        ma = num.zeros(me.shape,num.Float)
        for nucleus in paw.my_nuclei:
            Ra = nucleus.spos_c*paw.domain.cell_c
            Pi_i = nucleus.P_uni[self.vspin,self.i]
            Pj_i = nucleus.P_uni[self.vspin,self.j]
            Delta_pL = nucleus.setup.Delta_pL
            ni=len(Pi_i)
            ma0 = 0
            ma1 = num.zeros(me.shape,num.Float)
            for i in range(ni):
                for j in range(ni):
                    pji = Pi_i[i]*Pj_i[i]
                    ij = packed_index(i, j, ni)
                    # L=0 term
                    ma0 += Delta_pL[ij,0]*pji
                    # L=1 terms
                    if nucleus.setup.lmax>=1:
                        ma1 += num.array([ Delta_pL[ij,1], Delta_pL[ij,2], \
                                           Delta_pL[ij,2] ])
            ma += sqrt(4*pi/3)*ma1 + Ra*sqrt(4*pi)*ma0

##       print '<KSSingle> me,ma=',me,ma
        self.me = me + ma

    def fromstring(self,string):
        l = string.split()
##        print "l=",l
        self.i = int(l[0])
        self.j = int(l[1])
        self.pspin = int(l[2])
        self.vspin = int(l[3])
        self.energy = float(l[4])
        self.fij = float(l[5])
        self.me = num.array([float(l[6]),float(l[7]),float(l[8])])
##        print "me=",self.me
        
        return None

    def outstring(self):
        str = '%d %d   %d %d   %g %g' % \
               (self.i,self.j, self.pspin,self.vspin, self.energy, self.fij)
        str += '  '
        for m in self.me:
            str += ' %g' % m
        str += '\n'
        return str
        
    def __str__(self):
        str = "# <KSSingle> %d->%d %d(%d) eji=%g[eV]" % \
              (self.i, self.j, self.pspin, self.vspin,
               self.energy*27.211)
        return str
    
    #####################
    ## User interface: ##
    #####################

    def GetEnergy(self):
        return self.energy

    def GetWeight(self):
        return self.fij

    def GetPairDensity(self):
        return self.wfi*self.wfj
    
    def GetFineGridPairDensity(self):
        gd = self.paw.finegd
        n_g = gd.new_array()
        self.paw.interpolate(self.GetPairDensity(),n_g)
        return n_g 
