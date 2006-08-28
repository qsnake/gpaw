from math import pi, sqrt
import Numeric as num
import _gpaw
from gpaw import debug
from gpaw.utilities.tools import pack

# ..............................................................
# general excitation classes

class ExcitationList(list):
    """
    General Excitation List class
    """
    def __init__(self,calculator=None):

        if calculator==None:
            raise RuntimeError('You have to set a calculator for the ' +
                               'Excitation list')
        self.calculator = calculator

        # initialise empty list
        list.__init__(self)

    def GetEnergies(self):
        el = []
        for ex in self:
            el.append(ex.GetEnergy())
        return el
    
    def __str__(self):
        string= str(type(self))
        string+=" %d excitations:\n" % len(self)
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
    """Kohn-Sham single perticle excitations

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
                 jend=None):

        ExcitationList.__init__(self,calculator)
        
        if calculator != None:
            self.calculator=calculator
        
        paw = self.calculator.paw
        wf = paw.wf

        self.istart=istart
        self.jend=jend

        # here, we need to take care of the spins also for
        # closed shell systems (Sz=0)
        # vspin is the virtual spin of the wave functions,
        #       i.e. the spin used in the ground state calculation
        # pspin is the physical spin of the wave functions
        #       i.e. the spin of the excited states
        self.nvspins = wf.nspins
        self.npspins = wf.nspins
        if self.nvspins < 2:
            if nspins>self.nvspins:
                self.npspins = nspins

        # get possible transitions
        for ispin in range(self.npspins):
            vspin=ispin
##            print "vspin=",vspin,"ispin=",ispin
            if self.nvspins<2:
                vspin=0
            f=wf.kpt_u[vspin].f_n
            if self.jend==None: jend=len(f)
            else              : jend=min(self.jend+1,len(f))
##             print "<KSSingles::build> occupation list f=",f
##             print "<KSSingles::build> istart,jend=",istart,jend
            for i in range(istart,jend):
                for j in range(istart,jend):
                    fij=f[i]-f[j]
                    if fij>eps:
                        # this is an accepted transition
                        ks=KSSingle(i,j,ispin,vspin,paw)
                        self.append(ks)

class KSSingle(Excitation):
    """Single Kohn-Sham transition containing all it's indicees
    pspin=physical spin
    vspin=virtual  spin, i.e. spin in the code
    """
    def __init__(self,iidx=None,jidx=None,pspin=None,vspin=None,
                 paw=None):
        
        self.i=iidx
        self.j=jidx
        self.pspin=pspin
        self.vspin=vspin
        self.paw=paw
        wf=paw.wf
        f=wf.kpt_u[vspin].f_n
        self.fij=f[iidx]-f[jidx]
        e=wf.kpt_u[vspin].eps_n
        self.energy=e[jidx]-e[iidx]

        # calculate matrix elements
        
        # course grid contribution
        gd = wf.kpt_u[vspin].gd
        self.wfi = wf.kpt_u[vspin].psit_nG[iidx]
        self.wfj = wf.kpt_u[vspin].psit_nG[jidx]
        me = gd.calculate_dipole_moment(self.GetPairDensity())
##         print '<KSSingle> pseudo mij=',me
        
        # augmentation contributions
        for nucleus in paw.nuclei:
            Ra = nucleus.spos_c*paw.domain.cell_c
            Pi_i = nucleus.P_uni[self.vspin,self.i]
            Pj_i = nucleus.P_uni[self.vspin,self.j]
            D_ii = num.outerproduct(Pi_i, Pj_i)
            D_p  = pack(D_ii, symmetric=False)
            # L=0 term
            me += sqrt(4*pi)*Ra*num.dot(D_p, nucleus.setup.Delta_pL[:,0])
##             ma = sqrt(4*pi)*Ra*num.dot(D_p, nucleus.setup.Delta_pL[:,0])
            # L=1 terms
            if nucleus.setup.lmax>=1:
                for i in range(3):
                    # XXXX check def of Ylm used in setups XXXX
                    me[i] += sqrt(4*pi/3)*\
                             num.dot(D_p, nucleus.setup.Delta_pL[:,i+1])
        self.me=me
##         print '<KSSingle> mij,ma=',me,ma
        
    def __str__(self):
        str = "<KSSingle> %d->%d %d(%d) eji=%g[eV]" % \
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
