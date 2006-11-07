from math import sqrt
import Numeric as num
import _gpaw
from gpaw import debug
from gpaw.poisson_solver import PoissonSolver
from gpaw.excitation import Excitation,ExcitationList,KSSingles
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XC3DGrid, XCFunctional

"""This module defines a linear response TDDFT-class."""

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

class OmegaMatrix:
    """Omega matrix in Casidas linear response formalism
    """
    def __init__(self,
                 calculator=None,
                 kss=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001
                 ):
        self.calculator = calculator
        self.kss = kss
        if xc is not None:
            self.xc = XC3DGrid(xc,self.calculator.paw.finegd,
                               self.kss.npspins)
            # check derivativeLevel
            if derivativeLevel is None:
                derivativeLevel=\
                    self.xc.get_functional().get_max_derivative_level()
            self.derivativeLevel=derivativeLevel
        else:
            self.xc = None

        self.numscale=numscale
    
        self.full = self.get_full()

    def get_full(self,rpa=None):
        if rpa is None:
            rpa = self.get_rpa()
        Om = rpa

        if self.xc is None:
            return Om
        
##         print '>> kss=',self.kss
##         print '>> self.xc=',self.xc
##         print '>> self.numscale=',self.numscale
        xcf=self.xc.get_functional()
        print '<OmegaMatrix::get_full> xc=',xcf.get_name()
        print '<OmegaMatrix::get_full> derivative Level=',self.derivativeLevel
        print '<OmegaMatrix::get_full> numscale=',self.numscale

        paw = self.calculator.paw
        gd = paw.finegd    
        kss=self.kss
        nij = len(kss)
        
        if kss.nvspins==2: # spin polarised ground state calc.
            n_g = paw.density.nt_sg
            v_g=n_g[0].copy()
        else:
            if kss.npspins==2:
                n_g[0] = .5*paw.density.nt_sg[0]
                n_g[1] = n_g[0]
            else:
                n_g = paw.density.nt_sg[0]

        if self.derivativeLevel==0:
            if kss.npspins==2:
                v_g=n_g[0].copy()
            else:
                v_g=n_g.copy()
        elif self.derivativeLevel==1:
            if kss.npspins==2:
                vp_g=n_g.copy()
                vm_g=n_g.copy()
            else:
                vp_g=n_g.copy()
                vm_g=n_g.copy()
        elif self.derivativeLevel==2:
            print "Om(RPA)=\n",Om
            if kss.npspins==2:
                fxc=d2Excdnsdnt(n_g[0],n_g[1])
            else:
                fxc=d2Excdn2(n_g)
        else:
            raise ValueError('derivativeLevel can only be 0,1,2')
            
        ns=self.numscale
        xc=self.xc
        for ij in range(nij):
            
            if self.derivativeLevel == 1:
                if kss.npspins==2: # spin polarised
                    nv_g=n_g.copy()
                    nv_g[kss[ij].pspin] += ns*kss[ij].GetFineGridPairDensity()
                    xc.get_energy_and_potential(nv_g[0],vp_g[0],
                                                nv_g[1],vp_g[1])
                    nv_g=n_g.copy()
                    nv_g[kss[ij].pspin] -= ns*kss[ij].GetFineGridPairDensity()
                    xc.get_energy_and_potential(nv_g[0],vp_g[0],
                                                nv_g[1],vp_g[1])
                    vv_g = .5*(vp_g[kss[ij].pspin]-vm_g[kss[ij].pspin])/ns
                else: # spin unpolarised
                    nv_g=n_g + ns*kss[ij].GetFineGridPairDensity()
                    xc.get_energy_and_potential(nv_g,vp_g)
                    nv_g=n_g - ns*kss[ij].GetFineGridPairDensity()
                    xc.get_energy_and_potential(nv_g,vm_g)
                    vv_g = .5*(vp_g-vm_g)/ns

            for kq in range(ij,nij):
                
                weight=2.*sqrt(kss[ij].GetEnergy()*kss[kq].GetEnergy()*
                               kss[ij].GetWeight()*kss[kq].GetWeight())


                if self.derivativeLevel == 0:
                    # only Exc is available
                    
                    if kss.npspins==2: # spin polarised
                        nv_g = n_g.copy()
                        nv_g[kss[ij].pspin] +=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] +=\
                                        kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(\
                                        nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = n_g.copy()
                        nv_g[kss[ij].pspin] +=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] -= \
                                        kss[kq].GetFineGridPairDensity()
                        Excpm = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = n_g.copy()
                        nv_g[kss[ij].pspin] -=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] +=\
                                        kss[kq].GetFineGridPairDensity()
                        Excmp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = n_g.copy()
                        nv_g[kss[ij].pspin] -= \
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] -=\
                                        kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                    else: # spin unpolarised
                        nv_g=n_g + ns*kss[ij].GetFineGridPairDensity()\
                              + ns*kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=n_g + ns*kss[ij].GetFineGridPairDensity()\
                              - ns*kss[kq].GetFineGridPairDensity()
                        Excpm = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=n_g - ns*kss[ij].GetFineGridPairDensity()\
                              + ns*kss[kq].GetFineGridPairDensity()
                        Excmp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=n_g - ns*kss[ij].GetFineGridPairDensity()\
                              - ns*kss[kq].GetFineGridPairDensity()
                        Excmm = xc.get_energy_and_potential(nv_g,v_g)

                    Om[ij,kq] += weight *\
                                0.25*(Excpp-Excmp-Excpm+Excmm)/(ns*ns)
                              
                elif self.derivativeLevel == 1:
                    # vxc is available
                    Om[ij,kq] += weight *\
                        gd.integrate(kss[kq].GetFineGridPairDensity()*vv_g)

                elif self.derivativeLevel == 2:
                    # fxc is available
                    if kss.npspins==2: # spin polarised
                        Om[ij,kq] += weight *\
                            gd.integrate(kss[ij].GetFineGridPairDensity()*
                                         kss[kq].GetFineGridPairDensity()*
                                         fxc[kss[ij].pspin,kss[kq].pspin])
                    else: # spin unpolarised
                        Om[ij,kq] += weight *\
                            gd.integrate(kss[ij].GetFineGridPairDensity()*
                                         kss[kq].GetFineGridPairDensity()*
                                         fxc)
                if ij != kq:
                    Om[kq,ij] = Om[ij,kq]

        print ">> Om=\n",Om
        return Om

    def get_rpa(self,epsilon=0.001):
        """calculate RPA part of the omega matrix"""
        paw = self.calculator.paw
        gd = paw.finegd
        poisson = PoissonSolver(gd, paw.out, paw.hamiltonian.poisson_stencil)
        kss=self.kss

        # calculate omega matrix
        nij = len(kss)
        n_g = gd.new_array()
        phi_g = gd.new_array()
        Om = num.zeros((nij,nij),num.Float)
##        return Om
        for ij in range(nij):
##            print ">> ij,energy=",ij,kss[ij].GetEnergy()
            paw.density.interpolate(kss[ij].GetPairDensity(),n_g)
##            print ">> integral=",gd.integrate(n_g)
            poisson.solve(phi_g,n_g,charge=None)
            
            for kq in range(ij,nij):
##                print ">> kq=",kq
                paw.density.interpolate(kss[kq].GetPairDensity(),n_g)
                Om[ij,kq]=2.*sqrt(kss[ij].GetEnergy()*kss[kq].GetEnergy()*
                                  kss[ij].GetWeight()*kss[kq].GetWeight())*\
                                  gd.integrate(n_g*phi_g)
                if ij == kq:
                    Om[ij,kq] += kss[ij].GetEnergy()**2
                else:
                    Om[kq,ij]=Om[ij,kq]

##        print ">> Om=\n",Om
        return Om

    def diagonalize(self):
        self.eigenvectors = self.full.copy()
        self.eigenvalues = num.zeros((len(self.kss)),num.Float)
        info = diagonalize(self.eigenvectors, self.eigenvalues)
        if info != 0:
            raise RuntimeError('Diagonalisation error in OmegaMatrix')

class LrTDDFTExcitation(Excitation):
    def __init__(self,Om=None,i=None):
        if Om is None:
            raise RuntimeError
        if i is None:
            raise RuntimeError
        
        self.energy=sqrt(Om.eigenvalues[i])
        f = Om.eigenvectors[i]
        for j in range(len(Om.kss)):
            weight = f[j]*sqrt(Om.kss[j].GetEnergy()*Om.kss[j].GetWeight())
            if j==0:
                self.me = Om.kss[j].GetDipolME()*weight
            else:
                self.me += Om.kss[j].GetDipolME()*weight

    def __str__(self):
        str = "<LrTDDFTExcitation> om=%g[eV] me=(%g,%g,%g)" % \
              (self.energy*27.211,self.me[0],self.me[1],self.me[2])
        return str

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
    """
    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001):
        
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
##         Om.diagonalize()

##         for j in range(len(self.kss)):
##             self.append(LrTDDFTExcitation(Om,j))
        
    def get_Om(self):
        return self.Om
 
class LocalIntegrals:
    """Contains the local integrals needed for Linear response TDDFT"""
    def __init__(self,gen=None):
        if gen is not None:
            self.evaluate(gen)

    def evaluate(self,gen):
        # get Gaunt coefficients
        from gpaw.gaunt import gaunt

        # get Hartree potential calculator
        from gpaw.setup import Hartree

        # maximum angular momentum
        Lmax = 2 * max(gen.l_j,gen.lmax) + 1

        # unpack valence states * r:
        uv_j = []
        lv_j = []
        Nvi  = 0 
        for l, u_n in enumerate(gen.u_ln):
            for u in u_n:
                uv_j.append(u) # unpacked valence state array
                lv_j.append(l) # corresponding angular momenta
                Nvi += 2*l+1   # number of valence states (including m)
                
        # number of valence orbitals (j only, i.e. not m-number)
        Njval  = len(lv_j)

        # sum over first valence state index
        i1 = 0
        for jv1 in range(Njval):
            lv1 = lv_j[jv1] 

            # sum over second valence state index
            i2 = 0
            for jv2 in range(Njval):
                lv2 = lv_j[jv2]

                # two state "density"
                nij = uv_j[jv1]*uv_j[jv2]
                nij[1:] /= gen.r[1:]**2  
            

        




##     # diagonal +-1 elements in Hartree matrix
##     a1_g  = 1.0 - 0.5 * (gen.d2gdr2 * gen.dr**2)[1:]
##     a2_lg = -2.0 * num.ones((Lmax, gen.N - 1), num.Float)
##     x_g   = ((gen.dr / gen.r)**2)[1:]
##     for l in range(1, Lmax):
##         a2_lg[l] -= l * (l + 1) * x_g
##     a3_g = 1.0 + 0.5 * (gen.d2gdr2 * gen.dr**2)[1:]

##     # initialize potential calculator (returns v*r^2*dr/dg)
##     H = Hartree(a1_g, a2_lg, a3_g, gen.r, gen.dr).solve

##     # initialize X_ii matrix
##     X_ii = num.zeros((Nvi,Nvi), num.Float)

##     # do actual calculation of exchange contribution
##     Exx = 0.0
##     for j1 in nstates:
##         # angular momentum of first state
##         l1 = atom.l_j[j1]

##         for j2 in mstates:
##             # angular momentum of second state
##             l2 = atom.l_j[j2]

##             # joint occupation number
##             f12 = .5 * atom.f_j[j1]/(2. * l1 + 1) * \
##                        atom.f_j[j2]/(2. * l2 + 1)

##             # electron density
##             n = atom.u_j[j1]*atom.u_j[j2]
##             n[1:] /= atom.r[1:]**2

##             # determine potential times r^2 times length element dr/dg
##             vr2dr = num.zeros(atom.N, num.Float)

##             # L summation
##             for l in range(l1 + l2 + 1):
##                 # get potential for current l-value
##                 vr2drl = H(n, l)

##                 # take all m1 m2 and m values of Gaunt matrix of the form
##                 # G(L1,L2,L) where L = {l,m}
##                 G2 = gaunt[l1**2:(l1+1)**2, l2**2:(l2+1)**2,\
##                            l**2:(l+1)**2]**2

##                 # add to total potential
##                 vr2dr += vr2drl * num.sum(G2.copy().flat)

##             # add to total exchange the contribution from current two states
##             Exx += -.5 * f12 * num.dot(n,vr2dr)

