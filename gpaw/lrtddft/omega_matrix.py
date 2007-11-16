import sys
from math import sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER

from gpaw import debug
import gpaw.mpi as mpi
from gpaw.poisson import PoissonSolver
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.transformers import Transformer
from gpaw.utilities import pack,pack2,packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XC3DGrid, XCFunctional

import time

"""This module defines a Omega Matrix class."""

class OmegaMatrix:
    """
    Omega matrix in Casidas linear response formalism

    Parameters
      - calculator: the calculator object the ground state calculation
      - kss: the Kohn-Sham singles object
      - xc: the exchange correlation approx. to use
      - derivativeLevel: which level i of d^i Exc/dn^i to use
      - numscale: numeric epsilon for derivativeLevel=0,1
      - filehandle: the oject can be read from a filehandle
      - out: output stream
      - finegrid: level of fine grid to use. 0: nothing, 1 for poisson only,
        2 everything on the fine grid
    """
    def __init__(self,
                 calculator=None,
                 kss=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001,
                 filehandle=None,
                 out=None,
                 finegrid=2
                 ):
        
        if filehandle is not None:
            self.kss = kss
            self.read(fh=filehandle)
            if out is None:
                if mpi.rank != MASTER: out = DownTheDrain()
                else: out = sys.stdout
            self.out = out
            return None

        self.paw = calculator
        if out is None: out = calculator.txt
        self.out = out
        self.fullkss = kss
        
        # handle different grid possibilities
        self.finegrid=finegrid
        self.restrict=None
        self.poisson = PoissonSolver()
        if finegrid:
            self.poisson.initialize(self.paw.finegd,
                                    self.paw.hamiltonian.poisson_stencil)
            self.gd = self.paw.finegd
            if finegrid==1:
                self.gd = self.paw.gd
        else:
            self.poisson.initialize(self.paw.gd,
                                    self.paw.hamiltonian.poisson_stencil)
            self.gd = self.paw.gd
        self.restrict = Transformer(self.paw.finegd, self.paw.gd,
                                    self.paw.stencils[0]).apply

        if xc == 'RPA': xc=None # enable RPA as keyword
        if xc is not None:
            self.xc = XC3DGrid(xc,self.gd,
                               kss.npspins)
            # check derivativeLevel
            if derivativeLevel is None:
                derivativeLevel=\
                    self.xc.get_functional().get_max_derivative_level()
            self.derivativeLevel=derivativeLevel
            # change the setup xc functional if needed
            # the ground state calculation may have used another xc
            for setup in self.paw.setups:
                sxc = setup.xc_correction.xc
                if sxc.xcfunc.xcname != xc:
                    sxc.set_functional(XCFunctional(xc,kss.npspins))
        else:
            self.xc = None

        self.numscale=numscale
    
        self.singletsinglet=False
        if kss.nvspins<2 and kss.npspins<2:
             # this will be a singlet to singlet calculation only
             self.singletsinglet=True

        self.full = self.get_full()

    def get_full(self):

        self.paw.timer.start('Omega RPA')
        Om = self.get_rpa()
        self.paw.timer.stop()

        if self.xc is None:
            return Om

        self.paw.timer.start('Omega XC')
        xcf=self.xc.get_functional()
        paw = self.paw
        fgd = paw.finegd
        comm = fgd.comm

        # shorthands
        fg = self.finegrid is 2
        kss = self.fullkss
        nij = len(kss)
        

        # initialize densities
        # nt_sg is the smooth density on the fine grid with spin index

        if kss.nvspins==2:
            # spin polarised ground state calc.
            nt_sg = paw.density.nt_sg
        else:
            # spin unpolarised ground state calc.
            if kss.npspins==2:
                # construct spin polarised densities
                nt_sg = num.array([.5*paw.density.nt_sg[0],
                                 .5*paw.density.nt_sg[0]])
            else:
                nt_sg = paw.density.nt_sg
        # restrict the density if needed
        if fg:
            nt_s = nt_sg
        else:
            nt_s = self.gd.zeros(nt_sg.shape[0])
            for s in range(nt_sg.shape[0]):
                self.restrict(nt_sg[s],nt_s[s])
                
        # initialize vxc or fxc

        if self.derivativeLevel==0:
            raise NotImplementedError
            if kss.npspins==2:
                v_g=nt_sg[0].copy()
            else:
                v_g=nt_sg.copy()
        elif self.derivativeLevel==1:
            pass
        elif self.derivativeLevel==2:
            raise NotImplementedError
            if kss.npspins==2:
                fxc=d2Excdnsdnt(nt_sg[0],nt_sg[1])
            else:
                fxc=d2Excdn2(nt_sg)
        else:
            raise ValueError('derivativeLevel can only be 0,1,2')
            
        ns=self.numscale
        xc=self.xc
        print >> self.out, 'XC',nij,'transitions'
        for ij in range(nij):
            print >> self.out,'XC kss['+'%d'%ij+']' 

            timer = Timer()
            timer.start('init')
            timer2 = Timer()
                      
            if self.derivativeLevel == 1:
                # vxc is available
                # We use the numerical two point formula for calculating
                # the integral over fxc*n_ij. The results are
                # vvt_s        smooth integral
                # nucleus.I_sp atom based correction matrices (pack2)
                #              stored on each nucleus
                timer2.start('init v grids')
                vp_s=num.zeros(nt_s.shape,nt_s.typecode())
                vm_s=num.zeros(nt_s.shape,nt_s.typecode())
                if kss.npspins==2: # spin polarised
                    nv_s=nt_s.copy()
                    nv_s[kss[ij].pspin] += ns*kss[ij].GetPairDensity(fg)
                    xc.get_energy_and_potential(nv_s[0],vp_s[0],
                                                nv_s[1],vp_s[1])
                    nv_s=nt_s.copy()
                    nv_s[kss[ij].pspin] -= ns*kss[ij].GetPairDensity(fg)
                    xc.get_energy_and_potential(nv_s[0],vm_s[0],
                                                nv_s[1],vm_s[1])
                else: # spin unpolarised
                    nv=nt_s[0] + ns*kss[ij].GetPairDensity(fg)
                    xc.get_energy_and_potential(nv,vp_s[0])
                    nv=nt_s[0] - ns*kss[ij].GetPairDensity(fg)
                    xc.get_energy_and_potential(nv,vm_s[0])
                vvt_s = (.5/ns)*(vp_s-vm_s)
                timer2.stop()

                # initialize the correction matrices
                timer2.start('init v corrections')
                for nucleus in self.paw.my_nuclei:
                    # create the modified density matrix
                    Pi_i = nucleus.P_uni[kss[ij].spin,kss[ij].i]
                    Pj_i = nucleus.P_uni[kss[ij].spin,kss[ij].j]
                    P_ii = num.outerproduct(Pi_i,Pj_i)
                    # we need the symmetric form, hence we can pack
                    P_p = pack(P_ii,tolerance=1e30)
                    D_sp = nucleus.D_sp.copy()
                    D_sp[kss[ij].spin] += ns*P_p
                    nucleus.I_sp = \
                                 nucleus.setup.xc_correction.\
                                 two_phi_integrals(D_sp)
                    D_sp = nucleus.D_sp.copy()
                    D_sp[kss[ij].spin] -= ns*P_p
                    nucleus.I_sp -= \
                                 nucleus.setup.xc_correction.\
                                 two_phi_integrals(D_sp)
                    nucleus.I_sp /= 2.*ns
                timer2.stop()
                    
            timer.stop()
            t0 = timer.gettime('init')
            timer.start(ij)
            
            for kq in range(ij,nij):
                weight=2.*sqrt(kss[ij].GetEnergy()*kss[kq].GetEnergy()*
                               kss[ij].GetWeight()*kss[kq].GetWeight())

                if self.derivativeLevel == 0:
                    # only Exc is available
                    
                    if kss.npspins==2: # spin polarised
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] +=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] +=\
                                        kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(\
                                        nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] +=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] -= \
                                        kss[kq].GetFineGridPairDensity()
                        Excpm = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] -=\
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] +=\
                                        kss[kq].GetFineGridPairDensity()
                        Excmp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                        nv_g = nt_sg.copy()
                        nv_g[kss[ij].pspin] -= \
                                        kss[ij].GetFineGridPairDensity()
                        nv_g[kss[kq].pspin] -=\
                                        kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(\
                                            nv_g[0],v_g,nv_g[1],v_g)
                    else: # spin unpolarised
                        nv_g=nt_sg + ns*kss[ij].GetFineGridPairDensity()\
                              + ns*kss[kq].GetFineGridPairDensity()
                        Excpp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg + ns*kss[ij].GetFineGridPairDensity()\
                              - ns*kss[kq].GetFineGridPairDensity()
                        Excpm = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg - ns*kss[ij].GetFineGridPairDensity()\
                              + ns*kss[kq].GetFineGridPairDensity()
                        Excmp = xc.get_energy_and_potential(nv_g,v_g)
                        nv_g=nt_sg - ns*kss[ij].GetFineGridPairDensity()\
                              - ns*kss[kq].GetFineGridPairDensity()
                        Excmm = xc.get_energy_and_potential(nv_g,v_g)

                    Om[ij,kq] += weight *\
                                0.25*(Excpp-Excmp-Excpm+Excmm)/(ns*ns)
                              
                elif self.derivativeLevel == 1:
                    # vxc is available

                    timer2.start('integrate')
                    Om[ij,kq] += weight*\
                                 self.gd.integrate(kss[kq].GetPairDensity(fg)*
                                                   vvt_s[kss[kq].pspin])
                    timer2.stop()

                    timer2.start('integrate corrections')
                    Exc = 0.
                    for nucleus in self.paw.my_nuclei:
                        # create the modified density matrix
                        Pk_i = nucleus.P_uni[kss[kq].spin,kss[kq].i]
                        Pq_i = nucleus.P_uni[kss[kq].spin,kss[kq].j]
                        P_ii = num.outerproduct(Pk_i,Pq_i)
                        # we need the symmetric form, hence we can pack
                        # use pack as I_sp used pack2
                        P_p = pack(P_ii,tolerance=1e30)
                        Exc += num.dot(nucleus.I_sp[kss[kq].spin],P_p)
                    Om[ij,kq] += weight * self.gd.comm.sum(Exc)
                    timer2.stop()

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

            timer.stop()
##            timer2.write()
            if ij < (nij-1):
                t = timer.gettime(ij) # time for nij-ij calculations
                t = .5*t*(nij-ij)  # estimated time for n*(n+1)/2, n=nij-(ij+1)
                print >> self.out,'XC estimated time left',\
                      self.timestring(t0*(nij-ij-1)+t)

        self.paw.timer.stop()
        return Om

    def get_rpa(self):
        """calculate RPA part of the omega matrix"""

        # shorthands
        kss=self.fullkss
        finegrid=self.finegrid

        # calculate omega matrix
        nij = len(kss)
        print >> self.out,'RPA',nij,'transitions'
        
        Om = num.zeros((nij,nij),num.Float)
        
        for ij in range(nij):
            print >> self.out,'RPA kss['+'%d'%ij+']=', kss[ij]

            timer = Timer()
            timer.start('init')
            timer2 = Timer()
                      
            # smooth density including compensation charges
            timer2.start('GetPairDensityAndCompensationCharges 0')
            rhot_p = kss[ij].GetPairDensityAndCompensationCharges(
                finegrid is not 0)
            timer2.stop()
            
            # integrate with 1/|r_1-r_2|
            timer2.start('poisson')
            phit_p = num.zeros(rhot_p.shape,rhot_p.typecode())
            self.poisson.solve(phit_p,rhot_p,charge=None)
            timer2.stop()

            timer.stop()
            t0 = timer.gettime('init')
            timer.start(ij)

            if finegrid == 1:
                rhot = kss[ij].GetPairDensityAndCompensationCharges()
                phit = self.gd.zeros()
##                print "shapes 0=",phit.shape,rhot.shape
                self.restrict(phit_p,phit)
            else:
                phit = phit_p
                rhot = rhot_p

            for kq in range(ij,nij):
                if kq != ij:
                    # smooth density including compensation charges
                    timer2.start('kq GetPairDensityAndCompensationCharges')
                    rhot = kss[kq].GetPairDensityAndCompensationCharges(
                        finegrid is 2)
                    timer2.stop()

                timer2.start('integrate')
                pre = 2.*sqrt(kss[ij].GetEnergy()*kss[kq].GetEnergy()*
                                  kss[ij].GetWeight()*kss[kq].GetWeight())
                Om[ij,kq]= pre * self.gd.integrate(rhot*phit)
##                print "int=",Om[ij,kq]
                timer2.stop()

                # Add atomic corrections
                timer2.start('integrate corrections 2')
                Ia = 0.
                for nucleus in self.paw.my_nuclei:
                    ni = nucleus.get_number_of_partial_waves()
                    Pi_i = nucleus.P_uni[kss[ij].spin,kss[ij].i]
                    Pj_i = nucleus.P_uni[kss[ij].spin,kss[ij].j]
                    Dij_ii = num.outerproduct(Pi_i, Pj_i)
                    Dij_p = pack(Dij_ii, tolerance=1e3)
                    Pk_i = nucleus.P_uni[kss[kq].spin,kss[kq].i]
                    Pq_i = nucleus.P_uni[kss[kq].spin,kss[kq].j]
                    Dkq_ii = num.outerproduct(Pk_i, Pq_i)
                    Dkq_p = pack(Dkq_ii, tolerance=1e3)
                    C_pp = nucleus.setup.M_pp
                    #   ----
                    # 2 >      P   P  C    P  P
                    #   ----    ip  jr prst ks qt
                    #   prst
                    Ia += 2.0*num.dot(Dkq_p,num.dot(C_pp,Dij_p))
                timer2.stop()
                
                Om[ij,kq] += pre * self.gd.comm.sum(Ia)
                    
                if ij == kq:
                    Om[ij,kq] += kss[ij].GetEnergy()**2
                else:
                    Om[kq,ij]=Om[ij,kq]

            timer.stop()
##            timer2.write()
            if ij < (nij-1):
                t = timer.gettime(ij) # time for nij-ij calculations
                t = .5*t*(nij-ij)  # estimated time for n*(n+1)/2, n=nij-(ij+1)
                print >> self.out,'RPA estimated time left',\
                      self.timestring(t0*(nij-ij-1)+t)

        return Om

    def timestring(self,t):
        ti = int(t+.5)
        td = int(ti/86400)
        st=''
        if td>0:
            st+='%d'%td+'d'
            ti-=td*86400
        th = int(ti/3600)
        if th>0:
            st+='%d'%th+'h'
            ti-=th*3600
        tm = int(ti/60)
        if tm>0:
            st+='%d'%tm+'m'
            ti-=tm*60
        st+='%d'%ti+'s'
        return st

    def diagonalize(self, istart=None, jend=None):
        self.istart = istart
        self.jend = jend
        if istart is None and jend is None:
            # use the full matrix
            kss = self.fullkss
            evec = self.full.copy()
        else:
            # reduce the matrix
            if istart is None: istart = self.kss.istart
            if self.fullkss.istart > istart:
                raise RuntimeError('istart=%d has to be >= %d' %
                                   (istart,self.kss.istart))
            if jend is None: jend = self.kss.jend
            if self.fullkss.jend < jend:
                raise RuntimeError('jend=%d has to be <= %d' %
                                   (jend,self.kss.jend))
            print >> self.out,'# diagonalize: %d transitions original'\
                  % len(self.fullkss)
            map= []
            kss = KSSingles()
            for ij, k in zip(range(len(self.fullkss)),self.fullkss):
                if k.i >= istart and k.j <= jend:
                    kss.append(k)
                    map.append(ij)
            kss.update()
            nij = len(kss)
            print >> self.out,'# diagonalize: %d transitions now' % nij

            evec = num.zeros((nij,nij),num.Float)
            for ij in range(nij):
                for kq in range(nij):
                    evec[ij,kq] = self.full[map[ij],map[kq]]

        self.eigenvectors = evec        
        self.eigenvalues = num.zeros((len(kss)),num.Float)
        self.kss = kss
        info = diagonalize(self.eigenvectors, self.eigenvalues)
        if info != 0:
            raise RuntimeError('Diagonalisation error in OmegaMatrix')

    def Kss(self,kss=None):
        """Set and get own Kohn-Sham singles"""
        if kss is not None:
            self.fullkss = kss
        if(hasattr(self,'fullkss')):
            return self.fullkss
        else:
            return None
 
    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'r')
            else:
                f = fh

            f.readline()
            nij = int(f.readline())
            full = num.zeros((nij,nij),num.Float)
            for ij in range(nij):
                l = f.readline().split()
                for kq in range(ij,nij):
                    full[ij,kq] = float(l[kq-ij])
                    full[kq,ij] = full[ij,kq]
            self.full = full

            if fh is None:
                f.close()

    def write(self, filename=None, fh=None):
        """Write current state to a file."""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'w')
            else:
                f = fh

            f.write('# OmegaMatrix\n')
            nij = len(self.fullkss)
            f.write('%d\n' % nij)
            for ij in range(nij):
                for kq in range(ij,nij):
                    f.write(' %g' % self.full[ij,kq])
                f.write('\n')
            
            if fh is None:
                f.close()


    def __str__(self):
        str='<OmegaMatrix> '
        if hasattr(self,'eigenvalues'):
            str += 'dimension '+ ('%d'%len(self.eigenvalues))
            str += "\neigenvalues: "
            for ev in self.eigenvalues:
                str += ' ' + ('%f'%(sqrt(ev)*27.211))
        return str
    


