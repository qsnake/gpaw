import sys
from math import sqrt
import Numeric as num
import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER

from gpaw import debug
from gpaw.poisson_solver import PoissonSolver
from gpaw.lrtddft.excitation import Excitation,ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.utilities import packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XC3DGrid, XCFunctional

"""This module defines a Omega Matrix class."""

class OmegaMatrix:
    """Omega matrix in Casidas linear response formalism
    """
    def __init__(self,
                 calculator=None,
                 kss=None,
                 xc=None,
                 derivativeLevel=None,
                 numscale=0.001,
                 filehandle=None,
                 out=None,
                 ):
        
        if filehandle is not None:
            self.read(fh=filehandle)
            if out is None: out = sys.stdout
            self.out = out
            return None

        self.calculator = calculator
        if out is None: out = calculator.out
        self.out = out
        self.fullkss = kss
        if xc is not None:
            self.xc = XC3DGrid(xc,self.calculator.paw.finegd,
                               kss.npspins)
            # check derivativeLevel
            if derivativeLevel is None:
                derivativeLevel=\
                    self.xc.get_functional().get_max_derivative_level()
            self.derivativeLevel=derivativeLevel
        else:
            self.xc = None

        self.numscale=numscale
    
        self.singletsinglet=False
        if kss.nvspins<2 and kss.npspins<2:
             # this will be a singlet to singlet calculation only
             self.singletsinglet=True

        self.full = self.get_full()

    def get_full(self,rpa=None):
        if rpa is None:
            rpa = self.get_rpa()
        Om = rpa

        if self.xc is None:
            return Om
        
        xcf=self.xc.get_functional()
        print '<OmegaMatrix::get_full> xc=',xcf.get_name()
        print '<OmegaMatrix::get_full> derivative Level=',self.derivativeLevel
        print '<OmegaMatrix::get_full> numscale=',self.numscale

        paw = self.calculator.paw
        gd = paw.finegd    
        kss=self.fullkss
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

                    Exc = 0
                    for nucleus in self.my_nuclei:
                        D_sp = nucleus.D_sp
                        H_sp = num.zeros(D_sp.shape, num.Float) # not used for anything!
                        xc_correction = nucleus.setup.xc_correction
                        Exc += xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)


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

    def get_rpa(self):
        """calculate RPA part of the omega matrix"""
        paw = self.calculator.paw
        gd = paw.finegd
        poisson = PoissonSolver(gd, paw.hamiltonian.poisson_stencil)
        kss=self.fullkss

        # calculate omega matrix
        nij = len(kss)
        print >> self.out,'RPA',nij,'transitions'
        
        phit_g = gd.new_array()
        Om = num.zeros((nij,nij),num.Float)
        for ij in range(nij):
            print >> self.out,'RPA kss['+'%d'%ij+']=', kss[ij]

            # smooth density including compensation charges
            rhot_g = kss[ij].GetFineGridPairDensity()

            # integrate with 1/|r_1-r_2|
            poisson.solve(phit_g,rhot_g,charge=None,maxcharge=1e-12)
            
            for kq in range(ij,nij):
                if kq != ij:
                    # smooth density including compensation charges
                    rhot_g = kss[kq].GetFineGridPairDensity()

                pre = 2.*sqrt(kss[ij].GetEnergy()*kss[kq].GetEnergy()*
                                  kss[ij].GetWeight()*kss[kq].GetWeight())
                
                Om[ij,kq]= pre * gd.integrate(rhot_g*phit_g)

                # Add atomic corrections
                Ia = 0
                for nucleus in paw.my_nuclei:
                    ni = nucleus.get_number_of_partial_waves()
                    Pi_i = nucleus.P_uni[kss[ij].vspin,kss[ij].i]
                    Pj_i = nucleus.P_uni[kss[ij].vspin,kss[ij].j]
                    Pk_i = nucleus.P_uni[kss[kq].vspin,kss[kq].i]
                    Pq_i = nucleus.P_uni[kss[kq].vspin,kss[kq].j]
                    C_pp = nucleus.setup.M_pp

                    #   ----
                    # 2 >      P   P  C    P  P
                    #   ----    ip  jr prst ks qt
                    #   prst
                    for p in range(ni):
                        for r in range(ni):
                            pr = packed_index(p, r, ni)
                            for s in range(ni):
                                for t in range(ni):
                                    st = packed_index(s, t, ni)
                                    Ia += Pi_i[p]*Pj_i[r]*\
                                          2*C_pp[pr, st]*\
                                          Pk_i[s]*Pq_i[t]
                Om[ij,kq] += pre*Ia
                    
                if ij == kq:
                    Om[ij,kq] += kss[ij].GetEnergy()**2
                else:
                    Om[kq,ij]=Om[ij,kq]

        return Om

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
    


