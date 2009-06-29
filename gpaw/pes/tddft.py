import numpy as np

from ase import *

from gpaw.pes import BasePES
from gpaw import *
import gpaw.mpi as mpi
from gpaw.lrtddft import *
from gpaw.utilities import packed_index


class TDDFTPES(BasePES):
    def __init__(self, Mother, ExcitedDaughter, Daughter=None):
        if ExcitedDaughter.calculator is not None:
            self.c_d = ExcitedDaughter.calculator
        else:
            self.c_d = Daughter

        self.c_m=Mother
        self.lr_d=ExcitedDaughter
        
        self.c_m.converge_wave_functions()
        self.c_d.converge_wave_functions()
        
        self.check_grids()

        # Make good way for initialising these

        self.imax=0
        for kpt in self.c_m.wfs.kpt_u:
            self.imax+=int(kpt.f_n.sum())
            
        self.kmax=self.imax-1                                #k=0,...,kmax-1  ### os ad 
        self.lmax=2*self.c_d.get_number_of_bands()-self.kmax #l=0,...,lmax-1

        self.qnr_m=self._create_qnr(self.c_m)
        self.qnr_d=self._create_qnr(self.c_d)

        self.f=None
        self.be=None

    def _calculate(self):
        self._create_d()
        self._create_h()
        self._create_g0()
        self._create_g()
        self._create_f()
        

    def _create_d(self):
        """Creates a matrix containing overlaps between KS orbitals"""

        # This is crap code... !!!!!!!
        self.d=np.zeros((self.imax,self.kmax+self.lmax))

        for i in range(0, self.imax):
            for j in range(0, self.kmax + self.lmax):
                if self.qnr_m[i,1]==self.qnr_d[j,1]:
                    ks_m=self.c_m.wfs.get_wave_function_array(self.qnr_m[i,0], 0,
                                                              self.qnr_m[i,1])
                    ks_d=self.c_d.wfs.get_wave_function_array(self.qnr_d[j,0], 0,
                                                              self.qnr_d[j,1])
#                    print "--- mpi.rank, ks_m, ks_d=", mpi.rank, type(ks_m), type(ks_d)
                    me=np.vdot(ks_m , ks_d) * self.c_m.wfs.gd.dv
                    self.c_m.wfs.gd.comm.sum(me)
                    
                    self.d[i,j]=me+self._nuc_corr(self.qnr_m[i,0],self.qnr_d[j,0],self.qnr_m[i,2],self.qnr_d[j,2])

    def _create_h(self):
        self.h=np.zeros((self.imax,self.kmax,self.lmax))
        for i in range(self.imax):
            for k in range(self.kmax):
                for l in range(self.lmax):
                    keep_row=range(self.imax)
                    keep_row.remove(i)

                    keep_col=range(self.kmax)                 
                    keep_col.remove(k)
                    keep_col.append(self.kmax+l)

                    d_ikl=np.zeros((len(keep_row),len(keep_col)))

                    for col in range(len(keep_col)):
                        for row in range(len(keep_row)):
                            d_ikl[row,col]=self.d[keep_row[row],keep_col[col]]
                    
                    self.h[i,k,l]=np.linalg.det(d_ikl)

    def _create_g0(self):
        self.g0=np.zeros((self.imax))
        for i in range(self.imax):
            keep_row=range(self.imax)
            keep_row.remove(i)

            keep_col=range(self.kmax)                 
            d_i00=np.zeros((len(keep_row),len(keep_col)))

            for col in range(len(keep_col)):
                for row in range(len(keep_row)):
                    d_i00[row,col]=self.d[keep_row[row],keep_col[col]]
                    
            self.g0[i]=(-1)**(self.imax+i)*np.linalg.det(d_i00)


    def _create_g(self):
        self.lr_d.diagonalize()
        self.g=np.zeros((len(self.lr_d)+1,self.imax))
        self.g[0,:]=self.g0

        for I in range(len(self.lr_d)):
            for i in range(self.imax):
                gi=0
                for kl in range(len(self.lr_d)):

                    for index in [2 * self.lr_d.kss[kl].i,
                                  2 * self.lr_d.kss[kl].i+1]:
                        if (self.qnr_d[index,0:2] == 
                            np.array([self.lr_d.kss[kl].i,
                                      self.lr_d.kss[kl].pspin])).all():
                            k=index
                    for index in [2 * self.lr_d.kss[kl].j,
                                  2 * self.lr_d.kss[kl].j+1]:
                        if len(self.c_d.wfs.kpt_u)==1 and self.c_d.wfs.kpt_u[0].f_n.sum()%2==1:
                            if (self.qnr_d[index,0:2] == 
                                np.array([self.lr_d.kss[kl].j,
                                          (self.lr_d.kss[kl].pspin+1) % 2])).any(): #lort men i lrtddft har sidste fyldte og 1. tomme samme spin...
                                l=index-self.kmax

                        else:
                            if (self.qnr_d[index,0:2]==np.array([self.lr_d.kss[kl].j,self.lr_d.kss[kl].pspin])).all():
                                l=index-self.kmax

                    gi+=self.lr_d[I].f[kl]*self.h[i,k,l]
                    l=None
                    k=None

                self.g[1+I,i]=(-1.)**(self.imax+i)*gi

    def _create_f(self):
        self.f=(self.g*self.g).sum(axis=1)
        be0=self.c_d.get_potential_energy()-self.c_m.get_potential_energy()

        self.be = be0 + np.array([0] + list(self.lr_d.GetEnergies()))

    def _nuc_corr(self, i_m, j_d, k_m, k_d):
        ma = 0
        
        for a, P_ni_m in self.c_m.wfs.kpt_u[k_m].P_ani.items():
            P_ni_d = self.c_d.wfs.kpt_u[k_d].P_ani.items()[a][1]
            Pi_i = P_ni_m[i_m]
            Pj_i = P_ni_d[j_d]
            Delta_pL = self.c_m.wfs.setups[a].Delta_pL
            
            for i in range(len(Pi_i)):
                for j in range(len(Pj_i)):
                    pij = Pi_i[i]*Pj_i[j]
                    ij = packed_index(i, j, len(Pi_i))
                    ma += Delta_pL[ij,0]*pij
            
        return sqrt(4 * pi) * ma
        
    def _create_qnr(self,c): #[n, spin, kpt]
        qnr=np.zeros((2 * c.get_number_of_bands(), 3,), dtype=int)

        if len(c.wfs.kpt_u)==1:
            for j in range(0, shape(qnr)[0], 2):
                qnr[j,0]=j/2
                qnr[j+1,0]=j/2
                qnr[j,1]=0
                qnr[j+1,1]=1

        if len(c.wfs.kpt_u)==2: # Make this properly 
            
            if c.wfs.kpt_u[0].f_n.sum()<c.wfs.kpt_u[1].f_n.sum():
                place0=1
                place1=0
            else:
                place0=0
                place1=1
            
            for j in range(len(c.wfs.kpt_u[0].f_n)):
                qnr[2*j+place0,0]=j
                qnr[2*j+place0,1]=c.wfs.kpt_u[0].s
                qnr[2*j+place0,2]=0
        
            for j in range(len(c.wfs.kpt_u[1].f_n)):
                qnr[2*j+place1,0]=j
                qnr[2*j+place1,1]=c.wfs.kpt_u[1].s
                qnr[2*j+place1,2]=1

        return qnr
                


    def check_grids(self):
        if (self.c_m.wfs.gd.cell_c != self.c_d.wfs.gd.cell_c).any():
            raise RuntimeError('Not the same grid')
        if (self.c_m.wfs.gd.h_c != self.c_d.wfs.gd.h_c).any():
            raise RuntimeError('Not the same grid')
        if (self.c_m.atoms.positions != self.c_m.atoms.positions).any():
            raise RuntimeError('Not the same atomic positions')


