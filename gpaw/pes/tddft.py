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

        self.c_m = Mother
        self.gd = self.c_m.wfs.gd
        self.lr_d = ExcitedDaughter
        
        self.c_m.converge_wave_functions()
        self.c_d.converge_wave_functions()
        self.lr_d.diagonalize()
        
        self.check_systems()
        self.lr_d.jend=self.lr_d.kss[-1].j        
	
        # Make good way for initialising these

        self.imax=0
        for kpt in self.c_m.wfs.kpt_u:
            self.imax+=int(kpt.f_n.sum()) # Isn't there a 'number of electrons' varible?
            
        self.kmax=self.imax-1                                #k=0,...,kmax-1  ### os ad 
        self.lmax=2*(self.lr_d.jend+1)-self.kmax #l=0,...,lmax-1

        self.qnr_m=self._create_qnr(self.c_m , self.c_m.get_number_of_bands())
        self.qnr_d=self._create_qnr(self.c_d,self.lr_d.jend+1)

        self.f=None
        self.be=None
	self.first_peak_energy=None

    def _calculate(self):
        self._create_d()
        self._create_h()
        self._create_g0()
        self._create_g()
        self._create_f()
        

    def _create_d(self):
        """Creates a matrix containing overlaps between KS orbitals"""

        self.d=np.zeros((self.imax,self.kmax+self.lmax))

        for i in range(0, self.imax):
            s_m = self.qnr_m[i,1]
            n_m = self.qnr_m[i,0]
            for j in range(0, self.kmax + self.lmax):
                s_d = self.qnr_d[j,1]
                n_d = self.qnr_d[j,0]
                if s_m == s_d:
                    ks_m = self.c_m.wfs.kpt_u[s_m].psit_nG[n_m]
                    ks_d = self.c_d.wfs.kpt_u[s_d].psit_nG[n_d]
                    me = self.gd.integrate(ks_m * ks_d)
                        
                    self.d[i,j] = me + self._nuc_corr(self.qnr_m[i,0],
                                                      self.qnr_d[j,0],
                                                      self.qnr_m[i,2],
                                                      self.qnr_d[j,2])

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
        totspin=int(np.abs(self.c_d.get_magnetic_moment()))
        self.g=np.zeros((len(self.lr_d)+1,self.imax))
        self.g[0,:]=self.g0

        for I in range(len(self.lr_d)):
            
            for i in range(self.imax):
                gi=0
                for kl in range(len(self.lr_d)):

                    for index in range(2*self.lr_d.kss[kl].i-totspin, 2 * self.lr_d.kss[kl].i+2): 
                        if (self.qnr_d[index,0:2] == 
                            np.array([self.lr_d.kss[kl].i,
                                      self.lr_d.kss[kl].pspin])).all():
                            k=index
                            break
                        

                    for index in range(2*self.lr_d.kss[kl].j, 2*self.lr_d.kss[kl].j+2+totspin):

                        if len(self.c_d.wfs.kpt_u)==1 and self.c_d.wfs.kpt_u[0].f_n.sum()%2==1:
                            if (self.qnr_d[index,0:2] == 
                                np.array([self.lr_d.kss[kl].j,
                                          (self.lr_d.kss[kl].pspin+1) % 2])).any():
                                #Crap but in non spinpol lrtddft of open shell systems the HOMO and LUMO have equal quantum numbers, or so it seams 
                                l=index-self.kmax
                                break

                        else:
                            if (self.qnr_d[index,0:2]==np.array([self.lr_d.kss[kl].j,self.lr_d.kss[kl].pspin])).all():
                                l=index-self.kmax
                                break

                    gi+=self.lr_d[I].f[kl]*self.h[i,k,l]
                    l=None
                    k=None

                self.g[1+I,i]=(-1.)**(self.imax+i)*gi

    def _create_f(self):
        self.f=(self.g*self.g).sum(axis=1)

        if self.first_peak_energy==None:
            self.first_peak_energy=(self.c_d.get_potential_energy()
                                  -self.c_m.get_potential_energy())

        self.be = self.first_peak_energy + np.array([0] + list(self.lr_d.GetEnergies()))

    def _nuc_corr(self, i_m, j_d, k_m, k_d):
        ma = 0
        
        for a, P_ni_m in self.c_m.wfs.kpt_u[k_m].P_ani.items():
            P_ni_d = self.c_d.wfs.kpt_u[k_d].P_ani.items()[a][1]
            Pi_i = P_ni_m[i_m]
            Pj_i = P_ni_d[j_d]
            Delta_pL = self.c_m.wfs.setups[a].Delta_pL
            
            for i in range(len(Pi_i)):
                for j in range(len(Pj_i)):
                    pij = Pi_i[i] * Pj_i[j]
                    ij = packed_index(i, j, len(Pi_i))
                    ma += Delta_pL[ij, 0] * pij

#        print "0 mpi.rank, ma=", mpi.rank, ma, i_m, j_d, k_m, k_d, id(self.gd.comm), self.gd.comm
        self.gd.comm.sum(ma)
#        print "1 mpi.rank, ma=", mpi.rank, ma, i_m, j_d, k_m, k_d
        return sqrt(4 * pi) * ma
        
    def _create_qnr(self,c,nmax): #[n, spin, kpt, occ?]
        qnr=np.zeros((2 * nmax , 4,), dtype=int)

        if len(c.wfs.kpt_u)==1:
            for j in range(0, 2 * nmax, 2): 
                qnr[j,0]=j/2
                qnr[j+1,0]=j/2
                qnr[j,1]=0
                qnr[j+1,1]=1

        if len(c.wfs.kpt_u)==2: # Make this properly 
            
            for j in range(nmax):
                qnr[2*j,0]=j
                qnr[2*j,1]=c.wfs.kpt_u[0].s
                qnr[2*j,2]=0
                qnr[2*j,3]=-c.wfs.kpt_u[0].f_n[j]
        
            for j in range(nmax):
                qnr[2*j+1,0]=j
                qnr[2*j+1,1]=c.wfs.kpt_u[1].s
                qnr[2*j+1,2]=1
                qnr[2*j+1,3]=-c.wfs.kpt_u[1].f_n[j]

            qnr=qnr[qnr[:,3].argsort(),]
            qnr=np.abs(qnr) # Haha my code is so ugly that its funny... Want to sort but with 1 comming before 0           
            if 2*nmax>(c.wfs.kpt_u[1].f_n+c.wfs.kpt_u[0].f_n).sum():
                qnr_empty=qnr[int((c.wfs.kpt_u[1].f_n+c.wfs.kpt_u[0].f_n).sum()):,:]
                qnr_empty=qnr_empty[qnr_empty[:,0].argsort(),]
                qnr[int((c.wfs.kpt_u[1].f_n+c.wfs.kpt_u[0].f_n).sum()):,:]=qnr_empty

        return qnr


    def check_systems(self):
        if (self.c_m.wfs.gd.cell_c != self.c_d.wfs.gd.cell_c).any():
            raise RuntimeError('Not the same grid')
        if (self.c_m.wfs.gd.h_c != self.c_d.wfs.gd.h_c).any():
            raise RuntimeError('Not the same grid')
        if (self.c_m.atoms.positions != self.c_m.atoms.positions).any():
            raise RuntimeError('Not the same atomic positions')
        #if np.abs(self.c_m.get_magnetic_moment()-self.c_d.get_magnetic_moment())!=1.:
            #raise RuntimeError('Mother and daughter spin are not compatible')
        # Make number of electrons check...

