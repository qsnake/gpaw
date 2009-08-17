import cPickle as pickle
import numpy as npy
import numpy as np
from os.path import isfile
from ase.vibrations import Vibrations
from ase.parallel import rank, barrier
from gpaw.utilities import unpack2
from gpaw.lcao.pwf2 import PWF2
from gpaw.utilities.blas import r2k, gemm
from gpaw import GPAW
from gpaw.lcao.projected_wannier import dots
from gpaw.utilities.tools import tri2full
from ase import Bohr
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from ase.units import Bohr, Hartree

"""This module is used to calculate the electron-phonon coupling matrix,
    expressed in terms of GPAW LCAO orbitals."""

class ElectronPhononCouplingMatrix:
    """Class for calculating the electron-phonon coupling matrix, defined
       by the electron phonon interaction::
                      _                   _____
                      \     l   cc        /  h             cc
            H      =   )   M   c   c     /------   ( b  + b   )
             el-ph    /_    ij  i   j  \/   2 W       l    l
                        l,ij                     l
        
            where the electron phonon coupling matrix is given by
                    
                l           _ 
                M   = < i | \ /  V   * v  |j>
                 ij          'u   eff   l
  
          ::
    """
    def __init__(self, atoms, indices=None, name = 'v',delta=0.005, nfree=2):
        assert nfree in [2,4]
        self.nfree = nfree
        self.delta = delta
        
        if indices is None:
            indices = range(len(self.atoms))
        self.calc = atoms.get_calculator()
        self.atoms = atoms
        self.indices = np.asarray(indices)
        self.name = name
        self.p0 = self.atoms.positions.copy()
    

    def run(self):
        if not isfile(self.name + '.eq.pckl'):
            barrier()
            if rank == 0:
                vd = open(self.name + '.eq.pckl', 'wb')
                fd = open('vib.eq.pckl','wb')
                bfsd = open('data.pckl','wb')

            self.calc.calculate(self.atoms)
            Vt_G = self.calc.get_effective_potential()
            Vt_G = self.calc.gd.collect(Vt_G,broadcast=False)/Hartree
            dH_asp = self.calc.hamiltonian.dH_asp
            forces = self.atoms.get_forces()
            self.calc.write('eq.gpw')
            wfs = self.calc.wfs
            bfs = wfs.basis_functions
            nao = wfs.setups.nao
            C_MM = np.identity(nao)
            gd = wfs.gd
            phi_MG = gd.zeros(nao)
            bfs.lcao_to_grid(C_MM, phi_MG, 0)

            # Atomic correction stuff
            wfs = self.calc.wfs
            setups = wfs.setups
            pt = LFC(wfs.gd, [setup.pt_j for setup in setups],
                     wfs.kpt_comm, dtype=wfs.dtype, forces=True)
            spos_ac = self.atoms.get_scaled_positions()
            pt.set_positions(spos_ac)
            P_ani = pt.dict(len(phi_MG))
            pt.integrate(phi_MG,P_ani)           
            
            dP_anix = pt.dict(len(phi_MG), derivative = True)
            pt.derivative(phi_MG, dP_anix)

            if rank == 0:
                pickle.dump((Vt_G, dH_asp), vd,2)
                pickle.dump(forces,fd)
                pickle.dump({'C_MM': C_MM, 
                             'phi_MG': phi_MG, 
                             'dv': self.calc.gd.dv,
                             'P_ani': P_ani,
                             'dP_anix': dP_anix   
                        }, bfsd,2) 
                bfsd.close()
                vd.close()
                fd.close()
        
        p = self.atoms.positions.copy()
        for a in self.indices:
            for j in range(3):
                for sign in [-1,1]:
                    for ndis in range(1,self.nfree/2+1):       
                        name = '.%d%s%s.pckl' % (a,'xyz'[j], ndis*' +-'[sign])
                        if isfile(self.name + name):
                            continue
                        barrier()
                        if rank == 0:
                            vd = open(self.name + name , 'w')
                            fd = open('vib'+name, 'w')
                        self.atoms.positions[a,j]=p[a,j] + sign*ndis*self.delta
                        self.calc.calculate(self.atoms)
                        Vt_G = self.calc.get_effective_potential()
                        Vt_G =self.calc.gd.collect(Vt_G,broadcast=False)/Hartree
                        print Vt_G.shape
                        dH_asp = self.calc.hamiltonian.dH_asp
                        forces = self.atoms.get_forces()
                        if rank == 0:
                            pickle.dump((Vt_G,dH_asp),vd)
                            pickle.dump(forces,fd)
                            vd.close()
                            fd.close()
                        self.atoms.positions[a,j]=p[a,j]
        self.atoms.set_positions(p)
    
    def get_gradient(self):
        """Calculates gradient"""
        nx = len(self.indices)*3	
        veqt_G, dHeq_asp = pickle.load(open(self.name+'.eq.pckl'))
        gpts = veqt_G.shape 
        dvt_Gx = npy.zeros(gpts+(nx,)) 
        ddH_aspx = {}
        for a, dH_sp in dHeq_asp.items():
            ddH_aspx[a] = npy.empty(dH_sp.shape + (nx,))

        x = 0
        for a in self.indices:
            for i in range(3):
                name = '%s.%d%s' % (self.name,a,'xyz'[i])
                vtm_G, dHm_asp = pickle.load(open(name + '-.pckl'))
                vtp_G, dHp_asp = pickle.load(open(name + '+.pckl'))
                

                if self.nfree==4:
                    vtmm_G, dHmm_asp = pickle.load(open(name+'--.pckl'))
                    vtpp_G, dHpp_asp = pickle.load(open(name+'++.pckl'))
                    dvtdx_G = (-vtpp_G+8.0*vtp_G
                                -8.0*vtm_G+vtmm_G)/(12.0*self.delta/Bohr)
                    dvt_Gx[:,:,:,x] = dvtdx_G
                    for atom, ddH_spx in ddH_aspx.items():
                        ddH_aspx[atom][:,:,x] =(-dHpp_asp[atom]
                                                +8.0*dHp_asp[atom]
                                                -8.0*dHm_asp[atom]
                                                +dHmm_asp[atom])/(12*self.delta/Bohr)
                else: # nfree = 2
                    dvtdx_G = (vtp_G-vtm_G)/(2*self.delta/Bohr)
                    dvt_Gx[:,:,:,x] = dvtdx_G
                    for atom, ddH_spx in ddH_aspx.items():
                        ddH_aspx[atom][:, :, x] = (dHp_asp[atom]
                                    -dHm_asp[atom])/(2 * self.delta/Bohr)
                x+=1
        return dvt_Gx, ddH_aspx

    def get_Mlii(self, modes, vtonly = False):

        """
          ::
                  d                   d  ~
            < w | -- v | w' > = < w | -- v | w'>
                  dP                  dP

                               _
                              \        ~a     d   .       ~a
                            +  ) < w | p  >   -- /_\H   < p | w' >
                              /_        i     dP     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < w | -- p  >  /_\H    < p | w' >
                              /_       dP  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < w | p  >  /_\H    < -- p  | w' >
                              /_        i        ij    dP  j
                              a,ij

            ::
        """
 
        d  = pickle.load(open('data.pckl')) # bfs
        phi_MG = d['phi_MG']
        dv = d['dv']
        dvt_Gx, ddH_aspx = self.get_gradient()
	dim =len(phi_MG)
        M_lii = {}
        for f, mode in modes.items():
            mo = []    
            M_ii=np.zeros((dim,dim))
            for a in self.indices:
                mo.append(mode[a])
            mode = np.asarray(mo).flatten()
            dvtdP_G = np.dot(dvt_Gx,mode)   
        # calculate upper part of the coupling matrix. The integrations
        # should be done like this to avoid memory problems for larger
        # systems
            for i in range(dim): 
                for j in range(i,dim):
                    phi_i = np.expand_dims(phi_MG[i],axis=0)
                    phi_j = np.expand_dims(phi_MG[j],axis=0)
                    dvdP_ij = np.zeros((1,1),dtype=phi_i.dtype)
                    r2k(.5*dv,phi_i,dvtdP_G*phi_j,0.,dvdP_ij)
                    M_ii[i,j]=dvdP_ij[0,0] 
            tri2full(M_ii,'U')
            M_lii[f]=M_ii               
           
        if vtonly:
            for mode, M_ii in M_lii.items():
                M_lii[mode]*=Hartree/Bohr
            
            return M_lii

        P_ani = d['P_ani']
        # Add the term
        #  _
        # \        ~a     d   .       ~a
        #  ) < w | p  >   -- /_\H   < p | w' >
        # /_        i     dP     ij    j
        # a,ij

        Ma_lii = {}
        for f,mode in modes.items():
            Ma_lii[f]=np.zeros_like(M_lii.values()[0])
        
        spin = 0
        for f, mode in modes.items():
            mo = []
            for a in self.indices:
                mo.append(mode[a])
            mode = np.asarray(mo).flatten()
            
            for a, ddH_spx in ddH_aspx.items():
                ddHdP_sp = np.dot(ddH_spx, mode)
                ddHdP_ii = unpack2(ddHdP_sp[spin])
                Ma_lii[f]+=dots(P_ani[a], ddHdP_ii, P_ani[a].T)
        
        dP_anix = d['dP_anix']
        dH_asp = pickle.load(open('v.eq.pckl'))[1]
        
        Mb_lii = {}
        for f,mode in modes.items():
            Mb_lii[f]=np.zeros_like(M_lii.values()[0])

        for f, mode in modes.items():
            for a, dP_nix in dP_anix.items():
                dPdP_ni = npy.dot(dP_nix,mode[a])
                dH_ii = unpack2(dH_asp[a][0])    
                dPdP_MM = dots(dPdP_ni,dH_ii,P_ani[a].T) 
                Mb_lii[f]+=dPdP_MM + dPdP_MM.T           
        
        
        for mode, M_ii in M_lii.items():
            M_lii[mode]*=Hartree/Bohr
            Ma_lii[mode]*=Hartree/Bohr
            Mb_lii[mode]*=Hartree/Bohr
        
        for mode in M_lii.keys():
            M_lii[mode] += Ma_lii[mode] + Mb_lii[mode]

        return M_lii

        

    def get_Mlii2(self, modes, atoms, calc, vtonly=True):
        """bla bla ..
        
        ::
        
                  d                   d  ~
            < w | -- v | w' > = < w | -- v | w'>
                  dP                  dP

                               _
                              \        ~a     d   .       ~a
                            +  ) < w | p  >   -- /_\H   < p | w' >
                              /_        i     dP     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < w | -- p  >  /_\H    < p | w' >
                              /_       dP  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < w | p  >  /_\H    < -- p  | w' >
                              /_        i        ij    dP  j
                              a,ij


        """
        from gpaw import restart
        atoms, calc = restart('eq.gpw')
        spos_ac = atoms.get_scaled_positions()
        calc.initialize(atoms)
        calc.initialize_positions(atoms)
        dvt_Gx, ddH_aspx = self.get_gradient()
        nao = calc.wfs.setups.nao
        bfs = calc.wfs.basis_functions       

        M_lii = {}
        for f, mode in modes.items():
            mo = []    
            M_ii=np.zeros((nao,nao))
            for a in self.indices:
                mo.append(mode[a])
            mode = np.asarray(mo).flatten()
            dvtdP_G = np.dot(dvt_Gx,mode)   
            bfs.calculate_potential_matrix(dvtdP_G,M_ii,q=0)
            tri2full(M_ii,'L')
            M_lii[f]=M_ii               
           
        if vtonly:
            for mode, M_ii in M_lii.items():
                M_lii[mode]*=Hartree/Bohr
            
            return M_lii
