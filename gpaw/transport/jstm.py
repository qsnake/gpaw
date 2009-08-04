import time
import numpy as np
from ase.units import Bohr, Hartree
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import NewLocalizedFunctionsCollection as LFC
from gpaw.utilities.tools import tri2full
import gpaw.mpi as mpi
from gpaw.lcao.tools import remove_pbc, get_lead_lcao_hamiltonian
import cPickle as pickle
from gpaw import GPAW

def dump_lead_hs(calc, filename, return_hs = False, restart = False):
        if restart:
            atoms = calc.get_atoms()
            atoms.set_calculator(calc)
            calc.initialize_positions(atoms)
        fermi = calc.get_fermi_level()
        h, s = get_lead_lcao_hamiltonian(calc,direction='z')
        h = h[0] - fermi * s
        fd = open(filename + '.pckl', 'wb')
        pickle.dump((h,s),fd,2)
        fd.close()
        if return_hs:
            return (h,s)


def dump_hs(calc, filename, return_hs=False, restart = False):
        atoms = calc.get_atoms()
        atoms.set_calculator(calc)
        if restart:
            calc.initialize_positions(atoms)
        calc.wfs.eigensolver.calculate_hamiltonian_matrix(
            calc.hamiltonian, calc.wfs, calc.wfs.kpt_u[0])
        s = calc.wfs.S_qMM[0].copy()
        h = calc.wfs.eigensolver.H_MM * Hartree
        tri2full(s)
        tri2full(h)
        remove_pbc(atoms, h, s, 2)
        fd = open(filename + '.pckl', 'wb')        
        pickle.dump((h,s), fd)
        fd.close()
        if return_hs:
            return (h,s)


class LocalizedFunctions:
    """Localized functions object.
 
    A typical transverse plane of some grid...  (pbc's only in
    transverse directions)::

    ::

        --------------------------------------------------(3)
       |    Extended region                                |
       |    .........                         .........    |
       |    .    ---.-------------------------.--(2)  .    |
       |    .   |   .                         .   |   .    |
       |    o2..|....                         o3..|....    |    
       |        |                                 |        |
       |        |     Fixed region                |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |    ....|..oo                         ....|....    |
       |    .   |   .                         .   |   .    |
       |    .  (1)--.-------------------------.---    .    |
       |    o........                         o1.......    |
       |                                                   |
      (0)--------------------------------------------------
        
       Extended region: region which is used to extend the potential in
                         order to get rid of pbc's.
       
       o1, o2, o3: corners of LocalizedFunctions objects which are
       periodic translations of LF object with corner at o.
        
       Some vectors::

       (1)-(0) = (3)-(2) = pbc_cutoff (if pbc_cutoff = 0 <=> (0)=(1) /\ (2)=(3))
        o  - (1) = v1_c
        oo - (2) = v2_c   
        
        more doc to come.......
    """

    def __init__(self, gd, f_iG, corner_c, index=None, vt_G=None):
        self.gd = gd
        self.size_c = np.array(f_iG.shape[1:4])
        self.f_iG = f_iG
        self.corner_c = corner_c
        self.index = index
        self.vt_G = vt_G
        self.restricted = False

    def periodic(self, extension_c = [0,0]):
        """extension - Extension of the surface unit cell in terms of gridpoints"""
        self.extension_c = extension_c
        v1_c = np.sign(self.corner_c[:2] - extension_c[:2])
        v2_c = np.sign(self.corner_c[:2] + self.size_c[:2]-\
                      (self.gd.end_c[:2] - extension_c[:2] - 1)) #XXX -1 ?
        # Translation vectors along the axes of the transverse unit-cell.
        trans_c = []
        for i in np.where(v1_c == -1)[0]:
            v = np.zeros(3,dtype=int)    
            v[i] = 1
            trans_c.append(v)
        for i in np.where(v2_c == 1)[0]:
            v = np.zeros(3,dtype=int)
            v[i] = -1
            trans_c.append(v)
        # Translation vectors along the diagonal of the transverse unit-cell.
        trans_diag_c = []
        for i in range(len(trans_c)):
            for j in range(i,len(trans_c)):
                v = trans_c[i]+trans_c[j]
                if not len(np.where(v == 0)[0]) >= 2:
                    trans_diag_c.append(v)
        trans_c = trans_c + trans_diag_c
        trans_c.append(np.zeros(3)) # The original LF object
        trans_c[:] *= (self.gd.n_c)# XXX change to self.gd,n_c ?
        self.periodic_list = trans_c + self.corner_c
        list = []
        for corner in self.periodic_list:
            list.append(LocalizedFunctions(self.gd,self.f_iG,
                                           corner_c=corner,
                                           index=self.index,
                                           vt_G=self.vt_G))
        return list

    def __len__(self):
        return len(self.f_iG)

    def apply_t(self):
        """Apply kinetic energy operator and return new object."""
        p = 2  # padding
        newsize_c = self.size_c + 2 * p
        gd = GridDescriptor(N_c=newsize_c + 1,
                            cell_cv=self.gd.h_c * (newsize_c + 1),
                            pbc_c=False,
                            comm=mpi.serial_comm)
        T = Laplace(gd, scale =-1/2., n=p)
        f_ig = np.zeros((len(self.f_iG),) + tuple(newsize_c))
        f_ig[:, p:-p, p:-p, p:-p] = self.f_iG
        Tf_iG = np.empty_like(f_ig)
        T.apply(f_ig, Tf_iG)
        return LocalizedFunctions(self.gd, Tf_iG, self.corner_c - p,
                                  self.index)
        
    def overlap(self, other):
        start_c = np.maximum(self.corner_c, other.corner_c)
        stop_c = np.minimum(self.corner_c + self.size_c,
                            other.corner_c + other.size_c)
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c - self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]].reshape((len(self.f_iG), -1))
            bstart_c = start_c - other.corner_c
            bstop_c = stop_c - other.corner_c
            b_iG = other.f_iG[:,
                bstart_c[0]:bstop_c[0],
                bstart_c[1]:bstop_c[1],
                bstart_c[2]:bstop_c[2]].reshape((len(other.f_iG), -1))
            a_iG1 = a_iG.copy()
            if self.vt_G is not None:
                a_iG1 *= self.vt_G[start_c[0]:stop_c[0],
                                  start_c[1]:stop_c[1],
                                  start_c[2]:stop_c[2]].reshape((-1,))
            return self.gd.dv * np.inner(a_iG1, b_iG)
        else:
            return None
    
    def restrict(self):
        """Restricts the box of the objet to the current grid"""
        start_c = np.maximum(self.corner_c, np.zeros(3))
        stop_c = np.minimum(self.corner_c + self.size_c, self.gd.N_c)
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c -self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]]
            new_corner_c = self.corner_c
            for i in np.where(self.corner_c<0):
                new_corner_c[i] = 0
            
            if (self.f_iG.shape != a_iG.shape):
                self.restricted = True
 
            self.corner_c = new_corner_c           
            self.f_iG = a_iG
            self.size_c=np.asarray(a_iG.shape[1:])

    def __or__(self, other):
        if isinstance(other, LocalizedFunctions):
            return self.overlap(other)

        # other is a potential:
        vt_G = other
        return LocalizedFunctions(self.gd, self.f_iG, self.corner_c,
                                  self.index, vt_G)

class WannierFunction(LocalizedFunctions):
    def __init__(self, gd, wanf_G, corner_c, index=None):
        LocalizedFunctions.__init__(self, gd, wanf_G[np.newaxis, :, :, :],
                                    corner_c, index)

class AtomCenteredFunctions(LocalizedFunctions):
    def __init__(self, gd, spline_j, spos_c, index=None):
        rcut = max([spline.get_cutoff() for spline in spline_j])

        # Determination of vector distance, c, from position to origin of the box
        cell = gd.cell_cv.copy()
        diagonal = (cell[0,:]+cell[1,:])
        diagonal = diagonal/np.linalg.norm(diagonal)
        a = np.zeros_like(diagonal)
        a[0]=cell[0,:][1]
        a[1]=-cell[0,:][0]
        a=-a/np.linalg.norm(a)
        c = rcut/np.dot(diagonal,a)
        # Determine corner
        A = cell.T/gd.cell_c # Basis change matrix
        pos =np.dot(np.linalg.inv(cell.T), diagonal * c)
        pos[2] = rcut/gd.cell_c[2]
        corner_c = np.ceil(spos_c * gd.N_c - pos*gd.cell_c / gd.h_c).astype(int)
        self.center = np.ceil(pos*gd.cell_c / gd.h_c).astype(int)
        size_c = np.ceil(spos_c * gd.N_c + \
                 pos*gd.cell_c / gd.h_c).astype(int) - corner_c

        smallgd = GridDescriptor(N_c=size_c + 1,
                                 cell_cv=(np.dot(A,np.diag(gd.h_c * (size_c +1))).T),
                                 pbc_c=False,
                                 comm=mpi.serial_comm)

        self.test = (np.dot(A,np.diag(gd.h_c * (size_c +1))).T)
        self.smallgd=smallgd
        lfc = LFC(smallgd, [spline_j])
        lfc.set_positions((spos_c[np.newaxis, :] * gd.N_c - corner_c + 1) /
                          smallgd.N_c)
        ni = lfc.Mmax
        f_iG = smallgd.zeros(ni)
        lfc.add(f_iG, {0: np.eye(ni)})
        LocalizedFunctions.__init__(self, gd, f_iG, corner_c, 
                                     index=index)

class UnitCell:
    def __init__(self, gd, atoms, vt_G):
        self.gd = gd
        self.atoms = atoms
        self.vt_G = vt_G    

    def attach(self, functions, functions_kin):
        self.functions = functions
        self.functions_kin = functions_kin
        p0 = {}
        p0_kin = {}        
        for f, f_kin in zip(self.functions, self.functions_kin):       
            p0[f]=f.corner_c.copy() 
            p0_kin[f_kin]=f_kin.corner_c.copy()

        self.p0 = p0
        self.p0_kin = p0_kin
        
    def set_position(self, position_c):
        self.position = position_c
        for f, f_kin in zip(self.functions, self.functions_kin):
            f.corner_c =  position_c + self.p0[f]
            f_kin.corner_c = position_c + self.p0_kin[f_kin]

  
class STM:
    def __init__(self, tip, surface, **kwargs):
        self.tip = tip
        self.srf = surface
        self.stm_calc = None
        self.ediff = None        
        self.scans ={}
        tgd = tip.gd
        sgd = surface.gd
        assert not (tgd.h_c - sgd.h_c).any()

        # The default parameters
        self.input_parameters = {'tip_atom_index': 0,
                                 'dmin': 6.0,
                                 'hs1': None,
                                 'hs10': None,
                                 'hs2': None,
                                 'hs20': None,
                                 'align_bf': 0,
                                 'cvl1': 0,
                                 'cvl2': 0,
                                 'bias': 1.0,
                                 'de': 0.01,
                                 'energies': None,
                                 'w': 0.0,
                                 'eta1': 1e-3,
                                 'eta2': 1e-3,
                                 'logfile': '-',
                                 'lead1': None,
                                 'lead2':None}

        self.initialized = False
        self.transport_uptodate = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['hs1', 'hs10', 'hs2', 'hs20',
                       'cvl1', 'cvl2', 'bias',
                       'de', 'energies', 'w',
                       'align_bf','eta1','eta2',
                       'lead1', 'lead2']:
                self.transport_uptodate = False
                break
            elif key in ['tip_atom_index', 'dmin']:
                self.initialized = False
            elif key not in self.input_parameters:
                raise KeyError, '\'%s\' not a valid keyword' % key

        self.input_parameters.update(kwargs)
        log = self.input_parameters['logfile']
        if log is None:
            class Trash:
                def write(self,s):
                    pass
                def flush(self):
                    pass
            self.log = Trash()
        elif log == '-':
            from sys import stdout
            self.log = stdout
        elif 'logfile' in kwargs:
            self.log = open(log, 'w')

    def write(self, filename):
        stmc = self.stm_calc
        fd = open(filename,'wb')
        pickle.dump({'p': self.input_parameters,
                     'egft12_emm': [stmc.energies, stmc.gft1_emm, stmc.gft2_emm]
                    },fd,2)
        fd.close()

    def restart(self, filename):
        restart = pickle.load(open(filename))
        p = restart['p']
        self.set(**p)
        print >> self.log, '#Restarting from restart file'
        print >> self.log, '#   bias = ' + str(p['bias'])
        print >> self.log, '#   de = ' + str(p['de'])
        print >> self.log, '#   w = ' + str(p['w'])
        self.initialize_transport(restart = True) 
        self.stm_calc.energies = restart['egft12_emm'][0]
        self.stm_calc.gft1_emm = restart['egft12_emm'][1]    
        self.stm_calc.gft2_emm = restart['egft12_emm'][2]    
        self.stm_calc.bias = p['bias']
        class Dummy:
            def __init__(self, bias):
                self.bias = bias
        self.stm_calc.selfenergy1 = Dummy(p['bias'] * p['w'])    
        self.stm_calc.selfenergy2 = Dummy(p['bias'] * (p['w'] - 1))    
        self.log.flush()

    def hs_from_paw(self): # If you have a lot of memory
        p=self.input_parameters
        self.set(**{'hs1':dump_hs(self.tip,'hs1',1,1),
                    'hs2':dump_hs(self.srf,'hs2',1,1),
                    'hs10':dump_lead_hs(p['lead1'],'hs10',1,1),
                    'hs20':dump_lead_hs(p['lead2'],'hs20',1,1)})

    def initialize_transport(self, restart= False, ediff_only = False):
        p = self.input_parameters        
        h1, s1 = p['hs1']
        h10, s10 = p['hs10']
        h2, s2 = p['hs2']
        h20, s20 = p['hs20']
        cvl1 = p['cvl1']
        cvl2 = p['cvl2']
        align_bf = p['align_bf']
        de = p['de']
        bias = p['bias']        
        w = p['w']
        eta1 = p['eta1']
        eta2 = p['eta2']
        
        if cvl1 == 0:
            cvl1 = 1
            
        h1 = h1[:-cvl1, :-cvl1]
        s1 = s1[:-cvl1, :-cvl1]
        h2 = h2[cvl2:, cvl2:]
        s2 = s2[cvl2:, cvl2:]

        # Align bfs with the surface lead as a reference
        diff = (h2[align_bf,align_bf] - h20[align_bf, align_bf]) \
               / s2[align_bf, align_bf]
        diff = diff.real
        h2 -= diff * s2      
        h1 -= diff * s1        
        self.ediff = diff
        diff1 = (h10[-1,-1] - h1[-1,-1])/s1[-1,-1]
        h10 -= diff1 * s10
        if not ediff_only and not self.transport_uptodate:
            from ase.transport.stm import STM as STMCalc
            T = time.localtime()
            print >> self.log, '#%d:%02d:%02d'\
                     % (T[3], T[4], T[5])+' Updating transport calculator'

            if p['energies'] == None:
                energies = np.sign(bias) * \
                np.arange(-abs(bias)*w, -abs(bias)*(w-1)+de, de)
                energies.sort()
            else:
                energies = p['energies']

            stm_calc = STMCalc(h2,  s2, 
                               h1,  s1, 
                               h20, s20, 
                               h10, s10, 
                               eta1, eta2, 
                               w=w)
            if not restart:
                stm_calc.initialize(energies, bias = bias)
            
            self.stm_calc = stm_calc
            self.transport_uptodate = True            
            self.log.flush()
            print >> self.log, '#%d:%02d:%02d'\
                     % (T[3], T[4], T[5])+' Done'

    def initialize(self):
        if self.initialized and self.transport_uptodate:
            return
        elif not self.transport_uptodate and self.initialized:
            self.initialize_transport()
            return

        T = time.localtime()
        print >> self.log, '#%d:%02d:%02d'\
                     % (T[3], T[4], T[5])+' Initializing'

        p = self.input_parameters        
        self.dmin = p['dmin']/Bohr
        tip_atom_index = p['tip_atom_index']   
        
        tgd = self.tip.gd
        sgd = self.srf.gd
        
        tip_vt_G = self.tip.hamiltonian.vt_sG[0] 
        srf_vt_G = self.srf.hamiltonian.vt_sG[0]

        # Preselect surface and tip atoms
        tip_pos_av = self.tip.atoms.get_positions() / Bohr
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        self.tip_pos_av = tip_pos_av
        tip_zmin = tip_pos_av[tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()
        self.tip_zmin = tip_zmin
        tip_zmin_a = np.empty(len(tip_pos_av))
        
        for a, setup in enumerate(self.tip.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            tip_zmin_a[a] = tip_pos_av[a, 2] - rcutmax - tip_zmin
        self.tip_zmin_a = tip_zmin_a
        srf_zmax_a = np.empty(len(srf_pos_av))
        for a, setup in enumerate(self.srf.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            srf_zmax_a[a] = srf_pos_av[a, 2] + rcutmax - srf_zmax
        
        tip_indices = np.where(tip_zmin_a < srf_zmax_a.max() - self.dmin)[0]  
        srf_indices = np.where(srf_zmax_a > tip_zmin_a.min() + self.dmin)[0]  
        print >> self.log, '# dmin =', str(self.dmin*Bohr)
        print >> self.log, '#Tip atoms:', str(tip_indices)
        sdfstr = ' '.join([ '%d'%idx for idx in srf_indices] )
        print >> self.log.write('#Surface atoms: ' + sdfstr+'\n')
        self.tip_zmin_a = tip_zmin_a

        # Construction of the unit cell for the tip.
        # p is a padding so that the tip cell also can accomodate the boxes of the
        # kinetic energies
        p=2
        zmin_index = np.where(tip_zmin_a == tip_zmin_a.min())[0][0]
        zmax_index = np.where(tip_pos_av[:,2] \
                     == tip_pos_av[tip_indices,2].max())[0][0]
        
        cell_zmin = (tip_pos_av[zmin_index,2] +\
                    tip_zmin_a[zmin_index]) 
        cell_zmax = (2 * tip_pos_av[zmax_index,2]- \
                    tip_zmin_a[zmax_index] - tip_zmin) 
        if cell_zmax > tgd.cell_c[2]-tgd.h_c[2]: 
            cell_zmax = tgd.cell_c[2]-tgd.h_c[2]        
        cell_zmin_gpts = np.floor(cell_zmin / self.tip.gd.h_c[2]-p).astype(int)
        cell_zmax_gpts = np.floor(cell_zmax / self.tip.gd.h_c[2]).astype(int)

        new_size_c = np.array([tgd.n_c[0], tgd.n_c[1], 
                               cell_zmax_gpts - cell_zmin_gpts+1])
        self.newsize_tip = new_size_c
        cell_cv =np.zeros((3,3))
        cell_cv = (new_size_c+1) * (tgd.cell_cv.T/tgd.cell_c)\
                        * self.tip.gd.h_c   

        cell_cv = cell_cv.T
        self.cell_tip = cell_cv
        
        self.newsize_ctip = new_size_c
        self.cell_cvtip = cell_cv

        tgd = GridDescriptor(N_c=new_size_c + 1,
                             cell_cv=cell_cv,
                             pbc_c=False,
                             comm=mpi.serial_comm)
        
        
        tip_vt_G = tip_vt_G[:,:,cell_zmin_gpts:cell_zmin_gpts+tgd.n_c[2]]
        tatoms = self.tip.atoms.copy()[tip_indices]
        tatoms.pbc=False

        v =tgd.cell_cv.T/tgd.cell_c*tgd.h_c*Bohr
        v[2,2] = -(cell_zmin_gpts-1)* tgd.h_c[2]*Bohr
        tatoms.translate(v[:,0]+v[:,1]+v[:,2])
        
        tatoms.set_cell(tgd.cell_cv*Bohr)
        self.tip_cell = UnitCell(gd = tgd, 
                              atoms = tatoms, 
                              vt_G = tip_vt_G )
        
        #extension of the surface unit cell


        extension_c = np.ceil(0.5*tgd.cell_c/self.srf.gd.h_c).astype(int)
        extension_c[2] = 0
        self.extension_c = extension_c
       
        newsize_c = 2 * self.extension_c + sgd.n_c
        self.newsize_srf =newsize_c
        extd_vt_G = np.zeros(newsize_c)
        extd_vt_G[extension_c[0]:-extension_c[0],\
                  extension_c[1]:-extension_c[1],:] = srf_vt_G       
        extd_vt_G[:extension_c[0],extension_c[1]:-extension_c[1]]\
                 =srf_vt_G[-extension_c[0]:]
        extd_vt_G[-extension_c[0]:,extension_c[1]:-extension_c[1]]\
                 = srf_vt_G[:extension_c[0]]
        extd_vt_G[:,:extension_c[1]]\
                 = extd_vt_G[:,-2*extension_c[1]:-extension_c[1]]        
        extd_vt_G[:,-extension_c[1]:]\
                 = extd_vt_G[:,extension_c[1]:2*extension_c[1]]        

        self.extd_vt_G = extd_vt_G
        cell_cv = (newsize_c+1) * (self.srf.gd.cell_cv.T/self.srf.gd.cell_c)\
                        * self.srf.gd.h_c   
        cell_cv = cell_cv.T
        self.cell_srf = cell_cv


        extd_gd = GridDescriptor(N_c=newsize_c + 1,
                                    cell_cv=cell_cv,
                                    pbc_c=False,
                                    comm=mpi.serial_comm)   
        self.extd_gd = extd_gd  
        
        # functions
        self.tip_functions = []
        i=0
        for a in tip_indices:
            setup = self.tip.wfs.setups[a]
            spos_c = self.tip_cell.atoms.get_scaled_positions()[a]
            #spos_c = tip_pos_av[a] / self.tip.gd.cell_c
            if a == tip_atom_index:
                self.tip_atom_box_0 = i
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.tip_cell.gd, [phit], spos_c, i)
                self.tip_functions.append(f)
                i += len(f.f_iG)
        self.ni = i
           
        # Apply kinetic energy:
        self.tip_functions_kin = []
        for f in self.tip_functions:
            self.tip_functions_kin.append(f.apply_t())
        
        for f, f_kin in zip(self.tip_functions, self.tip_functions_kin):
            f.restrict()
            f_kin.restrict()
            if f.restricted:
                print >> self.log, '#Restriction of bf with indices ', \
                                    range(f.index, f.index+f.f_iG.shape[0])

        self.tip_cell.attach(self.tip_functions,self.tip_functions_kin)
        self.srf_functions = []
        j = 0
        for a in srf_indices:
            setup = self.srf.wfs.setups[a]
            spos_c = self.srf.atoms.get_scaled_positions()[a]
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.srf.gd, [phit], spos_c, j)
                self.srf_functions += f.periodic(extension_c)
                j += len(f.f_iG)
        self.nj = j
        self.spos_c11 = spos_c
        # Apply kinetic energy:
        self.srf_functions_kin = []
        for f in self.srf_functions:
            self.srf_functions_kin.append(f.apply_t())
        # Set positions of the surface orbitals in the extended grid
        for f, f_kin in zip(self.srf_functions, self.srf_functions_kin):
            f.corner_c += extension_c
            f_kin.corner_c += extension_c
                       

        srf_atoms = self.srf.atoms.copy()
        srf_atoms.set_cell(self.extd_gd.cell_cv*Bohr)
        
        v =(self.extension_c+1)*self.srf.gd.h_c*Bohr        
        v = np.dot(self.extd_gd.cell_cv.T/self.extd_gd.cell_c,v)        
        self.v =v
        srf_atoms.translate(v)
        self.srf_cell = UnitCell(self.extd_gd, 
                                   srf_atoms, 
                                   self.extd_vt_G)
    
        if not self.transport_uptodate:
            self.initialize_transport()            
 
        self.initialized = True
        self.log.flush()


    def set_tip_position(self, position):   
        """Positions origin of the tip unit cell at grdpt_c
            in the extended transverse surface unit cell."""         
        p = self.input_parameters
        tip_atom_index = p['tip_atom_index']
        h_c = self.srf.gd.h_c        
        
        tip_pos_av = self.tip_cell.atoms.get_positions() / Bohr
        tip_pos_av_grpt = self.tip_cell.gd.N_c\
                          * self.tip_cell.atoms.get_scaled_positions()
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        tip_zmin = tip_pos_av[tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()
        
        #corner of the tip unit cell in the extended grid        
        cell_corner_c = position + self.extension_c - tip_pos_av_grpt[tip_atom_index]
        cell_corner_c[2]  = (srf_zmax + self.dmin - tip_zmin) / h_c[2]
        cell_corner_c = np.round(cell_corner_c).astype(int)        

        self.tip_position = cell_corner_c + \
                            tip_pos_av_grpt[tip_atom_index] - self.extension_c        
        self.dmin = self.tip_position[2] * h_c[2] - srf_zmax
        self.tip_cell.set_position(cell_corner_c)        

        # Add the tip potential at the respective place in the extended grid
        # of the surface
        # XXX +1 due tue pbc = False
        size_c = self.tip_cell.gd.n_c
        current_Vt = self.extd_vt_G.copy()
        current_Vt[cell_corner_c[0]+1:cell_corner_c[0]+1 + size_c[0],
                   cell_corner_c[1]+1:cell_corner_c[1]+1 + size_c[1],
                   cell_corner_c[2]+1:cell_corner_c[2]+1 + size_c[2]] +=\
                    self.tip_cell.vt_G
        self.current_v=current_Vt
        self.tip_cell_corner_c = cell_corner_c

    def get_V(self, position_c):
        """Returns the overlap hamiltonian for a position of the tip_atom """
        if not self.initialized:
            self.initialize()

        self.set_tip_position(position_c)
        nj = self.nj
        ni = self.ni
        V_ij = np.zeros((nj,ni))
        S_ij = np.zeros((nj,ni))
        vt_G = self.current_v
        for s in self.srf_functions:
            j1 = s.index
            j2 = j1 + len(s)
            for t, t_kin in zip(self.tip_functions, self.tip_functions_kin):
                i1 = t.index
                i2 = i1 + len(t)
                V = (s | vt_G | t)
                S = (s | t )
                if V is None:
                    V = 0
                kin = (s | t_kin)
                if kin is None:
                    kin = 0
                V_ij[j1:j2, i1:i2] += V + kin
                if S is not None:
                    S_ij[j1:j2,i1:i2] += S
        return V_ij * Hartree - S_ij * self.ediff.real

    def get_transmission(self, position_c):
        V_ts = self.get_V(position_c)       
        T_stm = self.stm_calc.get_transmission(V_ts)
        return T_stm

    def get_current(self, position_c, bias=None):
        if bias == None:
            bias = self.stm_calc.bias
        position_c = tuple(position_c)+(0,)
        V_ts = self.get_V(position_c)
        Is = self.stm_calc.get_current(bias,V_ts)    
        return Is*77466.1509   #units: nA
    
    def get_s(self, position_c):
        self.set_tip_position(position_c)
        S_ij = np.zeros((self.nj, self.ni))
        for s in self.srf_functions:
            j1 = s.index
            j2 = j1 + len(s)
            for t in self.tip_functions:
                i1 = t.index
                i2 = i1 + len(t)
                overlap = (s | t) 
                if overlap is not None:
                    S_ij[j1:j2, i1:i2] += overlap
  
        return S_ij

    def reset(self):
        self.scans = {}

    def scan(self):
        self.set_tip_position([0,0,0])
        n_c = self.srf.gd.n_c
        scan = np.zeros(tuple(n_c[:2]))
        for x in range(n_c[0]):
             for y in range(n_c[1]):
                 I  = self.get_current([x,y])                       
                 scan[x,y] = I
                 self.log.flush()    
        self.scans['fullscan'] = scan

    def linescan(self, startstop=None):
        if startstop == None:
            start = np.array([0,0])
            stop = self.srf.gd.n_c[:2]-1
        else:
            start = np.asarray(startstop[0])
            stop = np.asarray(startstop[1])
        assert ((self.srf.gd.n_c[:2]-stop)>=0).all()
        v = (stop-start)/np.linalg.norm(stop-start)
        self.v = v
        n_c = self.srf.gd.n_c[:2]
        h_c = self.srf.gd.h_c[:2]
        h = np.linalg.norm(v*h_c)
        n = np.floor(np.linalg.norm((stop-start)*h_c)/h).astype(int) + 1
        linescan_n = np.zeros((n,))
        line = np.arange(n)*Bohr*h
        for i in range(n):
            grpt = start + v*i
            if np.round((grpt%1),5).any(): # Interpolate if nessesary
                C = np.empty((2,2,2)) # find four nearest neighbours
                C[0,0] = np.floor(grpt)
                C[1,0] = C[0,0] + np.array([1,0])
                C[0,1] = C[0,0] + np.array([0,1])
                C[1,1] = C[0,0] + np.array([1,1])
                xd = (grpt%1)[0]
                yd = (grpt%1)[1]
                if not self.scans.has_key('fullscan'):
                    I1 = self.get_current(C[0,0])*(1-xd) \
                        + self.get_current(C[1,0])*xd
                    I2 = self.get_current(C[0,1])*(1-xd) \
                        + self.get_current(C[1,1])*xd
                    I = I1 * (1-yd) + I2*yd
                else:
                    fullscan = self.scans['fullscan']
                    I1 = fullscan[tuple(C[0,0])]*(1-xd) \
                       + fullscan[tuple(C[1,0])]*xd
                    I2 = fullscan[tuple(C[0,1])]*(1-xd) \
                       + fullscan[tuple(C[1,1])]*xd
                    I = I1 * (1-yd) + I2*yd

            else:
                if not self.scans.has_key('fullscan'):
                    I = self.get_current(grpt.astype(int))
                else:                
                    I = self.scans['fullscan'][tuple(grpt.astype(int))]
            linescan_n[i] = I
        self.scans['linescan']=([start,stop],line,linescan_n) 

    def plot(self, vmin = None):
        import matplotlib
        import pylab
        from pylab import ogrid, imshow, cm, colorbar
        
        gd = self.srf.gd
        
        if self.scans.has_key('fullscan'):
            scan0_iG = self.scans['fullscan']
            shape = scan0_iG.shape
            scan_iG = np.zeros(tuple(np.asarray(shape)+1))
            scan_iG[:shape[0],:shape[1]] = scan0_iG
            scan_iG[-1,:shape[1]] = scan0_iG[0,:]
            scan_iG[:,-1] = scan_iG[:,0]

            h = 0.1          
            N_c = np.floor((gd.cell_cv[0,:2]+gd.cell_cv[1,:2])/h).astype(int)
            ortho_cell_c = np.array(N_c*gd.h_c[:2])
            plot = np.zeros(tuple(N_c))
            # Basis change matrix
            # e -> usual basis {(1,0),(0,1)}
            # o -> basis descrining original original cell
            # n -> basis describing the new cell

            eMo = (gd.cell_cv.T/gd.cell_c * gd.h_c)[:2,:2]
            eMn = np.eye(2) * h           
            oMn = np.dot(np.linalg.inv(eMo), eMn)
            
            for i in range(N_c[0]):
                for j in range(N_c[1]):
                    grpt = np.dot(oMn, [i,j])
                    if (grpt<0).any() or (np.ceil(grpt)>gd.n_c[:2]).any():
                        plot[i,j] = plot.max()+1000
                    else: # interpolate
                        C00 = np.floor(grpt).astype(int)
                        C01 = C00.copy()
                        C01[0]+=1
                        C10 = C00.copy()
                        C10[1]+=1
                        C11 = C10.copy()
                        C11[0] += 1
                        
                        x0 = grpt[0] - C00[0]
                        y0 = grpt[1] - C00[1]
                        P0 = scan_iG[tuple(C00)]*(1-x0)\
                           + scan_iG[tuple(C01)] * x0
                        P1 = scan_iG[tuple(C10)]*(1-x0)\
                           + scan_iG[tuple(C11)] * x0
                        plot[i,j] = P0 * (1-y0) + P1*y0

            plot = plot.T
            if vmin==None:
                vmin = scan0_iG.min()
            norm = matplotlib.colors.\
                   normalize(vmin=vmin,vmax=scan0_iG.max())
            
            f0 = pylab.figure()
            p0 = f0.add_subplot(111)
            x,y = ogrid[0:plot.shape[0]:1,0:plot.shape[1]:1]
            extent=[0,plot.shape[1]*h*Bohr,
                    0,plot.shape[0]*h*Bohr]        
            
            p0.set_ylabel('\xc5')
            p0.set_xlabel('\xc5')
            imshow(plot,
                   norm = norm,
                   interpolation = 'bicubic',
                   origin='lower',
                   cmap = cm.hot,
                   extent=extent)
            cb = colorbar()            
            cb.set_label('I[nA]')
                       
        if self.scans.has_key('linescan'):
            startstop, line, linescan_n = self.scans['linescan']
            start = startstop[0]
            stop = startstop[1]
            f1 = pylab.figure()
            p1 = f1.add_subplot(111)
            p1.plot(line, linescan_n)
            eMo = (gd.cell_cv.T/gd.cell_c)[:2,:2]
            start = np.dot(eMo,start * gd.h_c[:2]) * Bohr
            stop = np.dot(eMo, stop * gd.h_c[:2]) * Bohr
             
            if self.scans.has_key('fullscan'): #Add a line
                p0.plot([start[0], stop[0]], [start[1], stop[1]],'-b')
                self.extent=extent   
                p0.set_xlim(tuple(extent[:2]))
                p0.set_ylim(tuple(extent[-2:]))
        pylab.show()
