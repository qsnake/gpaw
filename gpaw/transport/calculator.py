import pickle

from gpaw.transport.selfenergy import LeadSelfEnergy, CellSelfEnergy
from gpaw.transport.greenfunction import GreenFunction
from ase.transport.tools import function_integral, fermidistribution
from ase import Atoms, Atom, monkhorst_pack, Hartree, Bohr
import ase
import numpy as np

import gpaw
from gpaw import GPAW, extra_parameters
from gpaw import Mixer, MixerDif, PoissonSolver
from gpaw import restart as restart_gpaw
from gpaw.transport.tools import k2r_hs, r2k_hs, tri2full, dot, Se_Sparse_Matrix
from gpaw.transport.intctrl import IntCtrl
from gpaw.transport.surrounding import Surrounding
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lcao.tools import get_realspace_hs
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import gemm
from gpaw.utilities.timing import Timer
from gpaw.wavefunctions import LCAOWaveFunctions
from gpaw import debug

class PathInfo:
    def __init__(self, type, nlead):
        self.type = type
        self.num = 0
        self.lead_num = nlead
        self.energy = []
        self.weight = []
        self.nres = 0
        self.sigma = []
        for i in range(nlead):
            self.sigma.append([])
        if type == 'eq':
            self.fermi_factor = []
        elif type == 'ne':
            self.fermi_factor = []
            for i in range(nlead):
                self.fermi_factor.append([[], []])
        else:
            raise TypeError('unkown PathInfo type')

    def add(self, elist, wlist, flist, siglist):
        self.num += len(elist)
        self.energy += elist
        self.weight += wlist
        if self.type == 'eq':
            self.fermi_factor += flist
        elif self.type == 'ne':
            for i in range(self.lead_num):
                for j in [0, 1]:
                    self.fermi_factor[i][j] += flist[i][j]
        else:
            raise TypeError('unkown PathInfo type')
        for i in range(self.lead_num):
            self.sigma[i] += siglist[i]

    def set_nres(self, nres):
        self.nres = nres
    
class Transport(GPAW):
    
    def __init__(self, **transport_kwargs):
        self.set_transport_kwargs(**transport_kwargs)
        if self.scat_restart:
            GPAW.__init__(self, self.restart_file + '.gpw')
            self.set_positions()
            self.verbose = self.transport_parameters['verbose']
        else:
            GPAW.__init__(self, **self.gpw_kwargs)            
            
    def set_transport_kwargs(self, **transport_kwargs):
        kw = transport_kwargs  
        p =  self.set_default_transport_parameters()
        self.gpw_kwargs = kw.copy()
        for key in kw:
            if key in ['use_lead', 'identical_leads',
                       'pl_atoms', 'pl_cells', 'pl_kpts',
                       'use_buffer', 'buffer_atoms', 'edge_atoms', 'bias',
                       'lead_restart',
                       
                       'use_env', 'env_atoms', 'env_cells', 'env_kpts',
                       'env_use_buffer', 'env_buffer_atoms', 'env_edge_atoms',
                       'env_bias', 'env_pbc', 'env_restart',                     
                       
                       'LR_leads', 'gate',  'cal_loc', 'align',                       
                       'recal_path', 'use_qzk_boundary', 'use_linear_vt_mm',
                       'use_linear_vt_array',
                       'scat_restart', 'save_file', 'restart_file', 'fixed_boundary']:
                
                del self.gpw_kwargs[key]
            #----descript the lead-----    
            if key in ['use_lead']:
                p['use_lead'] = kw['use_lead']
            if key in ['identical_leads']:
                p['identical_leads'] = kw['identical_leads']
            if key in ['pl_atoms']:
                p['pl_atoms'] = kw['pl_atoms']
            if key in ['pl_cells']:
                p['pl_cells'] = kw['pl_cells']
            if key in ['pl_kpts']:
                p['pl_kpts'] = kw['pl_kpts']
            if key in ['use_buffer']:
                p['use_buffer'] = kw['use_buffer']
            if key in ['buffer_atoms']:
                p['buffer_atoms'] = kw['buffer_atoms']
            if key in ['edge_atoms']:
                p['edge_atoms'] = kw['edge_atoms']
            if key in ['bias']:
                p['bias'] = kw['bias']                
            if key in ['lead_restart']:
                p['lead_restart'] = kw['lead_restart']
            #----descript the environment----   
            if key in ['use_env']:
                p['use_env'] = kw['use_env']
            if key in ['env_atoms']:
                p['env_atoms'] = kw['env_atoms']
            if key in ['env_cells']:
                p['env_cells'] = kw['env_cells']
            if key in ['env_kpts']:
                p['env_kpts'] = kw['env_kpts']
            if key in ['env_buffer_atoms']:
                p['env_buffer_atoms'] = kw ['env_buffer_atoms']
            if key in ['env_edge_atoms']:
                p['env_edge_atoms'] = kw['env_edge_atoms']
            if key in ['env_pbc']:
                p['env_pbc'] = kw['env_pbc']
            if key in ['env_bias']:
                p['env_bias'] = kw['env_bias']
            if key in ['env_restart']:
                p['env_restart'] = kw['env_restart']
            #----descript the scattering region----     
            if key in ['LR_leads']:         
                p['LR_leads'] = kw['LR_leads']
            if key in ['gate']:
                p['gate'] = kw['gate']
            if key in ['cal_loc']:
                p['cal_loc'] = kw['cal_loc']
            if key in ['recal_path']:
                p['recal_path'] = kw['recal_path']
            if key in ['align']:
                p['align'] = kw['align']
            if key in ['use_qzk_boundary']:
                p['use_qzk_boundary'] = kw['use_qzk_boundary']
            if key in ['use_linear_vt_mm']:
                p['use_linear_vt_mm'] = kw['use_linear_vt_mm']
            if key in ['use_linear_vt_array']:
                p['use_linear_vt_array'] = kw['use_linear_vt_array']                
            if key in ['scat_restart']:
                p['scat_restart'] = kw['scat_restart']
            if key in ['save_file']:
                p['save_file'] = kw['save_file']
            if key in ['restart_file']:
                p['restart_file'] = kw['restart_file']
            if key in ['fixed_boundary']:
                p['fixed_boundary'] = kw['fixed_boundary']
            if key in ['spinpol']:
                p['spinpol'] = kw['spinpol']
            if key in ['verbose']:
                p['verbose'] = kw['verbose']

        self.transport_parameters = p

        self.use_lead = p['use_lead']
        self.identical_leads = p['identical_leads']
        self.pl_atoms = p['pl_atoms']
        self.lead_num = len(self.pl_atoms)
        self.bias = p['bias']

        if self.use_lead:
            self.pl_cells = p['pl_cells']
            self.pl_kpts = p['pl_kpts']
            self.lead_restart = p['lead_restart']
            self.use_buffer = p['use_buffer']
            self.buffer_atoms = p['buffer_atoms']
            self.edge_atoms = p['edge_atoms']
            assert self.lead_num == len(self.pl_cells)
            #assert self.lead_num == len(self.buffer_atoms)
            #assert self.lead_num == len(self.edge_atoms[0])
            assert self.lead_num == len(self.bias)
            
        self.use_env = p['use_env']
        self.env_atoms = p['env_atoms']
        self.env_num = len(self.env_atoms)
        self.env_bias = p['env_bias']

        if self.use_env:
            self.env_cells = p['env_cells']
            self.env_kpts = p['env_kpts']
            self.env_buffer_atoms = p['env_buffer_atoms']
            self.env_edge_atoms = p['env_edge_atoms']
            self.env_pbc = p['env_pbc']
            self.env_restart = p['env_restart']
            assert self.env_num == len(self.env_cells)
            assert self.env_num == len(self.env_buffer_atoms)
            assert self.env_num == len(self.env_edge_atoms[0])
            assert self.env_num == len(self.env_bias)

        self.LR_leads = p['LR_leads']            
        self.gate = p['gate']
        self.cal_loc = p['cal_loc']
        self.recal_path = p['recal_path']
        self.use_qzk_boundary = p['use_qzk_boundary']
        self.align =  p['align']
        self.use_linear_vt_mm = p['use_linear_vt_mm']
        self.use_linear_vt_array = p['use_linear_vt_array']        
        self.scat_restart = p['scat_restart']
        self.save_file = p['save_file']
        self.restart_file = p['restart_file']
        self.fixed = p['fixed_boundary']
        self.spinpol = p['spinpol']
        self.verbose = p['verbose']
        self.d = p['d']
       
        if self.scat_restart and self.restart_file == None:
            self.restart_file = 'scat'
        
        self.master = (world.rank==0)
    
        bias = self.bias + self.env_bias
        self.cal_loc = self.cal_loc and max(abs(bias)) != 0
 
        if self.use_linear_vt_mm:
            self.use_buffer = False
        
        if self.LR_leads and self.lead_num != 2:
            raise RuntimeErrir('wrong way to use keyword LR_leads')
       
        self.initialized_transport = False
       
        self.atoms_l = [None] * self.lead_num
        self.atoms_e = [None] * self.env_num
        
        kpts = kw['kpts']
        if np.product(kpts) == kpts[self.d]:
            self.gpw_kwargs['usesymm'] = None
        else:
            self.gpw_kwargs['usesymm'] = False
   
    def construct_grid_descriptor(self, N_c, cell_cv,
                                  pbc_c, domain_comm, parsize):
        GPAW.construct_grid_descriptor(self, N_c, cell_cv,
                                  pbc_c, domain_comm, parsize)
        self.gd.use_fixed_bc = True

    def set_default_transport_parameters(self):
        p = {}
        p['use_lead'] = True
        p['identical_leads'] = True
        p['pl_atoms'] = []
        p['pl_cells'] = []
        p['pl_kpts'] = []
        p['use_buffer'] = False
        p['buffer_atoms'] = None
        p['edge_atoms'] = None
        p['bias'] = [0, 0]
        p['d'] = 2
        p['lead_restart'] = False

        p['use_env'] = False
        p['env_atoms'] = []
        p['env_cells']  = []
        p['env_kpts'] = []
        p['env_use_buffer'] = False
        p['env_buffer_atoms'] = None
        p['env_edge_atoms'] = None
        p['env_pbc'] = True
        p['env_bias'] = []
        p['env_restart'] = False
        
        p['LR_leads'] = True
        p['gate'] = 0
        p['cal_loc'] = False
        p['recal_path'] = False
        p['use_qzk_boundary'] = False
        p['align'] = False
        p['use_linear_vt_mm'] = False
        p['use_linear_vt_array'] = False        
        p['scat_restart'] = False
        p['save_file'] = True
        p['restart_file'] = None
        p['fixed_boundary'] = True
        p['spinpol'] = False
        p['verbose'] = False
        return p     

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        
    def initialize_transport(self, dryrun=False, restart=True):
        if not self.initialized:
            self.initialize()
        self.nspins = self.wfs.nspins

        if self.LR_leads:
            self.ntkmol = self.gpw_kwargs['kpts'][self.d]
            self.ntklead = self.pl_kpts[self.d]
            if self.ntkmol == len(self.wfs.bzk_kc):
                self.npk = 1
                self.kpts = self.wfs.bzk_kc
            else:
                self.npk = len(self.wfs.ibzk_kc) / self.ntkmol
                self.kpts = self.wfs.ibzk_kc
        else:
            self.npk = 1
            self.kpts = self.wfs.ibzk_kc
            self.ntkmol = len(self.kpts)
            if self.use_lead:
                self.ntklead = np.product(self.pl_kpts)

        bzk_kc = self.wfs.bzk_kc 
        self.gamma = len(bzk_kc) == 1 and not bzk_kc[0].any()
        self.nbmol = self.wfs.setups.nao

        if self.use_lead:
            if self.LR_leads:
                self.dimt_lead = []
                self.dimt_buffer = []
            self.nblead = []
            self.edge_index = [[None] * self.lead_num, [None] * self.lead_num]

        if self.use_env:
            self.nbenv = []
            self.env_edge_index = [[None] * self.env_num, [None] * self.env_num]
         
        if dryrun:
            self.atoms_l = []
            self.atoms_e = []
        for i in range(self.lead_num):
            if dryrun:
                self.atoms_l.append([])
            self.atoms_l[i] = self.get_lead_atoms(i)
            calc = self.atoms_l[i].calc
            atoms = self.atoms_l[i]
            if not calc.initialized:
                calc.initialize(atoms)
                if not dryrun:
                    calc.set_positions(atoms)
            self.nblead.append(calc.wfs.setups.nao)
            if self.LR_leads:
                self.dimt_lead.append(calc.gd.N_c[self.d])

        for i in range(self.env_num):
            self.atoms_e[i] = self.get_env_atoms(i)
            calc = self.atoms_e[i].calc
            atoms = self.atoms_e[i]
            if not calc.initialized:
                calc.initialize(atoms)
                if not dryrun:
                    calc.set_positions(atoms)
            self.nbenv.append(calc.wfs.setups.nao)

        if self.use_lead:
            if self.LR_leads and self.npk == 1:    
                self.lead_kpts = self.atoms_l[0].calc.wfs.bzk_kc
            else:
                self.lead_kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        if self.use_env:
            self.env_kpts = self.atoms_e[0].calc.wfs.ibzk_kc               
        self.allocate_cpus()
        self.initialize_matrix()
       
        if self.use_lead:
            self.get_lead_index()
            self.get_buffer_index()
        if self.use_env:
            self.get_env_index()
            self.get_env_buffer_index()

        if self.use_lead:
            if self.nbmol <= np.sum(self.nblead):
                self.use_buffer = False
                if self.master:
                    self.text('Moleucle is too small, force not to use buffer')
            
        if self.use_lead:
            if self.use_buffer: 
                self.buffer = [len(self.buffer_index[i])
                                                   for i in range(self.lead_num)]
                self.print_index = self.buffer_index
            else:
                self.buffer = [0] * self.lead_num
                self.print_index = self.lead_index
            
        if self.use_env:
            self.env_buffer = [len(self.env_buffer_index[i])
                                               for i in range(self.env_num)]
        self.set_buffer()
            
        self.current = 0
        self.linear_mm = None

        if not dryrun:        
            for i in range(self.lead_num):
                if self.identical_leads and i > 0:
                    self.update_lead_hamiltonian(i, 'lead0')    
                else:
                    self.update_lead_hamiltonian(i)
                self.initialize_lead(i)

            for i in range(self.env_num):
                self.update_env_hamiltonian(i)
                self.initialize_env(i)
        elif restart:
            for l in range(self.lead_num):
                (self.lead_fermi[l],
                 self.hl_skmm[l],
                 self.dl_skmm[l],
                 self.sl_kmm[l]) = self.pl_read('lead' + str(l)+ '.mat')
                self.initialize_lead(l)            

        self.fermi = self.lead_fermi[0]

        world.barrier()
        if self.use_lead:
            self.check_edge()
            self.get_edge_density()
        
        if self.fixed:
            self.atoms.calc = self
            self.surround = Surrounding(type='LR',
                                        atoms=self.atoms,
                                        atoms_l=self.atoms_l,
                                        directions=['z-','z+'],
                                        lead_index=self.lead_index,
                                        bias=self.bias)
            self.surround.initialize()
        if not dryrun:
            if not self.fixed:
                self.set_positions()
            else:
                self.surround.set_positions()
                self.get_hamiltonian_initial_guess()
        #del self.atoms_l
        del self.atoms_e
        self.initialized_transport = True

    def get_hamiltonian_initial_guess(self):
        atoms = self.atoms.copy()
        atoms.pbc[self.d] = True
        kwargs = self.gpw_kwargs.copy()
        kwargs['poissonsolver'] = PoissonSolver(nn=2)
        #kwargs['kpts'] = (1,1,1)
        kwargs['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        atoms.set_calculator(gpaw.GPAW(**kwargs))
        atoms.get_potential_energy()
        h_skmm, s_kmm =  self.get_hs(atoms.calc, 'lead')
        ntk = 1
        kpts = atoms.calc.wfs.ibzk_qc
        self.h_skmm = self.substract_pk(ntk, kpts, h_skmm, 'h')
        self.s_kmm = self.substract_pk(ntk, kpts, s_kmm)
        fd = file('guess.dat', 'wb')
        pickle.dump((self.h_skmm, self.s_kmm), fd, 2)
        fd.close()
        del atoms

    def get_hamiltonian_initial_guess2(self):
        self.h_skmm = self.hl_spkmm[0].copy()
        self.s_kmm = self.sl_pkmm[0].copy()

    def get_lead_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        lead_basis_list = []
        for i in range(self.lead_num):
            calc = self.atoms_l[i].calc
            lead_basis_list.append([setup.niAO for setup in calc.wfs.setups])
            for j, lj  in zip(self.pl_atoms[i], range(len(self.pl_atoms[i]))):
                begin = np.sum(np.array(basis_list[:j], int))
                l_begin = np.sum(np.array(lead_basis_list[i][:lj], int))
                if self.edge_atoms == None:
                    self.edge_index[0][i] = -i * lead_basis_list[i][-1]
                    self.edge_index[1][i] = -i * lead_basis_list[i][-1]
                    assert self.lead_num == 2
                else:
                    if j == self.edge_atoms[1][i]:
                        self.edge_index[1][i] = begin
                    if lj == self.edge_atoms[0][i]:
                        self.edge_index[0][i] = l_begin
                for n in range(basis_list[j]):
                    self.lead_index[i].append(begin + n) 
            self.lead_index[i] = np.array(self.lead_index[i], int)
            
    def get_env_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        env_basis_list = []
        for i in range(self.env_num):
            calc = self.atoms_e[i].calc
            env_basis_list.append([setup.niAO for setup in calc.wfs.setups])
            for j, lj in zip(self.env_atoms[i], range(len(self.env_atoms[i]))):
                begin = np.sum(np.array(basis_list[:j], int))
                l_begin = np.sum(np.array(env_basis_list[i][:lj], int))
                if j == self.env_edge_atoms[1][i]:
                    self.env_edge_index[1][i] = begin
                if lj == self.env_edge_atoms[0][i]:
                    self.env_edge_index[0][i] = l_begin
                for n in range(basis_list[j]):
                    self.env_index[i].append(begin + n)
            self.env_index[i] = np.array(self.env_index[i], int)
            
    def get_buffer_index(self):
        if not self.use_buffer:
            self.dimt_buffer = [0] * self.lead_num
        elif self.LR_leads:
            for i in range(self.lead_num):
                if i == 0:
                    self.buffer_index[i] = self.lead_index[i] - self.nblead[i]
                if i == 1:
                    self.buffer_index[i] = self.lead_index[i] + self.nblead[i]
            self.dimt_buffer = self.dimt_lead
        else:
            basis_list = [setup.niAO for setup in self.wfs.setups]
            for i in range(self.lead_num):
                for j in self.buffer_atoms[i]:
                    begin = np.sum(np.array(basis_list[:j], int))
                    for n in range(basis_list[j]):
                        self.buffer_index[i].append(begin + n) 
                self.buffer_index[i] = np.array(self.buffer_index[i], int)
    
    def get_env_buffer_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        for i in range(self.env_num):
            for j in self.env_buffer_atoms[i]:
                begin = np.sum(np.array(basis_list[:j], int))
                for n in range(basis_list[j]):
                    self.env_buffer_index[i].append(begin + n)
            self.env_buffer_index[i] = np.array(self.env_buffer_index[i], int)
           
    def initialize_matrix(self):
        if self.use_lead:
            self.hl_skmm = []
            self.dl_skmm = []
            self.sl_kmm = []
            self.hl_spkmm = []
            self.dl_spkmm = []
            self.sl_pkmm = []
            self.hl_spkcmm = []
            self.dl_spkcmm = []
            self.sl_pkcmm = []
            self.ed_pkmm = []
            self.lead_index = []
            self.inner_lead_index = []
            self.buffer_index = []
            self.lead_fermi = np.empty([self.lead_num])

        if self.use_env:
            self.he_skmm = []
            self.de_skmm = []
            self.se_kmm = []
            self.he_smm = []
            self.de_smm = []
            self.se_mm = []
            self.env_index = []
            self.inner_env_index = []
            self.env_buffer_index = []
            self.env_ibzk_kc = []
            self.env_weight = []
            self.env_fermi = np.empty([self.lead_num])

        npk = self.my_npk
        if self.npk == 1 and self.wfs.kpt_comm.size == 1:
            dtype = float
        else:
            dtype = complex
        ns = self.my_nspins        
        for i in range(self.lead_num):
            nk = len(self.my_lead_kpts)
            nb = self.nblead[i]
            self.hl_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.dl_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.sl_kmm.append(np.empty((nk, nb, nb), complex))

            self.hl_spkmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.dl_spkmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.sl_pkmm.append(np.empty((npk, nb, nb), dtype))

            self.hl_spkcmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.dl_spkcmm.append(np.empty((ns, npk, nb, nb), dtype))
            self.sl_pkcmm.append(np.empty((npk, nb, nb), dtype))

            self.ed_pkmm.append(np.empty((ns, npk, nb, nb), dtype))

            self.lead_index.append([])
            self.inner_lead_index.append([])
            self.buffer_index.append([])
       
        for i in range(self.env_num):
            calc = self.atoms_e[i].calc
            nk = len(calc.wfs.ibzk_kc)
            nb = self.nbenv[i]
            self.he_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.de_skmm.append(np.empty((ns, nk, nb, nb), complex))
            self.se_kmm.append(np.empty((nk, nb, nb), complex))
            self.he_smm.append(np.empty((ns, nb, nb)))
            self.de_smm.append(np.empty((ns, nb, nb)))
            self.se_mm.append(np.empty((nb, nb)))
            self.env_index.append([])
            self.inner_env_index.append([])
            self.env_buffer_index.append([])
            self.env_ibzk_kc.append([])
            self.env_weight.append([])
                
        if self.use_lead:
            self.ec = np.zeros([self.lead_num, ns])        

        if self.gamma:
            dtype = float
        else:
            dtype = complex
 
        nk = len(self.my_kpts)
        nb = self.nbmol
        self.h_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.d_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.s_kmm = np.empty((nk, nb, nb), dtype)
        if self.npk == 1 and self.wfs.kpt_comm.size == 1:
            dtype = float
        else:
            dtype = complex        
        
        self.h_spkmm = np.empty((ns, npk, nb, nb), dtype)
        self.d_spkmm = np.empty((ns, npk, nb, nb), dtype)
        self.s_pkmm = np.empty((npk, nb, nb), dtype)

        self.h_spkcmm = np.empty((ns, npk, nb, nb), dtype)
        self.d_spkcmm = np.empty((ns, npk, nb, nb), dtype)
        self.s_pkcmm = np.empty((npk, nb, nb), dtype)
        
    def allocate_cpus(self):
        # when do parallel calculation, if world.size > nspins * npk,
        # use energy parallel, otherwise just parallel for s,k pairs, and
        # s have first priority
        rank = world.rank
        size = world.size
        npk = self.npk
        ns = self.nspins
        nspk = ns * npk
        if size > nspk:
            parsize_energy = size // (nspk)
            nspins_each = 1
            npk_each = 1
            assert size % nspk == 0
        else:
            parsize_energy = 1
            if size == 1:
                nspins_each = ns
                npk_each = npk
            else:
                npk_each = npk // (size // ns)
                nspins_each = 1
            assert nspk % size == 0
        r0 = (rank // parsize_energy) * parsize_energy
        ranks = np.arange(r0, r0 + parsize_energy)
        self.energy_comm = world.new_communicator(ranks)

        pkpt_size = npk // npk_each
        pke_size = pkpt_size * self.energy_comm.size
        spin_size = size // pke_size
        spin_rank = rank // pke_size

        r0 = rank % self.energy_comm.size + spin_rank * pke_size
        ranks = np.arange(r0, r0 + pke_size , self.energy_comm.size)
        self.pkpt_comm = world.new_communicator(ranks)
        
        r0 = rank % pke_size
        ranks = np.arange(r0, r0 + size, pke_size)
        self.spin_comm = world.new_communicator(ranks)
           
        pk0 = self.pkpt_comm.rank * npk_each
        self.my_pk = np.arange(pk0, pk0 + npk_each)
        self.my_npk = npk_each
        self.my_nspins = nspins_each  
           
        self.my_kpts = np.empty((npk_each * self.ntkmol, 3))
        kpts = self.kpts
        for i in range(self.ntkmol):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_kpts[j * self.ntkmol + i] = kpts[k * self.ntkmol + i]        

        self.my_lead_kpts = np.empty([npk_each * self.ntklead, 3])
        kpts = self.lead_kpts
        for i in range(self.ntklead):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_lead_kpts[j * self.ntklead + i] = kpts[
                                                        k * self.ntklead + i] 
        self.my_enum = 20
        e0 = self.energy_comm.rank * self.my_enum
        self.my_ep = np.arange(e0, e0 + self.my_enum)
        self.total_ep = self.my_enum * self.energy_comm.size
        #self.check_comm()
    
    def check_comm(self):
        print 'wolrd.size, world.rank, my_nspins, my_npk, energy_comm, pkpt_comm, spin_comm'
        print  world.size, world.rank, self.my_nspins, self.my_npk, self.energy_comm.size, self.energy_comm.rank, \
        self.pkpt_comm.size, self.pkpt_comm.rank, self.spin_comm.size, self.spin_comm.rank
        raise SystemExit
        
    def distribute_energy_points(self):
        rank = self.energy_comm.rank
        ns = self.my_nspins
        self.par_energy_index = np.empty([ns, self.my_npk, 2, 2], int)
        for s in range(ns):
            for k in range(self.my_npk):
                neeq = self.eqpathinfo[s][k].num
                neeq_each = neeq // self.energy_comm.size
                
                if neeq % self.energy_comm.size != 0:
                    neeq_each += 1
                begin = rank % self.energy_comm.size * neeq_each
                if (rank + 1) % self.energy_comm.size == 0:
                    end = neeq 
                else:
                    end = begin + neeq_each
                    
                self.par_energy_index[s, k, 0] = [begin, end]

                nene = self.nepathinfo[s][k].num
                nene_each = nene // self.energy_comm.size
                if nene % self.energy_comm.size != 0:
                    nene_each += 1
                begin = rank % self.energy_comm.size * nene_each
                if (rank + 1) % self.energy_comm.size == 0:
                    end = nene
                else:
                    end = begin + nene_each
      
                self.par_energy_index[s, k, 1] = [begin, end] 

    def update_lead_hamiltonian(self, l, restart_file=None):
        if not self.lead_restart and restart_file==None:
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            kpts = self.lead_kpts 
            self.hl_skmm[l], self.sl_kmm[l] = self.get_hs(atoms.calc)
            self.lead_fermi[l] = atoms.calc.get_fermi_level()
            self.dl_skmm[l] = self.initialize_density_matrix('lead', l)
            if self.save_file:
                atoms.calc.write('lead' + str(l) + '.gpw', db=True,
                                keywords=['transport', 'electrode', 'lcao'])                    
                self.pl_write('lead' + str(l) + '.mat',
                                                  (self.lead_fermi[l],
                                                   self.hl_skmm[l],
                                                   self.dl_skmm[l],
                                                   self.sl_kmm[l]))            
        else:
            if restart_file == None:
                restart_file = 'lead' + str(l)
            atoms, calc = restart_gpaw(restart_file +'.gpw')
            calc.set_positions()
            self.recover_kpts(calc)
            self.atoms_l[l] = atoms
            (self.lead_fermi[l],
             self.hl_skmm[l],
             self.dl_skmm[l],
             self.sl_kmm[l]) = self.pl_read(restart_file +'.mat')

    def update_env_hamiltonian(self, l):
        if not self.env_restart:
            atoms = self.atoms_e[l]
            atoms.get_potential_energy()
            self.he_skmm[l], self.se_kmm[l] = self.get_hs(atoms.calc)
            self.env_fermi[l] = atoms.calc.get_fermi_level()
            self.de_skmm[l] = self.initialize_density_matrix('env', l)
            if self.save_file:
                atoms.calc.write('env' + str(l) + '.gpw', db=True,
                                 keywords=['transport',
                                        'electrode','env', 'lcao'])                    
                self.pl_write('env' + str(l) + '.mat',
                                                  (self.he_skmm[l],
                                                   self.de_skmm[l],
                                                   self.se_kmm[l]))            
        else:
            atoms, calc = restart_gpaw('env' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_e[l] = atoms
            (self.he_skmm[l],
             self.de_skmm[l],
             self.se_kmm[l]) = self.pl_read('env' + str(l) + '.mat')
            
    def update_scat_hamiltonian(self, atoms):
        if not self.scat_restart:
            if atoms is None:
                atoms = self.atoms
            if not self.fixed:
                GPAW.get_potential_energy(self, atoms)
                self.h_skmm, self.s_kmm = self.get_hs(self, 'scat')               
            
            self.atoms = atoms.copy()
            rank = world.rank
            if self.gamma:
                self.h_skmm = np.real(self.h_skmm).copy()
         
            if not self.fixed:
                self.d_skmm = self.initialize_density_matrix('scat')
            else:
                self.d_skmm = np.zeros(self.h_skmm.shape, self.h_skmm.dtype)
            
            if self.save_file and not self.fixed:
                self.write('scat.gpw', db=True, keywords=['transport',
                                                          'scattering region',
                                                          'lcao'])
                self.pl_write('scat.mat', (self.h_skmm,
                                           self.d_skmm,
                                           self.s_kmm))
            self.save_file = False
        else:
            self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(
                                                     self.restart_file + '.mat')
            self.set_text('restart.txt', self.verbose)
            self.scat_restart = False
            
    def get_hs(self, calc, region='lead'):
        wfs = calc.wfs
        eigensolver = wfs.eigensolver
        ham = calc.hamiltonian
        S_qMM = wfs.S_qMM.copy()
        for S_MM in S_qMM:
            tri2full(S_MM)
        H_sqMM = np.empty((self.my_nspins,) + S_qMM.shape, complex)
        for kpt in wfs.kpt_u:
            if self.fixed and region == 'scat':
                H_MM = self.calculate_hamiltonian_matrix(kpt)
            else:
                eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
                H_MM = eigensolver.H_MM
            tri2full(H_MM)
            H_MM *= Hartree
            if self.fixed and region == 'scat':
                ind = self.gate_mol_index
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                H_MM[ind.T, ind] += self.gate * S_qMM[kpt.q, ind.T, ind]
            if self.my_nspins == 2:
                H_sqMM[kpt.s, kpt.q] = H_MM
            else:
                H_sqMM[0, kpt.q] = H_MM
        return H_sqMM, S_qMM

    def calculate_hamiltonian_matrix(self, kpt):
        assert self.fixed
        wfs = self.wfs
        hamiltonian = self.hamiltonian
        eigensolver = wfs.eigensolver
        assert eigensolver.has_initialized
        s = kpt.s
        q = kpt.q
        H_MM = self.surround.calculate_potential_matrix(hamiltonian.vt_sG,
                                                                      s, q)
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop

        test = np.zeros(H_MM.shape, H_MM.dtype) 
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][s]), P_Mi.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), P_Mi.dtype)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            if Mstart != -1:
                P_Mi = P_Mi[Mstart:Mstop]
            gemm(1.0, dHP_iM, P_Mi, 1.0, H_MM)
            gemm(1.0, dHP_iM, P_Mi, 1.0, test)            
          
        eigensolver.gd.comm.sum(H_MM)
        H_MM += self.surround.sah_spkmm[s, q]
        test += self.surround.sah_spkmm[s, q]
        self.test_atomic_hamiltonian_matrix = np.diag(test)
        H_MM += wfs.T_qMM[q]
        return H_MM
        
    def get_lead_atoms(self, l):
        """Here is a multi-terminal version """
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl.center()
        atomsl._pbc[self.d] = True
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_env_atoms(self, l):
        atoms = self.atoms.copy()
        atomsl = atoms[self.env_atoms[l]]
        atomsl.cell = self.env_cells[l]
        atomsl.center()
        atomsl._pbc = np.array(self.env_pbc, dtype=bool)
        atomsl.set_calculator(self.get_env_calc(l))
        return atomsl
    
    def get_lead_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        if not hasattr(self, 'pl_kpts') or self.pl_kpts==None:
            kpts = self.kpts
            kpts[self.d] = 2 * int(25.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts = self.pl_kpts
        p['kpts'] = kpts
        if 'mixer' in p:
            if not self.spinpol:
                p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
            else:
                p['mixer'] = MixerDif(0.1, 5, metric='new', weight=100.0)
        p['poissonsolver'] = PoissonSolver(nn=2)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return gpaw.GPAW(**p)
    
    def get_env_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        #p['usesymm'] = True
        p['kpts'] = self.env_kpts
        if 'mixer' in p:
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'env%i_' % (l + 1) + p['txt']
        return gpaw.GPAW(**p)

    def negf_prepare(self, atoms=None):
        if not self.initialized_transport:
            self.initialize_transport()
        self.update_scat_hamiltonian(atoms)
        world.barrier()
        self.initialize_mol()
        self.boundary_check()
    
    def initialize_lead(self, l):
        nspins = self.my_nspins
        ntk = self.ntklead
        nblead = self.nblead[l]
        kpts = self.my_lead_kpts
        position = [0, 0, 0]
        spk = self.substract_pk
        if l == 0:
            position[self.d] = 1.0
        elif l == 1:
            position[self.d] = -1.0
        else:
            raise NotImplementError('can no deal with multi-terminal now')
        self.hl_spkmm[l] = spk(ntk, kpts, self.hl_skmm[l], 'h')
        self.sl_pkmm[l] = spk(ntk, kpts, self.sl_kmm[l])
        self.hl_spkcmm[l] = spk(ntk, kpts, self.hl_skmm[l], 'h', position)
        self.sl_pkcmm[l] = spk(ntk, kpts, self.sl_kmm[l], 's', position)
        self.dl_spkmm[l] = spk(ntk, kpts, self.dl_skmm[l], 'h')
        self.dl_spkcmm[l] = spk(ntk, kpts, self.dl_skmm[l], 'h', position)

    def initialize_env(self, l):
        wfs = self.atoms_e[l].calc.wfs
        kpts = wfs.ibzk_kc
        weight = wfs.weight_k
        self.he_smm[l], self.se_mm[l] = get_realspace_hs(self.he_skmm[l],
                                                         self.se_kmm[l],
                                                         kpts,
                                                         weight,
                                                         R_c=(0,0,0),
                                                         usesymm=False)
        self.de_smm[l] = get_realspace_hs(self.de_skmm[l],
                                          None,
                                          kpts,
                                          weight,
                                          R_c=(0,0,0),
                                          usesymm=False)
        self.env_ibzk_kc[l] = kpts
        self.env_weight[l] = weight
        
    def initialize_mol(self):
        ntk = self.ntkmol
        kpts = self.my_kpts
        self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
        self.s_pkmm = self.substract_pk(ntk, kpts, self.s_kmm)
        self.d_spkmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h')
        self.remove_matrix_corner()
        #This line only for two_terminal
        if self.LR_leads:
            self.s_pkcmm , self.d_spkcmm = self.fill_density_matrix()
        else:
            self.s_pkcmm = self.substract_pk(ntk, kpts, self.s_kmm, 's', (0,0,1.))
            self.d_spkcmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h', (0,0,1.))
            print 'need something more for 2-D surrounding'

    def substract_pk(self, ntk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        npk = self.my_npk
        weight = np.array([1.0 / ntk] * ntk )
        if hors not in 'hs':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / ntk,) + dim[2:]
            pk_mm = np.empty(dim, k_mm.dtype)
            dim = (dim[0],) + (ntk,) + dim[2:]
            tk_mm = np.empty(dim, k_mm.dtype)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / ntk,) + dim[1:]
            pk_mm = np.empty(dim, k_mm.dtype)
            dim = (ntk,) + dim[1:]
            tk_mm = np.empty(dim, k_mm.dtype)
        if self.LR_leads:
            tkpts = self.pick_out_tkpts(ntk, kpts)
        else:
            tkpts = kpts
        for i in range(npk):
            n = i * ntk
            for j in range(ntk):
                if hors == 'h':
                    tk_mm[:, j] = np.copy(k_mm[:, n + j])
                elif hors == 's':
                    tk_mm[j] = np.copy(k_mm[n + j])
            if hors == 'h':
                pk_mm[:, i] = k2r_hs(tk_mm, None, tkpts, weight, position)
            elif hors == 's':
                pk_mm[i] = k2r_hs(None, tk_mm, tkpts, weight, position)
        return pk_mm   
            
    def check_edge(self):
        tolx = 1e-6
        position = [0,0,0]
        position[self.d] = 2.0
        ntk = self.ntklead
        kpts = self.my_lead_kpts
        s_pkmm = self.substract_pk(ntk, kpts, self.sl_kmm[0], 's', position)
        matmax = np.max(abs(s_pkmm))
        if matmax > tolx:
            self.text('Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax)
    
    def get_edge_density(self):
        for n in range(self.lead_num):
            for i in range(self.my_nspins):
                for j in range(self.my_npk):
                    self.ed_pkmm[n][i, j] = dot(self.dl_spkcmm[n][i, j],
                                                 self.sl_pkcmm[n][j].T.conj())
                    self.ec[n, i] += np.real(np.trace(self.ed_pkmm[n][i, j]))   
        self.pkpt_comm.sum(self.ec)
        self.ed_pkmm *= 3 - self.nspins
        self.ec *= 3 - self.nspins
        self.total_edge_charge = 0
        if self.spin_comm.rank == 0:
            for i in range(self.my_nspins):
                for n in range(self.lead_num):
                    self.total_edge_charge  += self.ec[n, i] / self.npk
        self.text('edge_charge =%f' % (self.total_edge_charge))

    def pick_out_tkpts(self, ntk, kpts):
        npk = self.npk
        tkpts = np.zeros([ntk, 3])
        for i in range(ntk):
            tkpts[i, self.d] = kpts[i, self.d]
        return tkpts

    def initialize_density_matrix(self, region, l=0):
        npk = self.npk
        my_npk = self.my_npk
        ns = self.my_nspins
        
        if region == 'lead':
            ntk = self.ntklead
            calc = self.atoms_l[l].calc
            d_skmm = np.empty(self.hl_skmm[l].shape, self.hl_skmm[l].dtype)
            nk = ntk * npk
            weight = [1. / nk] * ntk * my_npk * ns

        if region == 'env':
            calc = self.atoms_e[l].calc
            weight = calc.wfs.weight_k
            d_skmm = np.empty(self.he_skmm[l].shape, self.he_skmm[l].dtype)
        
        if region == 'scat':
            ntk = self.ntkmol
            calc = self
            d_skmm = np.empty(self.h_skmm.shape, self.h_skmm.dtype)
            nk = ntk * npk
            weight = [1. / nk] * ntk * my_npk * ns
        for kpt, i in zip(calc.wfs.kpt_u, range(len(calc.wfs.kpt_u))):
            C_nm = kpt.C_nM
            f_nn = np.diag(kpt.f_n)
            if ns == 2:
                d_skmm[kpt.s, kpt.q] = np.dot(C_nm.T.conj(),
                                             np.dot(f_nn, C_nm)) / weight[i]
            else:
                d_skmm[0, kpt.q] = np.dot(C_nm.T.conj(),
                                             np.dot(f_nn, C_nm)) / weight[i]
        return d_skmm
    
    def fill_density_matrix(self):
        nb = self.nblead[0]
        dtype = self.s_pkmm.dtype
        s_pkcmm = np.zeros(self.s_pkmm.shape, dtype)
        s_pkcmm[:, -nb:, :nb] = self.sl_pkcmm[0]
        d_spkcmm = np.zeros(self.d_spkmm.shape, dtype)
        d_spkcmm[:, :, -nb:, :nb] = self.dl_spkcmm[0]                    
        return s_pkcmm, d_spkcmm

    def boundary_check(self):
        tol = 5.e-4
        if self.use_lead:
            ham_diff = np.empty([self.lead_num])
            den_diff = np.empty([self.lead_num])
            self.e_float = np.empty([self.lead_num])
            for i in range(self.lead_num):
                ind = self.lead_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                h_mat_diff = self.h_spkmm[:, :, ind.T, ind] - self.hl_spkmm[i]
                d_mat_diff = self.d_spkmm[:, :, ind.T, ind] - self.dl_spkmm[i]
                ham_diff[i] = np.max(abs(h_mat_diff))
                den_diff[i] = np.max(abs(d_mat_diff))
                im = self.edge_index[1][i]
                il = self.edge_index[0][i]
                self.e_float[i] = (self.h_spkmm[0, 0, im, im] -
                                          self.hl_spkmm[i][0, 0, il, il]) / \
                                          self.sl_pkmm[i][0, il, il]
            
            self.edge_ham_diff = np.max(ham_diff)
            self.edge_den_diff = np.max(den_diff)
            if self.align:
                for i in range(self.lead_num):
                    self.hl_spkmm[i][:] += self.sl_pkmm[i] * self.e_float[i]
                    self.hl_spkcmm[i][:] += self.sl_pkcmm[i] * self.e_float[i]
                    self.hl_skmm[i][:] += self.sl_kmm[i] * self.e_float[i]
                    self.lead_fermi[i] += self.e_float[i]
                self.fermi = self.lead_fermi[0]    

            self.text('********cancel boundary shift***********')

            for i in range(self.lead_num):
                if ham_diff[i] > tol and self.master:
                    self.text('Warning*: hamiltonian boundary difference lead %d  %f' %
                                                                 (i, self.e_float[i]))
                if den_diff[i] > tol and self.master:
                    self.text('Warning*: density boundary difference lead %d %f' % 
                                                             (i, den_diff[i]))

        if self.use_env:
            env_ham_diff = np.empty([self.env_num])
            env_den_diff = np.empty([self.env_num])            
            self.env_e_float = np.empty([self.env_num])
            for i in range(self.env_num):
                ind = self.env_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                env_h_mat_diff = self.h_skmm[:, :, ind.T, ind] - \
                                                               self.he_skmm[i]
                env_d_mat_diff = self.d_skmm[:, :, ind.T, ind] - \
                                                               self.de_skmm[i]            
                env_ham_diff[i] = np.max(abs(env_h_mat_diff))
                env_den_diff[i] = np.max(abs(env_d_mat_diff))
                im = self.env_edge_index[1][i]
                ie = self.env_edge_index[0][i]
                self.env_e_float[i] = (self.h_skmm[0, 0, im, im] -
                                       self.he_skmm[i][0, 0, ie, ie]) / \
                                       self.se_kmm[i][0, ie, ie]
            self.env_edge_ham_diff = np.max(env_ham_diff)
            self.env_dege_den_diff = np.max(env_den_diff)
        
            for i in range(self.env_num):
                self.he_skmm[i][:] += self.se_kmm[i] * self.env_e_float[i]
                self.he_smm[i][:] += self.se_mm[i] * self.env_e_float[i]

            
    def get_selfconsistent_hamiltonian(self):
        self.initialize_scf()
        ##temperary lines
        self.hamiltonian.S = 0
        self.hamiltonian.Etot = 0
        ##temp
        while not self.cvgflag and self.step < self.max_steps:
            self.iterate()
            self.cvgflag = self.d_cvg and self.h_cvg
            self.step +=  1
        
        self.scf.converged = self.cvgflag
        for kpt in self.wfs.kpt_u:
            kpt.rho_MM = None
            kpt.eps_n = np.zeros((self.nbmol))
            kpt.f_n = np.zeros((self.nbmol))

        self.linear_mm = None
        if not self.scf.converged:
            raise RuntimeError('Transport do not converge in %d steps' %
                                                              self.max_steps)
    
    def get_hamiltonian_matrix(self):
        self.timer.start('HamMM')            
        self.den2fock()
        self.timer.stop('HamMM')
        self.h_spkmm = self.substract_pk(self.ntkmol, self.my_kpts,
                                         self.h_skmm, 'h')
        self.remove_matrix_corner()
        if self.master:
            self.text('HamMM', self.timer.gettime('HamMM'), 'second')        
  
    def get_density_matrix(self):
        self.timer.start('DenMM')
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        ns = self.my_nspins

        if self.use_qzk_boundary:
            self.fill_lead_with_scat()
            for i in range(self.lead_num):
                self.selfenergies[i].set_bias(0)
        if self.recal_path:
            nb = self.nbmol_inner
            npk = self.my_npk
            den = np.empty([ns, npk, nb, nb], complex)
            denocc = np.empty([ns, npk, nb, nb], complex)
            if self.cal_loc:
                denloc = np.empty([ns, npk, nb, nb], complex) 
            for s in range(ns):
                for k in range(self.my_npk):
                    den[s, k] = self.get_eqintegral_points(s, k)
                    denocc[s, k] = self.get_neintegral_points(s, k)
                    if self.cal_loc:
                        denloc[s, k] = self.get_neintegral_points(s, k,
                                                                  'locInt')
                    d_mm = self.spin_coff * (
                                                              den[s, k] +
                                                              denocc[s, k])
                    self.d_spkmm[s, k, ind.T, ind] = (d_mm +
                                                      d_mm.T.conj()) / 2
                    #del self.eqpathinfo[s][k][:]
                    #del self.nepathinfo[s][k][:]
        else:
            for s in range(ns):
                for k in range(self.my_npk):
                    self.d_spkmm[s, k, ind.T, ind] = self.spin_coff *   \
                                                     self.fock2den(s, k)
        self.add_matrix_corner()
        self.timer.stop('DenMM')
        if self.master:
            n_epoint = len(self.eqpathinfo[0][0].energy) + len(
                                         self.nepathinfo[0][0].energy)
            self.text('Energy Points on integral path %d' % n_epoint)
            self.text('DenMM', self.timer.gettime('DenMM'), 'second')

    def iterate(self):
        if self.master:
            self.text('----------------step %d -------------------'
                                                                % self.step)
        #self.keep_trace()
        self.h_cvg = self.check_convergence('h')
        self.get_density_matrix()
        self.get_hamiltonian_matrix()
        self.d_cvg = self.check_convergence('d')
        self.txt.flush()

    def keep_trace(self):
        data = self.get_boundary_info()
        self.ele_data['step_data' + str(self.step)] = data
        if self.master:
            fd = file('ele.dat', 'wb')
            pickle.dump(self.ele_data, fd, 2)
            fd.close()
        
    def check_convergence(self, var):
        cvg = False
        if var == 'h':
            if self.step > 0:
                self.diff_h = self.gd.integrate(np.fabs(self.hamiltonian.vt_sG -
                                    self.ham_vt_old))
                self.diff_h = np.max(self.diff_h)
                if self.master:
                    self.text('hamiltonian: diff = %f  tol=%f' % (self.diff_h,
                                                  self.ham_vt_tol))
                if self.diff_h < self.ham_vt_tol:
                    cvg = True
            self.ham_vt_old = np.copy(self.hamiltonian.vt_sG)
        if var == 'd':
            if self.step > 0:
                self.diff_d = self.density.mixer.get_charge_sloshing()
                if self.step == 1:
                    self.min_diff_d = self.diff_d
                elif self.diff_d < self.min_diff_d:
                    self.min_diff_d = self.diff_d
                    #self.output('step')
                if self.master:
                    self.text('density: diff = %f  tol=%f' % (self.diff_d,
                                                  self.scf.max_density_error))
                if self.diff_d < self.scf.max_density_error:
                    cvg = True
        return cvg
 
    def initialize_scf(self):
        bias = self.bias + self.env_bias
        if not self.fixed:
            self.intctrl = IntCtrl(self.occupations.kT * Hartree,
                                                        self.lead_fermi, bias)
        else:
            self.intctrl = IntCtrl(self.occupations.kT * Hartree,
                                                        self.lead_fermi, bias)            
            self.surround.reset_bias(bias) 
        self.initialize_green_function()
        self.calculate_integral_path()
        self.distribute_energy_points()
    
        if self.master:
            self.text('------------------Transport SCF-----------------------') 
            bias_info = 'Bias:'
            for i in range(self.lead_num):
                bias_info += 'lead' + str(i) + ': ' + str(self.bias[i]) + 'V'
            self.text(bias_info)
            self.text('Gate: %f V' % self.gate)

        #------for check convergence------
        self.ham_vt_old = np.empty(self.hamiltonian.vt_sG.shape)
        self.ham_vt_diff = None
        self.ham_vt_tol = 1e-2
        
        self.step = 0
        self.cvgflag = False
        self.spin_coff = 3. - self.nspins
        self.max_steps = 200
        self.h_cvg = False
        self.d_cvg = False
        self.ele_data = {}
        
    def initialize_path(self):
        self.eqpathinfo = []
        self.nepathinfo = []
        self.locpathinfo = []
        for s in range(self.my_nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            if self.cal_loc:
                self.locpathinfo.append([])                
            if self.cal_loc:
                self.locpathinfo.append([])
            for k in self.my_pk:
                self.eqpathinfo[s].append(PathInfo('eq', self.lead_num + self.env_num))
                self.nepathinfo[s].append(PathInfo('ne', self.lead_num + self.env_num))    
                if self.cal_loc:
                    self.locpathinfo[s].append(PathInfo('eq',
                                                         self.lead_num + self.env_num))
                    
    def calculate_integral_path(self):
        self.initialize_path()
        nb = self.nbmol_inner
        ns = self.my_nspins
        npk = self.my_npk
        den = np.empty([ns, npk, nb, nb], complex)
        denocc = np.empty([ns, npk, nb, nb], complex)
        if self.cal_loc:
            denloc = np.empty([ns, npk, nb, nb], complex)            
        for s in range(ns):
            for k in range(npk):      
                den[s, k] = self.get_eqintegral_points(s, k)
                denocc[s, k] = self.get_neintegral_points(s, k)
                if self.cal_loc:
                    denloc[s, k] = self.get_neintegral_points(s, k, 'locInt')        
        
    def get_eqintegral_points(self, s, k):
        maxintcnt = 50
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], self.d_spkmm.dtype)
        intctrl = self.intctrl
        
        self.zint = [0] * maxintcnt
        self.fint = []

        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append([])
        for i in range(self.env_num):
            nbenv = self.nbenv[i]
            self.tgtint.append([])
        self.cntint = -1

        self.reset_lead_hs(s, k)        
        if self.use_env:
            print 'Attention here, maybe confusing npk and nk'
            env_sg = self.env_selfenergies
            for i in range(self.env_num):
                env_sg[i].h_skmm = self.he_skmm[i]
                env_sg[i].s_kmm = self.se_kmm[i]
        
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]
       
        #--eq-Integral-----
        [grsum, zgp, wgp, fcnt] = function_integral(self, 'eqInt')
        # res Calcu
        grsum += self.calgfunc(intctrl.eqresz, 'resInt')    
        grsum.shape = (nbmol, nbmol)
        den += 1.j * (grsum - grsum.T.conj()) / np.pi / 2

        # --sort SGF --
        nres = len(intctrl.eqresz)
        self.eqpathinfo[s][k].set_nres(nres)
        elist = zgp + intctrl.eqresz
        wlist = wgp + [1.0] * nres

        fcnt = len(elist)
        sgforder = [0] * fcnt
        for i in range(fcnt):
            sgferr = np.min(abs(elist[i] -
                                      np.array(self.zint[:self.cntint + 1 ])))
                
            sgforder[i] = np.argmin(abs(elist[i]
                                     - np.array(self.zint[:self.cntint + 1])))
            if sgferr > 1e-12:
                self.text('Warning: SGF not Found. eqzgp[%d]= %f %f'
                                                        %(i, elist[i],sgferr))
        flist = []
        siglist = []
        for i in range(self.lead_num + self.env_num):
            siglist.append([])
        for i in sgforder:
            flist.append(self.fint[i])

        for i in range(self.lead_num + self.env_num):
            for j in sgforder:
                sigma = self.tgtint[i][j]
                siglist[i].append(sigma)
        self.eqpathinfo[s][k].add(elist, wlist, flist, siglist)    
   
        del self.fint, self.tgtint, self.zint
        return den 
    
    def get_neintegral_points(self, s, k, calcutype='neInt'):
        intpathtol = 1e-8
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        maxintcnt = 50
        intctrl = self.intctrl

        self.zint = [0] * maxintcnt
        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append([])
        
        for i in range(self.env_num):
            nbenv = self.nbenv[i]
            self.tgtint.append([])

        self.reset_lead_hs(s, k)
        if self.use_env:
            print 'Attention here, maybe confusing npk and nk'
            env_sg = self.env_selfenergies
            for i in range(self.env_num):
                env_sg[i].h_skmm = self.he_skmm[i]
                env_sg[i].s_kmm = self.se_kmm[i]
                
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        if calcutype == 'neInt' or calcutype == 'neVirInt':
            for n in range(1, len(intctrl.neintpath)):
                self.cntint = -1
                self.fint = []
                for i in range(self.lead_num + self.env_num):
                    self.fint.append([[],[]])
                if intctrl.kt <= 0:
                    neintpath = [intctrl.neintpath[n - 1] + intpathtol,
                                 intctrl.neintpath[n] - intpathtol]
                else:
                    neintpath = [intctrl.neintpath[n-1], intctrl.neintpath[n]]
                if intctrl.neintmethod== 1:
    
                    # ----Auto Integral------
                    sumga, zgp, wgp, nefcnt = function_integral(self,
                                                                    calcutype)
    
                    nefcnt = len(zgp)
                    sgforder = [0] * nefcnt
                    for i in range(nefcnt):
                        sgferr = np.min(np.abs(zgp[i] - np.array(
                                            self.zint[:self.cntint + 1 ])))
                        sgforder[i] = np.argmin(np.abs(zgp[i] -
                                       np.array(self.zint[:self.cntint + 1])))
                        if sgferr > 1e-12:
                            self.text('--Warning: SGF not found, \
                                    nezgp[%d]=%f %f' % (i, zgp[i], sgferr))
                else:
                    # ----Manual Integral------
                    nefcnt = max(np.ceil(np.real(neintpath[1] -
                                                    neintpath[0]) /
                                                    intctrl.neintstep) + 1, 6)
                    nefcnt = int(nefcnt)
                    zgp = np.linspace(neintpath[0], neintpath[1], nefcnt)
                    zgp = list(zgp)
                    wgp = np.array([3.0 / 8, 7.0 / 6, 23.0 / 24] + [1] *
                             (nefcnt - 6) + [23.0 / 24, 7.0 / 6, 3.0 / 8]) * (
                                                          zgp[1] - zgp[0])
                    wgp = list(wgp)
                    sgforder = range(nefcnt)
                    sumga = np.zeros([1, nbmol, nbmol], complex)
                    for i in range(nefcnt):
                        sumga += self.calgfunc(zgp[i], calcutype) * wgp[i]
                den += sumga[0] / np.pi / 2
                flist = [] 
                siglist = []
                for i in range(self.lead_num + self.env_num):
                    flist.append([[],[]])
                    siglist.append([])
                for l in range(self.lead_num):
                    #nblead = self.nblead[l]
                    #sigma= np.empty([nblead, nblead], complex)
                    for j in sgforder:
                        for i in [0, 1]:
                            fermi_factor = self.fint[l][i][j]
                            flist[l][i].append(fermi_factor)   
                        sigma = self.tgtint[l][j]
                        siglist[l].append(sigma)
                if self.use_env:
                    nl = self.lead_num
                    for l in range(self.env_num):
                        for j in sgforder:
                            for i in [0, 1]:
                                fermi_factor = self.fint[l + nl][i][j]
                                flist[l + nl][i].append(fermi_factor)
                            sigma = self.tgtint[l + nl][j]
                            siglist[l + nl].append(sigma)
                self.nepathinfo[s][k].add(zgp, wgp, flist, siglist)
        # Loop neintpath
        elif calcutype == 'locInt':
            self.cntint = -1
            self.fint =[]
            sumgr, zgp, wgp, locfcnt = function_integral(self, 'locInt')
            # res Calcu :minfermi
            sumgr -= self.calgfunc(intctrl.locresz[0, :], 'resInt')
            # res Calcu :maxfermi
            sumgr += self.calgfunc(intctrl.locresz[1, :], 'resInt')
            
            sumgr.shape = (nbmol, nbmol)
            den = 1.j * (sumgr - sumgr.T.conj()) / np.pi / 2
         
            # --sort SGF --
            nres = intctrl.locresz.shape[-1]
            self.locpathinfo[s][k].set_nres(2 * nres)
            loc_e = intctrl.locresz.copy()
            loc_e.shape = (2 * nres, )
            elist = zgp + loc_e.tolist()
            wlist = wgp + [-1.0] * nres + [1.0] * nres
            fcnt = len(elist)
            sgforder = [0] * fcnt
            for i in range(fcnt):
                sgferr = np.min(abs(elist[i] -
                                      np.array(self.zint[:self.cntint + 1 ])))
                
                sgforder[i] = np.argmin(abs(elist[i]
                                     - np.array(self.zint[:self.cntint + 1])))
                if sgferr > 1e-12:
                    self.text('Warning: SGF not Found. eqzgp[%d]= %f %f'
                                                        %(i, elist[i],sgferr))
            flist = []
            siglist = []
            for i in range(self.lead_num + self.env_num):
                siglist.append([])
            for i in sgforder:
                flist.append(self.fint[i])
            #sigma= np.empty([nblead, nblead], complex)
            for i in range(self.lead_num):
                for j in sgforder:
                    sigma = self.tgtint[i][j]
                    siglist[i].append(sigma)
            if self.use_env:
                nl = self.lead_num
                for i in range(self.env_num):
                    for j in sgforder:
                        sigma = self.tgtint[i + nl][j]
                        siglist[i + nl].append(sigma)
            self.locpathinfo[s][k].add(elist, wlist, flist, siglist)           
        
        del self.zint, self.tgtint
        if len(intctrl.neintpath) >= 2:
            del self.fint
        return den
         
    def calgfunc(self, zp, calcutype):			 
        #calcutype = 
        #  - 'eqInt':  gfunc[Mx*Mx,nE] (default)
        #  - 'neInt':  gfunc[Mx*Mx,nE]
        #  - 'resInt': gfunc[Mx,Mx] = gr * fint
        #              fint = -2i*pi*kt
      
        intctrl = self.intctrl
        sgftol = 1e-10
        stepintcnt = 50
        nbmol = self.nbmol_inner
                
        if type(zp) == list:
            pass
        elif type(zp) == np.ndarray:
            pass
        else:
            zp = [zp]
        nume = len(zp)
        
        if calcutype == 'resInt':
            gfunc = np.zeros([nbmol, nbmol], complex)
        else:
            gfunc = np.zeros([nume, nbmol, nbmol], complex)
        for i in range(nume):
            sigma = np.zeros([nbmol, nbmol], complex)
            gamma = np.zeros([self.lead_num, nbmol, nbmol], complex)
            if self.use_env:
                env_gamma = np.zeros([self.env_num, nbmol, nbmol], complex)
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt

            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in range(self.lead_num):
                if j == 0:
                    tri_type = 'L'
                else:
                    tri_type = 'R'
                tgt = Se_Sparse_Matrix(self.selfenergies[j](zp[i]), tri_type) 
                self.tgtint[j].append(tgt)
            
            if self.use_env:
                nl = self.lead_num
                for j in range(self.env_num):
                    self.tgtint[j + nl][self.cntint] = \
                                               self.env_selfenergies[j](zp[i])
                    
            for j in range(self.lead_num):
                ind = self.inner_lead_index[j]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                tgt = self.tgtint[j][self.cntint].restore()     
                sigma[ind.T, ind] +=  tgt       
                gamma[j, ind.T, ind] += 1.j *(tgt - tgt.T.conj())
            
            if self.use_env:
                nl = self.lead_num
                for j in range(self.env_num):
                    ind = self.inner_env_index[j]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += self.tgtint[j + nl][self.cntint]
                    env_gamma[j, ind.T, ind] += \
                                    self.env_selfenergies[j].get_lambda(zp[i])

            gr = self.greenfunction.calculate(zp[i], sigma)       
        
            # --ne-Integral---
            kt = intctrl.kt
            if calcutype == 'neInt':
                gammaocc = np.zeros([nbmol, nbmol], complex)
                for n in range(self.lead_num):
                    lead_ef = intctrl.leadfermi[n]
                    min_ef = intctrl.minfermi
                    max_ef = intctrl.maxfermi
                    self.fint[n][0].append(fermidistribution(zp[i] - lead_ef,
                                           kt) - fermidistribution(zp[i] -
                                          min_ef, kt))
                    self.fint[n][1].append(fermidistribution(zp[i] - max_ef,
                                           kt) - fermidistribution(zp[i] -
                                            lead_ef, kt))                    
                    gammaocc += gamma[n] * self.fint[n][0][self.cntint]
                if self.use_env:
                    nl = self.lead_num
                    for n in range(self.env_num):
                        env_ef = intctrl.envfermi[n]
                        min_ef = intctrl.minfermi
                        max_ef = intctrl.maxfermi
                        self.fint[n + nl][0].append(
                                         fermidistribution(zp[i] - env_ef, kt)
                                      - fermidistribution(zp[i] - min_ef, kt))
                        self.fint[n + nl][1].append(
                                         fermidistribution(zp[i] - max_ef, kt)
                                      - fermidistribution(zp[i] - env_ef, kt))
                        gammaocc += env_gamma[n] * \
                                             self.fint[n + nl][0][self.cntint]
                        
                aocc = dot(gr, gammaocc)
                aocc = dot(aocc, gr.T.conj())
                gfunc[i] = aocc

            elif calcutype == 'neVirInt':
                gammavir = np.zeros([nbmol, nbmol], complex)
                for n in range(self.lead_num):
                    lead_ef = intctrl.leadfermi[n]
                    min_ef = intctrl.minfermi
                    max_ef = intctrl.maxfermi
                    self.fint[n][0].append(fermidistribution(zp[i] - lead_ef,
                                           kt) - fermidistribution(zp[i] -
                                          min_ef, kt))
                    self.fint[n][1].append(fermidistribution(zp[i] - max_ef,
                                           kt) - fermidistribution(zp[i] -
                                            lead_ef, kt))
                    gammavir += gamma[n] * self.fint[n][1][self.cntint]
                if self.use_env:
                    nl = self.lead_num
                    for n in range(self.env_num):
                        env_ef = intctrl.envfermi[n]
                        min_ef = intctrl.minfermi
                        max_ef = intctrl.maxfermi
                        self.fint[n + nl][0].append(
                                         fermidistribution(zp[i] - env_ef, kt)
                                      - fermidistribution(zp[i] - min_ef, kt))
                        self.fint[n + nl][1].append(
                                         fermidistribution(zp[i] - max_ef, kt)
                                      - fermidistribution(zp[i] - env_ef, kt))
                        gammavir += env_gamma[n] * \
                                             self.fint[n + nl][1][self.cntint]                
                avir = dot(gr, gammavir)
                avir = dot(avir, gr.T.conj())
                gfunc[i] = avir
            # --local-Integral--
            elif calcutype == 'locInt':
                # fmax-fmin
                max_ef = intctrl.maxfermi
                min_ef = intctrl.minfermi
                self.fint.append(fermidistribution(zp[i] - max_ef, kt) - 
                                 fermidistribution(zp[i] - min_ef, kt) )
                gfunc[i] = gr * self.fint[self.cntint]
 
            # --res-Integral --
            elif calcutype == 'resInt':
                self.fint.append(-2.j * np.pi * kt)
                gfunc += gr * self.fint[self.cntint]
            #--eq-Integral--
            else:
                if kt <= 0:
                    self.fint.append(1.0)
                else:
                    min_ef = intctrl.minfermi
                    self.fint.append(fermidistribution(zp[i] - min_ef, kt))
                gfunc[i] = gr * self.fint[self.cntint]    
        return gfunc        
    
    def fock2den(self, s, k):
        intctrl = self.intctrl
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        den = self.eq_fock2den(s, k)
        denocc = self.ne_fock2den(s, k, ov='occ')    
        den += denocc

        if self.cal_loc:
            denloc = self.eq_fock2den(s, k, el='loc')
            denvir = self.ne_fock2den(s, k, ov='vir')
            weight_mm = self.integral_diff_weight(denocc, denvir,
                                                                 'transiesta')
            diff = (denloc - (denocc + denvir)) * weight_mm
            den += diff
            percents = np.sum( diff * diff ) / np.sum( denocc * denocc )
            self.text('local percents %f' % percents)
        den = (den + den.T.conj()) / 2
        return den    

    def ne_fock2den(self, s, k, ov='occ'):
        pathinfo = self.nepathinfo[s][k]
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        begin = self.par_energy_index[s, k, 1, 0]
        end = self.par_energy_index[s, k, 1, 1]        
        zp = pathinfo.energy

        for i in range(begin, end):
            sigma = np.zeros(den.shape, complex)
            sigmalesser = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i].restore()
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]
            gr = self.greenfunction.calculate(zp[i], sigma)

            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])                
                sigmatmp = pathinfo.sigma[n][i].restore()
                if ov == 'occ':
                    fermifactor = np.real(pathinfo.fermi_factor[n][0][i])
                elif ov == 'vir':
                    fermifactor = np.real(pathinfo.fermi_factor[n][1][i])                    
                sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())
            
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigmatmp = pathinfo[n + nl][i]
                    if ov == 'occ':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][0][i])
                    elif ov == 'vir':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][1][i])
                    sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                             sigmatmp - sigmatmp.T.conj())
   
            glesser = dot(sigmalesser, gr.T.conj())
            glesser = dot(gr, glesser)
            weight = pathinfo.weight[i]            
            den += glesser * weight / np.pi / 2
        self.energy_comm.sum(den)
        return den  

    def eq_fock2den(self, s, k, el='eq'):
        if el =='loc':
            pathinfo = self.locpathinfo[s][k]
        else:
            pathinfo = self.eqpathinfo[s][k]
        
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        begin = self.par_energy_index[s, k, 0, 0]
        end = self.par_energy_index[s, k, 0, 1]
        zp = pathinfo.energy
        for i in range(begin, end):
            sigma = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i].restore()
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]
            gr = self.greenfunction.calculate(zp[i], sigma)
            fermifactor = pathinfo.fermi_factor[i]
            weight = pathinfo.weight[i]
            den += gr * fermifactor * weight
        self.energy_comm.sum(den)
        den = 1.j * (den - den.T.conj()) / np.pi / 2            
        return den

    def den2fock(self):
        self.get_density()
        self.update_hamiltonian(self.density)
        if self.use_linear_vt_array:
            self.hamiltonian.vt_sG += self.get_linear_potential()
        self.h_skmm, self.s_kmm = self.get_hs(self, 'scat')
        #self.spy('h_skmm', self.h_skmm)
        #self.spy('s_kmm', self.s_kmm)
        if self.use_linear_vt_mm:
            if self.linear_mm == None:
                self.linear_mm = self.get_linear_potential_matrix()            
            self.h_skmm += self.linear_mm
   
    def get_forces(self, atoms):
        if (atoms.positions != self.atoms.positions).any():
            self.scf.converged = False
        if hasattr(self.scf, 'converged') and self.scf.converged:
            pass
        else:
            self.negf_prepare(atoms)
            self.get_selfconsistent_hamiltonian()
        self.forces.F_av = None
        f = GPAW.get_forces(self, atoms)
        return f
    
    def get_potential_energy(self, atoms=None, force_consistent=False):
        if hasattr(self.scf, 'converged') and self.scf.converged:
            pass
        else:
            self.negf_prepare()
            self.get_selfconsistent_hamiltonian()
        if force_consistent:
            # Free energy:
            return Hartree * self.hamiltonian.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return Hartree * (self.hamiltonian.Etot + 0.5 * self.hamiltonian.S)
       
    def get_density(self):
        #Calculate pseudo electron-density based on green function.
        ns = self.my_nspins
        ntk = self.ntkmol
        npk = self.my_npk
        nb = self.nbmol
        dr_mm = np.zeros([ns, npk, 3, nb, nb], self.d_spkmm.dtype)
        qr_mm = np.zeros([ns, npk, nb, nb])
        
        for s in range(ns):
            for i in range(npk):
                dr_mm[s, i, 0] = self.d_spkcmm[s, i].T.conj()
                dr_mm[s, i, 1] = self.d_spkmm[s, i]
                dr_mm[s, i, 2]= self.d_spkcmm[s, i]
                qr_mm[s, i] += dot(dr_mm[s, i, 1], self.s_pkmm[i]) 
        
        for i in range(self.lead_num):
            ind = self.print_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            qr_mm[:, :, ind.T, ind] += self.ed_pkmm[i]
        
        self.pkpt_comm.sum(qr_mm)
        self.spin_comm.sum(qr_mm)
        qr_mm /= self.npk
        world.barrier()
        
        if self.master:
            self.print_boundary_charge(qr_mm)
           
        rvector = np.zeros([3, 3])
        rvector[:, self.d] = [-1, 0, 1]
        tkpts = self.pick_out_tkpts(ntk, self.my_kpts)

        self.d_skmm.shape = (ns, npk, ntk, nb, nb)

        for s in range(ns):
            if ntk != 1:
                for i in range(ntk):
                    for j in range(npk):
                        self.d_skmm[s, j, i] = r2k_hs(None,
                                                      dr_mm[s, j, :],
                                                      rvector,
                                                      tkpts[i])
                        self.d_skmm[s, j, i] /=  ntk * self.npk 
            else:
                for j in range(npk):
                    self.d_skmm[s, j, 0] =  dr_mm[s, j, 1]
                    self.d_skmm[s, j, 0] /= self.npk 
        self.d_skmm.shape = (ns, ntk * npk, nb, nb)
       
        for kpt in self.wfs.kpt_u:
            if ns == 2:
                kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
            else:
                kpt.rho_MM = self.d_skmm[0, kpt.q]
        self.update_density()

    def update_density(self):
        if not self.fixed:
            self.density.update(self.wfs)
        else:
            wfs = self.wfs
            density = self.density
            self.surround.calculate_pseudo_density(density, wfs)
            density.nt_sG += self.surround.streching_nt_sG
            #self.spy('nt_sG', density.nt_sG)
            #wfs.calculate_atomic_density_matrices(density.D_asp)
            self.surround.calculate_atomic_density_matrices()
            #self.spy('D_asp', density.D_asp)
            comp_charge = density.calculate_multipole_moments()
            #self.spy('comp', comp_charge)
            #if (isinstance(wfs, LCAOWaveFunctions) or
            #    extra_parameters.get('normalize')):
            #    density.normalize(comp_charge)
            #density.mix(comp_charge)
            
            
            pseudo_charge = density.gd.integrate(density.nt_sG).sum()
            #self.spy('pchar', pseudo_charge)
            print pseudo_charge, world.rank, 'pseudo_charge'
            if pseudo_charge != 0:
                x = -(density.charge + comp_charge) / pseudo_charge
                print 'scaling', x
                #density.nt_sG *= x                
            if not density.mixer.mix_rho:
                density.mixer.mix(density)
                comp_charge = None
            self.surround.interpolate_density(density, comp_charge)
            #self.spy('nt_sg', density.nt_sg)
            self.surround.calculate_pseudo_charge(density, comp_charge)
            #self.spy('rhot_g', density.rhot_g)
            if density.mixer.mix_rho:
                density.mixer.mix(density)            
            density.rhot_g -= self.surround.extra_rhot_g
            #self.spy('extra_rhot_g', self.surround.extra_rhot_g)

    def update_hamiltonian(self, density):
        ham = self.hamiltonian        
        ham.timer.start('Hamiltonian')
        if ham.vt_sg is None:
            ham.vt_sg = ham.finegd.empty(ham.nspins)
            ham.vHt_g = ham.finegd.zeros()
            ham.vt_sG = ham.gd.empty(ham.nspins)
            ham.poisson.initialize()

        Ebar = ham.finegd.integrate(ham.vbar_g, density.nt_g,
                                     global_integral=False) 

        vt_g = ham.vt_sg[0]
        vt_g[:] = ham.vbar_g

        Eext = 0.0
        if ham.vext_g is not None:
            vt_g += ham.vext_g.get_potential(ham.finegd)
            Eext = np.vdot(vt_g, density.nt_g) * ham.finegd.dv - Ebar

        if ham.nspins == 2:
            ham.vt_sg[1] = vt_g
       
        if not self.fixed: 
            if ham.nspins == 2:
                Exc = ham.xc.get_energy_and_potential(
                    density.nt_sg[0], ham.vt_sg[0],
                    density.nt_sg[1], ham.vt_sg[1])
            else:
                Exc = ham.xc.get_energy_and_potential(
                    density.nt_sg[0], ham.vt_sg[0])
        else:
            Exc, ham.vt_sg = self.surround.get_xc(density.nt_sg, ham.vt_sg)

        ham.timer.start('Poisson')
        # npoisson is the number of iterations:
        assert abs(density.charge) < 1e-6
        ham.npoisson = ham.poisson.solve_neutral(ham.vHt_g, density.rhot_g,
                                                          eps=ham.poisson.eps)
        ham.timer.stop('Poisson')
        #self.spy('vHt_g', ham.vHt_g)
        dim = density.rhot_g.shape[0] / 2
        if self.fixed:
            self.surround.boundary_data['rhot_g1'] = density.rhot_g[dim, dim]
            self.surround.boundary_data['vHt_g1'] = ham.vHt_g[dim, dim]        
        Epot = 0.5 * ham.finegd.integrate(ham.vHt_g, density.rhot_g,
                                           global_integral=False)
        Ekin = 0.0
        if not self.fixed:
            for vt_g, vt_G, nt_G in zip(ham.vt_sg, ham.vt_sG, density.nt_sG):
                vt_g += ham.vHt_g
                ham.restrict(vt_g, vt_G)
                Ekin -= ham.gd.integrate(vt_G, nt_G - density.nct_G,
                                          global_integral=False)            
        else:
            for vt_g, nt_G, s in zip(ham.vt_sg, density.nt_sG, range(ham.nspins)):
                vt_g += ham.vHt_g
                ham.vt_sG[s] = self.surround.restrict(ham.vt_sg, s)
                Ekin -= ham.gd.integrate(ham.vt_sG[s], nt_G - density.nct_G,
                                               global_integral=False)
        #self.spy('vt_sg', ham.vt_sg)
        #self.spy('vt_sG', ham.vt_sG)
        # Calculate atomic hamiltonians:
        ham.timer.start('Atomic Hamiltonians')
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((ham.setups[a].lmax + 1)**2)
        density.ghat.integrate(ham.vHt_g, W_aL)
        ham.dH_asp = {}
        for a, D_sp in density.D_asp.items():
            W_L = W_aL[a]
            setup = ham.setups[a]

            D_p = D_sp.sum(0)
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            Ekin += np.dot(setup.K_p, D_p) + setup.Kc
            Ebar += setup.MB + np.dot(setup.MB_p, D_p)
            Epot += setup.M + np.dot(D_p, (setup.M_p + np.dot(setup.M_pp, D_p)))

            if setup.HubU is not None:
                nspins = len(ham.D_sp)
                i0 = setup.Hubi
                i1 = i0 + 2 * setup.Hubl + 1
                for D_p, H_p in zip(ham.D_sp, ham.H_sp): # XXX ham.H_sp ??
                    N_mm = unpack2(D_p)[i0:i1, i0:i1] / 2 * nspins 
                    Eorb = setup.HubU/2. * (N_mm - np.dot(N_mm,N_mm)).trace()
                    Vorb = setup.HubU * (0.5 * np.eye(i1-i0) - N_mm)
                    Exc += Eorb                    
                    Htemp = unpack(H_p)
                    Htemp[i0:i1,i0:i1] += Vorb
                    H_p[:] = pack2(Htemp)

            if 0:#vext is not None:
                # Tailor expansion to the zeroth order
                Eext += vext[0][0] * (sqrt(4 * pi) * ham.Q_L[0] + setup.Z)
                dH_p += vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0]
                if len(vext) > 1:
                    # Tailor expansion to the first order
                    Eext += sqrt(4 * pi / 3) * np.dot(vext[1], ham.Q_L[1:4])
                    # there must be a better way XXXX
                    Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                          setup.Delta_pL[:, 2],
                                          setup.Delta_pL[:, 3]])
                    dH_p += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)

            ham.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, dH_sp, a)
            dH_sp += dH_p

            Ekin -= (D_sp * dH_sp).sum()

        ham.timer.stop('Atomic Hamiltonians')

        # Make corrections due to non-local xc:
        xcfunc = ham.xc.xcfunc
        ham.Enlxc = xcfunc.get_non_local_energy()
        ham.Enlkin = xcfunc.get_non_local_kinetic_corrections()
        if ham.Enlxc != 0 or ham.Enlkin != 0:
            print 'Where should we do comm.sum() ?'

        comm = ham.gd.comm
        ham.Ekin0 = comm.sum(Ekin)
        ham.Epot = comm.sum(Epot)
        ham.Ebar = comm.sum(Ebar)
        ham.Eext = comm.sum(Eext)
        ham.Exc = comm.sum(Exc)

        ham.Exc += ham.Enlxc
        ham.Ekin0 += ham.Enlkin

        ham.timer.stop('Hamiltonian')
    
    def print_boundary_charge(self, qr_mm):
        qr_mm = np.sum(np.sum(qr_mm, axis=0), axis=0)
        edge_charge = []
        natom_inlead = np.empty([self.lead_num], int)
        natom_print = np.empty([self.lead_num], int)
        
        for i in range(self.lead_num):
            natom_inlead[i] = len(self.pl_atoms[i])
            nb_atom = self.nblead[i] / natom_inlead[i]
            if self.use_buffer:
                pl1 = self.buffer[i]
            else:
                pl1 = self.nblead[i]
            natom_print[i] = pl1 / nb_atom
            ind = self.print_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            edge_charge.append(np.diag(qr_mm[ind.T, ind]))
            edge_charge[i].shape = (natom_print[i], nb_atom)
            edge_charge[i] = np.sum(edge_charge[i], axis=1)
        
        self.text('***charge distribution at edges***')
        if self.verbose:
            for n in range(self.lead_num):
                info = []
                for i in range(natom_print[n]):
                    info.append('--' +  str(edge_charge[n][i])+'--')
                self.text(info)

        else:
            info = ''
            for n in range(self.lead_num):
                edge_charge[n].shape = (natom_print[n] / natom_inlead[n],
                                                             natom_inlead[n])
                edge_charge[n] = np.sum(edge_charge[n],axis=1)
                nl = int(natom_print[n] / natom_inlead[n])
                for i in range(nl):
                    info += '--' +  str(edge_charge[n][i]) + '--'
                if n != 1:
                    info += '---******---'
            self.text(info)
        self.text('***total charge***')
        self.text(np.trace(qr_mm))

    def calc_total_charge(self, d_spkmm):
        nbmol = self.nbmol 
        qr_mm = np.empty([self.my_nspins, self.my_npk, nbmol, nbmol])
        for i in range(self.my_nspins):  
            for j in range(self.my_npk):
                qr_mm[i,j] = dot(d_spkmm[i, j], self.s_pkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))
        Qmol += np.sum(self.ec)
        Qmol = self.pkpt_comm.sum(Qmol) / self.npk
        self.spin_comm.sum(Qmol)
        return Qmol        

    def get_linear_potential(self):
        local_linear_potential = self.gd.zeros(self.nspins)
        linear_potential = self.gd.collect(local_linear_potential, True)
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        buffer_dim = self.dimt_buffer
        scat_dim = dimt - np.sum(buffer_dim)
        bias= np.array(self.bias)
        bias /= Hartree
        vt = np.empty([dimt])
        if buffer_dim[1] !=0:
            vt[:buffer_dim[0]] = bias[0]
            vt[-buffer_dim[1]:] = bias[1]         
            vt[buffer_dim[0]: -buffer_dim[1]] = np.linspace(bias[0],
                                                         bias[1], scat_dim)
        else:
            vt = np.linspace(bias[0], bias[1], scat_dim)
        for s in range(self.nspins):
            for i in range(dimt):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dimp) + 1)
        self.gd.distribute(linear_potential, local_linear_potential)  
        return local_linear_potential
    
    def output(self, filename):
        self.pl_write(filename + '.mat', (self.h_skmm,
                                          self.d_skmm,
                                          self.s_kmm))
        world.barrier()
        self.write(filename + '.gpw', db=True, keywords=['transport',
                                                        'bias_step',
                                                         'lcao'])
        if self.master:
            fd = file(filename, 'wb')
            pickle.dump((
                        self.bias,
                        self.gate,
                        self.intctrl,
                        #self.eqpathinfo,
                        #self.nepathinfo,
                        self.forces,
                        self.current,
                        self.step,
                        self.cvgflag
                        ), fd, 2)
            fd.close()
        world.barrier()

    def input(self, filename):
        GPAW.__init__(self, filename + '.gpw')
        self.set_positions()
        self.initialize_transport(dryrun=True)
        fd = file(filename, 'rb')
        (self.bias,
         self.gate,
         self.intctrl,
         #self.eqpathinfo,
         #self.nepathinfo,
         self.forces,
         self.current,
         self.step,
         self.cvgflag
         ) = pickle.load(fd)
        fd.close()
        self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(filename + '.mat')
        self.initialize_mol()
     
    def set_calculator(self, e_points, leads=[0,1]):
        from ase.transport.calculators import TransportCalculator
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
     
        h_scat = self.h_spkmm[:, :, ind.T, ind]
        h_scat = np.sum(h_scat[0, :], axis=0) / self.npk
        h_scat = np.real(h_scat)
        
        l1 = leads[0]
        l2 = leads[1]
        
        h_lead1 = self.double_size(np.sum(self.hl_spkmm[l1][0], axis=0),
                                   np.sum(self.hl_spkcmm[l1][0], axis=0))
        h_lead2 = self.double_size(np.sum(self.hl_spkmm[l2][0], axis=0),
                                   np.sum(self.hl_spkcmm[l2][0], axis=0))
        h_lead1 /= self.npk
        h_lead2 /= self.npk
        
        h_lead1 = np.real(h_lead1)
        h_lead2 = np.real(h_lead2)
        
        s_scat = np.sum(self.s_pkmm[:, ind.T, ind], axis=0) / self.npk
        s_scat = np.real(s_scat)
        
        s_lead1 = self.double_size(np.sum(self.sl_pkmm[l1], axis=0),
                                   np.sum(self.sl_pkcmm[l1], axis=0))
        s_lead2 = self.double_size(np.sum(self.sl_pkmm[l2], axis=0),
                                   np.sum(self.sl_pkcmm[l2], axis=0))
        
        s_lead1 /= self.npk
        s_lead2 /= self.npk
        
        s_lead1 = np.real(s_lead1)
        s_lead2 = np.real(s_lead2)
        
        tcalc = TransportCalculator(energies=e_points,
                                    h = h_scat,
                                    h1 = h_lead1,
                                    h2 = h_lead1,
                                    s = s_scat,
                                    s1 = s_lead1,
                                    s2 = s_lead1,
                                    dos = True
                                   )
        return tcalc
    
    def calculate_dos(self, E_range=[-6,2], point_num = 60, leads=[0,1]):
        data = {}
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        tcalc = self.set_calculator(e_points, leads)
        tcalc.get_transmission()
        tcalc.get_dos()
        f1 = self.intctrl.leadfermi[leads[0]] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[leads[1]] * (np.zeros([10, 1]) + 1)
        a1 = np.max(tcalc.T_e)
        a2 = np.max(tcalc.dos_e)
        l1 = np.linspace(0, a1, 10)
        l2 = np.linspace(0, a2, 10)
        data['e_points'] = e_points
        data['T_e'] = tcalc.T_e
        data['dos_e'] = tcalc.dos_e
        data['f1'] = f1
        data['f2'] = f2
        data['l1'] = l1
        data['l2'] = l2
        return data
  
    def abstract_d_and_v(self):
        data = {}
        for s in range(self.nspins):
            nt = self.gd.collect(self.density.nt_sG[s], True)
            vt = self.gd.collect(self.hamiltonian.vt_sG[s], True)
            for name, d in [('x', 0), ('y', 1), ('z', 2)]:
                data['s' + str(s) + 'nt_1d_' +
                     name] = self.array_average_in_one_d(nt, d)
                data['s' + str(s) + 'nt_2d_' +
                     name] = self.array_average_in_two_d(nt, d)            
                data['s' + str(s) + 'vt_1d_' +
                     name] = self.array_average_in_one_d(vt, d)
                data['s' + str(s) + 'vt_2d_' +
                     name] = self.array_average_in_two_d(vt, d)
        return data    
    
    def array_average_in_one_d(self, a, d=2):
        nx, ny, nz = a.shape
        if d==0:
            b = np.array([np.sum(a[i]) for i in range(nx)]) / (ny * nz)
        if d==1:
            b = np.array([np.sum(a[:, i, :]) for i in range(ny)]) / (nx * nz)        
        if d==2:
            b = np.array([np.sum(a[:, :, i]) for i in range(nz)]) / (nx * ny)
        return b
    
    def array_average_in_two_d(self, a, d=0):
        b = np.sum(a, axis=d) / a.shape[d]
        return b        
    
    def plot_dos(self, E_range, point_num = 30, leads=[0,1]):
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        tcalc = self.set_calculator(e_points, leads)
        tcalc.get_transmission()
        tcalc.get_dos()
        f1 = self.intctrl.leadfermi[leads[0]] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[leads[1]] * (np.zeros([10, 1]) + 1)
        a1 = np.max(tcalc.T_e)
        a2 = np.max(tcalc.dos_e)
        l1 = np.linspace(0, a1, 10)
        l2 = np.linspace(0, a2, 10)
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(e_points, tcalc.T_e, 'b-o', f1, l1, 'r--', f2, l1, 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(e_points, tcalc.dos_e, 'b-o', f1, l2, 'r--', f2, l2, 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.show()
        
    def plot_v(self, vt=None, tit=None, ylab=None,
                                             l_MM=False, plot_buffer=False):
        import pylab
        self.use_linear_vt_mm = l_MM
        if vt == None:
            vt = self.hamiltonian.vt_sG + self.get_linear_potential()
        dim = vt.shape
        for i in range(3):
            vt = np.sum(vt, axis=0) / dim[i]
        db = self.dimt_buffer
        if plot_buffer:
            td = len(vt)
            pylab.plot(range(db[0]), vt[:db[0]] * Hartree, 'g--o')
            pylab.plot(range(db[0], td - db[1]),
                               vt[db[0]: -db[1]] * Hartree, 'b--o')
            pylab.plot(range(td - db[1], td), vt[-db[1]:] * Hartree, 'g--o')
        elif db[1]==0:
            pylab.plot(vt[db[0]:] * Hartree, 'b--o')
        else:
            pylab.plot(vt[db[0]: db[1]] * Hartree, 'b--o')
        if ylab == None:
            ylab = 'energy(eV)'
        pylab.ylabel(ylab)
        if tit == None:
            tit = 'bias=' + str(self.bias)
        pylab.title(tit)
        pylab.show()

    def plot_d(self, nt=None, tit=None, ylab=None, plot_buffer=False):
        import pylab
        if nt == None:
            nt = self.density.nt_sG
        dim = nt.shape
        for i in range(3):
            nt = np.sum(nt, axis=0) / dim[i]
        db = self.dimt_buffer
        if plot_buffer:
            td = len(nt)
            pylab.plot(range(db[0]), nt[:db[0]], 'g--o')
            pylab.plot(range(db[0], td - db[1]), nt[db[0]: -db[1]], 'b--o')
            pylab.plot(range(td - db[1], td), nt[-db[1]:], 'g--o')
        elif db[1] == 0:
            pylab.plot(nt[db[0]:], 'b--o')            
        else:
            pylab.plot(nt[db[0]: db[1]], 'b--o')            
        if ylab == None:
            ylab = 'density'
        pylab.ylabel(ylab)
        if tit == None:
            tit = 'bias=' + str(self.bias)
        pylab.title(tit)
        pylab.show()        
           
    def double_size(self, m_ii, m_ij):
        dim = m_ii.shape[-1]
        mtx = np.empty([dim * 2, dim * 2])
        mtx[:dim, :dim] = m_ii
        mtx[-dim:, -dim:] = m_ii
        mtx[:dim, -dim:] = m_ij
        mtx[-dim:, :dim] = m_ij.T.conj()
        return mtx
    
    def get_current(self):
        E_Points, weight, fermi_factor = self.get_nepath_info()
        tcalc = self.set_calculator(E_Points)
        tcalc.initialize()
        tcalc.update()
        numE = len(E_Points) 
        current = [0, 0]
        for i in [0,1]:
            for j in range(numE):
                current[i] += tcalc.T_e[j] * weight[j] * fermi_factor[i][0][j]
        self.current = current[0] - current[1]
        return self.current
    
    def plot_eigen_channel(self, energy=[0]):
        tcalc = self.set_calculator(energy)
        tcalc.initialize()
        tcalc.update()
        T_MM = tcalc.T_MM[0]
        from gpaw.utilities.lapack import diagonalize
        nmo = T_MM.shape[-1]
        T = np.zeros([nmo])
        info = diagonalize(T_MM, T)
        dmo = np.empty([nmo, nmo, nmo])
        for i in range(nmo):
            dmo[i] = np.dot(T_MM[i].T.conj(),T_MM[i])
        basis_functions = self.wfs.basis_functions
        for i in range(nmo):
            wt = self.gd.zeros(1)
            basis_functions.construct_density(dmo[i], wt[0], 0)
            import pylab
            wt=np.sum(wt, axis=2) / wt.shape[2] 
            if abs(T[i]) > 0.001:
                pylab.matshow(wt[0])
                pylab.title('T=' + str(T[i]))
                pylab.show()     
                
    def get_nepath_info(self):
        if hasattr(self, 'nepathinfo'):
            energy = self.nepathinfo[0][0].energy
            weight = self.nepathinfo[0][0].weight
            fermi_factor = self.nepathinfo[0][0].fermi_factor
      
        return energy, weight, fermi_factor
    
    def set_buffer(self):
        self.nbmol_inner = self.nbmol 
        if self.use_lead:
            self.nbmol_inner -= np.sum(self.buffer)
        if self.use_env:
            self.nbmol_inner -= np.sum(self.env_buffer)
        ind = np.arange(self.nbmol)
        buffer_ind = []
        lead_ind = []

        for i in range(self.lead_num):
            buffer_ind += list(self.buffer_index[i])
            lead_ind += list(self.lead_index[i])
        for i in range(self.env_num):
            buffer_ind += list(self.env_buffer_index[i])

        ind = np.delete(ind, buffer_ind)
        self.inner_mol_index = ind
        self.gate_mol_index = np.delete(ind, lead_ind)
        
        for i in range(self.lead_num):
             self.inner_lead_index[i] = np.searchsorted(ind,
                                                           self.lead_index[i])
        for i in range(self.env_num):
             self.inner_env_index[i] = np.searchsorted(ind,
                                                            self.env_index[i])

    def integral_diff_weight(self, denocc, denvir, method='transiesta'):
        if method=='transiesta':
            eta = 1e-16
            weight = denocc * denocc.conj() / (denocc * denocc.conj() +
                                               denvir * denvir.conj() + eta)
        return weight

    def pl_write(self, filename, matlist):
        if type(matlist)!= tuple:
            matlist = (matlist,)
            nmat = 1
        else:
            nmat = len(matlist)
        total_matlist = []

        for i in range(nmat):
            if type(matlist[i]) == np.ndarray:
                dim = matlist[i].shape
                if len(dim) == 4:
                    dim = (dim[0],) + (dim[1] * self.pkpt_comm.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] * self.pkpt_comm.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_write')
                if self.pkpt_comm.rank == 0:
                    totalmat = np.empty(dim, dtype=matlist[i].dtype)
                    self.pkpt_comm.gather(matlist[i], 0, totalmat)
                    total_matlist.append(totalmat)
                else:
                    self.pkpt_comm.gather(matlist[i], 0)                    
            else:
                total_matlist.append(matlist[i])
        if world.rank == 0:
            fd = file(filename, 'wb')
            pickle.dump(total_matlist, fd, 2)
            fd.close()
        world.barrier()

    def pl_read(self, filename, collect=False):
        fd = file(filename, 'rb')
        total_matlist = pickle.load(fd)
        fd.close()
        nmat= len(total_matlist)
        matlist = []
        for i in range(nmat):
            if type(total_matlist[i]) == np.ndarray and not collect:
                dim = total_matlist[i].shape
                if len(dim) == 4:
                    dim = (dim[0],) + (dim[1] / self.pkpt_comm.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] / self.pkpt_comm.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_read')
                local_mat = np.empty(dim, dtype=total_matlist[i].dtype)
                self.pkpt_comm.scatter(total_matlist[i], local_mat, 0)
            elif type(total_matlist[i]) == np.ndarray:
                local_mat = np.empty(total_matlist[i].shape,
                                             dtype= total_matlist[i].dtype)
                local_mat = total_matlist[i]
                self.pkpt_comm.broadcast(local_mat, 0)
            else:
                local_mat = np.zeros([1])
                local_mat[0] = total_matlist[i]
                self.pkpt_comm.broadcast(local_mat, 0)
                local_mat = local_mat[0]
            matlist.append(local_mat)
        return matlist

    def fill_lead_with_scat(self):
        for  i in range(self.lead_num):
            ind = self.inner_lead_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            self.hl_spkmm[i] = self.h_spkmm[:, :, ind.T, ind]
            self.sl_pkmm[i] = self.s_pkmm[:, ind.T, ind]
        
        nblead = self.nblead[0]   
        self.hl_spkcmm[0] = self.h_spkmm[:, :, :nblead, nblead:2 * nblead]
        self.sl_pkcmm[0] = self.s_pkmm[:, :nblead, nblead:2 * nblead]
        
        self.hl_spkcmm[1] = self.h_spkmm[:, :, -nblead:, -nblead*2 : -nblead]
        self.sl_pkcmm[1] = self.s_pkmm[:, -nblead:, -nblead*2 : -nblead]

    def get_lead_layer_num(self):
        tol = 1e-4
        temp = []
        for lead_atom in self.atoms_l[0]:
            for i in range(len(temp)):
                if abs(atom.position[self.d] - temp[i]) < tol:
                    break
                temp.append(atom.position[self.d])

    def get_linear_potential_matrix(self):
        # only ok for self.d = 2 now
        nn = 64
        N_c = self.gd.N_c.copy()
        h_c = self.gd.h_c
        N_c[self.d] += nn
        pbc = self.atoms._pbc
        cell = N_c * h_c
        from gpaw.grid_descriptor import GridDescriptor
        comm = self.gd.comm
        GD = GridDescriptor(N_c, cell, pbc, comm)
        basis_functions = self.initialize_projector(extend=True, nn=nn)
        local_linear_potential = GD.empty(self.my_nspins)
        linear_potential = GD.zeros(self.my_nspins, global_array=True)
        dim_s = self.gd.N_c[self.d] #scat
        dim_t = linear_potential.shape[3]#transport direction
        dim_p = linear_potential.shape[1:3] #transverse 
        bias = np.array(self.bias) /Hartree
        vt = np.empty([dim_t])
        vt[:nn / 2] = bias[0] / 2.0
        vt[-nn / 2:] = bias[1] / 2.0
        vt[nn / 2: -nn / 2] = np.linspace(bias[0]/2.0, bias[1]/2.0, dim_s)
        for s in range(self.my_nspins):
            for i in range(dim_t):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dim_p) + 1)
        GD.distribute(linear_potential, local_linear_potential)
        wfs = self.wfs
        nq = len(wfs.ibzk_qc)
        nao = wfs.setups.nao
        H_sqMM = np.empty([self.my_nspins, nq, nao, nao], wfs.dtype)
        H_MM = np.empty([nao, nao], wfs.dtype) 
        for kpt in wfs.kpt_u:
            basis_functions.calculate_potential_matrix(local_linear_potential[0],
                                                       H_MM, kpt.q)
            tri2full(H_MM)
            H_MM *= Hartree
            if self.my_nspins == 2:
                H_sqMM[kpt.s, kpt.q] = H_MM
            else:
                H_sqMM[0, kpt.q] = H_MM
        self.gd.comm.sum(H_sqMM)  
        return H_sqMM

    def estimate_transport_matrix_memory(self):
        self.initialize_transport(dryrun=True, restart=False)
        sum = 0
        ns = self.nspins
        if self.use_lead:
            nk = len(self.my_lead_kpts)
            nb = max(self.nblead)
            npk = self.my_npk
            unit_real = np.array(1,float).itemsize
            unit_complex = np.array(1, complex).itemsize
            if self.npk == 1:
                unit = unit_real
            else:
                unit = unit_complex
            sum += self.lead_num * (2 * ns + 1)* nk * nb**2 * unit_complex
            sum += self.lead_num * (2 * ns + 1)* npk * nb**2 * unit
            
            if self.LR_leads:
                sum += ( 2 * ns + 1) * npk * nb ** 2 * unit
            sum += ns * npk * nb**2 * unit
            print 'lead matrix memery  MB',  sum *1e-6
           
            ntgt = 200
            tmp = self.lead_num * ns * npk * ntgt * nb**2 * unit_complex
            sum += tmp
            print 'selfenergy memery  MB',  tmp *1e-6

        if self.use_env:
            nk = len(self.env_kpts)
            nb = self.nbenv
            sum += self.env_num * (2 * ns + 1) * nk * nb ** 2 * unit_complex
            sum += self.env_num * (2 * ns + 1) * nb ** 2 * unit_real
            
            sum += self.env_num * ns * ntgt * nb**2 * unit_complex
            
        if self.gamma:
            unit = unit_real
        else:
            unit = unit_complex
        nk = len(self.my_kpts)
        nb = self.nbmol
        tmp = 0 
        tmp += (2*ns + 1) * nk * nb**2 * unit

        if self.npk == 1:
            unit = unit_real
        else:
            unit = unit_complex
        tmp += 2 * (2* ns + 1) * npk * nb**2 * unit
        sum += tmp

        print 'scat matrix memery  MB',  tmp *1e-6
        print 'total memery  MB',  sum *1e-6
        raise SystemExit

    def remove_matrix_corner(self):
        if self.atoms._pbc[self.d] and self.ntkmol == 1 and self.nbmol > np.sum(self.nblead):
            nb = max(self.nblead)
            self.h_spkmm[:, :, -nb:, :nb] = 0
            self.s_pkmm[:, -nb:, :nb] = 0
            self.h_spkmm[:, :, :nb, -nb:] = 0
            self.s_pkmm[:, :nb, -nb:] = 0
            
    def add_matrix_corner(self):
        if self.atoms._pbc[self.d] and self.ntkmol == 1 and self.nbmol > np.sum(self.nblead):
            nb = self.nblead[0]
            ns = self.d_spkmm.shape[0]
            npk = self.d_spkmm.shape[1]
            for s in range(ns):
                for k in range(npk):
                    self.d_spkmm[s, k, :nb, -nb:] = self.dl_spkcmm[1][s, k]
                    self.d_spkmm[s, k, -nb:, :nb] = self.dl_spkcmm[1][s, k].T.conj()
            
    def reset_lead_hs(self, s, k):
        if self.use_lead:    
            sg = self.selfenergies
            for i in range(self.lead_num):
                sg[i].h_ii = self.hl_spkmm[i][s, k]
                sg[i].s_ii = self.sl_pkmm[i][k]
                sg[i].h_ij = self.hl_spkcmm[i][s, k]
                sg[i].s_ij = self.sl_pkcmm[i][k]
                sg[i].h_im = self.hl_spkcmm[i][s, k]
                sg[i].s_im = self.sl_pkcmm[i][k]        

    def initialize_projector(self, extend=False, nn=64):
        N_c = self.gd.N_c.copy()
        h_c = self.gd.h_c
        N_c[self.d] += nn
        pbc = self.atoms._pbc
        cell = N_c * h_c
        from gpaw.grid_descriptor import GridDescriptor
        comm = self.gd.comm
        GD = GridDescriptor(N_c, cell, pbc, comm)
        from gpaw.lfc import BasisFunctions
        basis_functions = BasisFunctions(GD,     
                                        [setup.phit_j
                                        for setup in self.wfs.setups],
                                        self.wfs.kpt_comm,
                                        cut=True)
        pos = self.atoms.positions.copy()
        if extend:
            for i in range(len(pos)):
                pos[i, self.d] += nn * h_c[self.d] * Bohr / 2.
        spos_ac = np.linalg.solve(np.diag(cell) * Bohr, pos.T).T % 1.0
        if not self.wfs.gamma:
            basis_functions.set_k_points(self.wfs.ibzk_qc)
        basis_functions.set_positions(spos_ac)
        return basis_functions
    
    def initialize_green_function(self):
        self.selfenergies = []
        if self.use_lead:
            for i in range(self.lead_num):
                self.selfenergies.append(LeadSelfEnergy((self.hl_spkmm[i][0,0],
                                                             self.sl_pkmm[i][0]), 
                                                (self.hl_spkcmm[i][0,0],
                                                             self.sl_pkcmm[i][0]),
                                                (self.hl_spkcmm[i][0,0],
                                                             self.sl_pkcmm[i][0]),
                                                 1e-8))
    
                self.selfenergies[i].set_bias(self.bias[i])
            
        if self.use_env:
            self.env_selfenergies = []
            for i in range(self.env_num):
                self.env_selfenergies.append(CellSelfEnergy((self.he_skmm[i],
                                                             self.se_kmm[i]),
                                                            (self.he_smm[i],
                                                             self.se_mm[i]),
                                                             self.env_ibzk_kc[i],
                                                             self.env_weight[i],
                                                            1e-8))

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_spkmm[0,0],
                                           S=self.s_pkmm[0], eta=0)        

    def calculate_green_function_of_k_point(self, s, k, energy):
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]
        
        self.reset_lead_hs(s, k)
        nbmol = self.nbmol_inner
        sigma = np.zeros([nbmol, nbmol], complex)
        for i in range(self.lead_num):
            ind = self.inner_lead_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            sigma[ind.T, ind] += self.selfenergies[i](energy)
        return self.greenfunction.calculate(energy, sigma)
       
    def calculate_real_dos(self, energy):
        ns = self.my_nspins
        nk = self.my_npk
        nb = self.nbmol_inner
        gr_skmm = np.zeros([ns, nk, nb, nb], complex)
        gr_mm = np.zeros([nb, nb], complex)
        self.initialize_green_function()
        for s in range(ns):
            for k in range(nk):
                gr_mm = self.calculate_green_function_of_k_point(s,
                                                                    k, energy)
                gr_skmm[s, k] =  (gr_mm - gr_mm.conj()) /2.
                print gr_skmm.dtype
                
        self.dos_sg = self.project_from_orbital_to_grid(gr_skmm)

    def plot_real_dos(self, direction=0, mode='average', nl=0):
        import pylab
        dim = self.dos_sg.shape
        print 'diff', np.max(abs(self.dos_sg[0] - self.dos_sg[1]))
        ns, nx, ny, nz = dim
        if mode == 'average':
            for s in range(ns):
                dos_g = np.sum(self.dos_sg[s], axis=direction)
                dos_g /= dim[direction]
                pylab.matshow(dos_g)
                pylab.show()
        elif mode == 'sl': # single layer mode
            for s in range(ns):
                if direction == 0:
                    dos_g = self.dos_sg[s, nl]
                elif direction == 1:
                    dos_g = self.dos_sg[s, :, nl]
                elif direction == 2:
                    dos_g = self.dos_g[s, :, :, nl]
                pylab.matshow(dos_g)
                pylab.show()
    
    def calculate_iv(self, v_limit=3, num_v=16):
        bias = np.linspace(0, v_limit, num_v)
        self.file_num = num_v
        current = np.empty([num_v])
        result = {}
        self.negf_prepare() 
        for i in range(num_v):
            v = bias[i]
            self.bias = [v/2., -v /2.]
            self.get_selfconsistent_hamiltonian()
            result['step_data' + str(i)] = self.result_for_one_bias_step()            
            current[i] = self.get_current()
            #self.output('bias' + str(i))
            result['i_v'] = (bias[:i+1], current[:i+1])    
            result['N'] = i + 1
            if self.master:
                fd = file('result.dat', 'wb')
                pickle.dump(result, fd, 2)
                fd.close()
        if self.fixed:
            del self.surround
 
    def restart_and_abstract_result(self, v_limit=3, num_v=16):
        bias = np.linspace(0, v_limit, num_v)
        current = np.empty([num_v])
        result = {}
        result['N'] = num_v
        for i in range(num_v):
            self.input('bias' + str(i))
            self.hamiltonian.vt_sG += self.get_linear_potential()
            result['step_data' + str(i)] = self.result_for_one_bias_step()            
            current[i] = self.current
        result['i_v'] = (bias, current)    
        if self.master:
            fd = file('result.dat', 'wb')
            pickle.dump(result, fd, 2)
            fd.close()
            
    def result_for_one_bias_step(self):
        step_data = {}
        step_data['bias'] = self.bias
        step_data['v_d'] = self.abstract_d_and_v()
        ef = self.fermi 
        step_data['t_dos'] = self.calculate_dos([ef-4, ef+4])
        return step_data

    def get_boundary_info(self):
        data={}
        data['dmm'] = np.diag(self.d_spkmm[0,0])
        data['hmm'] = np.diag(self.h_spkmm[0,0])
        dim = self.density.nt_sG.shape[1] / 2
        data['den'] = self.density.nt_sG[0, dim, dim]
        data['ham'] = self.hamiltonian.vt_sG[0, dim, dim]
        if hasattr(self, 'test_atomic_hamiltonian_matrix'):
            data['atomic_ham'] = self.test_atomic_hamiltonian_matrix
        if self.fixed:
            data['boundary'] = self.surround.boundary_data
        data['step'] = self.step
        return data
  
    def analysis(self,  flag='bias'):
        if self.master:
            if flag == 'bias':
                fd = file('result.dat', 'r')
                result = pickle.load(fd)
                fd.close()
                num_v = result['N']
                for i in range(num_v):
                    step_data = result['step_data' + str(i)]
                    self.plot_step_data(step_data)
                self.plot_iv(result['i_v'])
            elif flag == 'ele':
                fd = file('ele.dat', 'r')
                result = pickle.load(fd)
                fd.close()
                num_e = len(result)
                for i in range(num_e):
                    step_data = result['step_data' + str(i)]
                    self.plot_boundary_info(step_data)

    def analysis_compare(self):
        if self.master:
            fd = file('result.dat', 'r')
            result = pickle.load(fd)
            fd.close()
            num_v = result['N']
            step_data1 = result['step_data' + str(0)]
            step_data2 = result['step_data' + str(1)]
            self.compare_step_data(step_data1, step_data2)
            
    def compare_step_data(self, step_data1, step_data2):
        overview_d = 1
        self.nspins = 1
        sd = step_data1['t_dos']
        bias = step_data1['bias']
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()
        
        sd = step_data2['t_dos']
        bias = step_data2['bias']
        import pylab
        from mytools import gnu_save
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()        
        
        sd1 = step_data1['v_d']
        sd2 = step_data2['v_d']
        bias1 = step_data1['bias']
        bias2 = step_data2['bias']
        dd = ['x', 'y', 'z']
        for s in range(self.nspins):
            tmp = sd1['s' + str(s) + 'vt_1d_' + dd[self.d]] - sd2['s' + str(s) + 'vt_1d_' + dd[self.d]]
            pylab.plot(tmp * Hartree, 'b--o') 
            pylab.ylabel('potential(eV)')
            pylab.title('spin' + str(s) + 'bias=' + str(bias1) + '-' + str(bias2))
            pylab.show()

            tmp = sd1['s' + str(s) + 'nt_1d_' + dd[self.d]] - sd2['s' + str(s) + 'nt_1d_' + dd[self.d]]        
            pylab.plot(tmp, 'b--o')
            pylab.ylabel('density')
            pylab.title('spin' + str(s) + 'bias=' + str(bias1) + '-' + str(bias2))
            pylab.show()

            tmp = sd1['s' + str(s) + 'vt_2d_' + dd[overview_d]] - sd2['s' + str(s) + 'vt_2d_' + dd[overview_d]]          
            cb = pylab.matshow(tmp * Hartree)
            pylab.title('spin' + str(s) + 'potential(eV) at bias=' + str(bias1) + '-' + str(bias2))
            pylab.colorbar()
            pylab.show()

            tmp = sd1['s' + str(s) + 'nt_2d_' + dd[overview_d]] - sd2['s' + str(s) + 'nt_2d_' + dd[overview_d]]              
            cb = pylab.matshow(tmp)
            gnu_save('diff_s', tmp) 
            pylab.title('spin' + str(s) + 'density at bias=' + str(bias1) + '-' + str(bias2))
            pylab.colorbar()                
            pylab.show()        
       
    def plot_step_data(self, step_data):
        overview_d = 1
        #self.d = 0
        self.nspins = 2
        sd = step_data['t_dos']
        bias = step_data['bias']
        import pylab
        pylab.figure(1)
        pylab.subplot(211)
        pylab.plot(sd['e_points'], sd['T_e'], 'b-o', sd['f1'], sd['l1'],
                                    'r--', sd['f2'], sd['l1'], 'r--')
        pylab.ylabel('Transmission Coefficients')
        pylab.subplot(212)
        pylab.plot(sd['e_points'], sd['dos_e'], 'b-o', sd['f1'], sd['l2'],
                                      'r--', sd['f2'], sd['l2'], 'r--')
        pylab.ylabel('Density of States')
        pylab.xlabel('Energy (eV)')
        pylab.title('bias=' + str(bias))
        pylab.show()
        
        sd = step_data['v_d']
        dd = ['x', 'y', 'z']
        for s in range(self.nspins):
            pylab.plot(sd['s' + str(s) + 'vt_1d_' + dd[self.d]] * Hartree, 'b--o') 
            pylab.ylabel('potential(eV)')
            pylab.title('spin' + str(s) + 'bias=' + str(bias))
            pylab.show()
        
            pylab.plot(sd['s' + str(s) + 'nt_1d_' + dd[self.d]], 'b--o')
            pylab.ylabel('density')
            pylab.title('spin' + str(s) + 'bias=' + str(bias))
            pylab.show()
        
            cb = pylab.matshow(sd['s' + str(s) + 'vt_2d_' + dd[overview_d]] * Hartree)
            pylab.title('spin' + str(s) + 'potential(eV) at bias=' + str(bias))
            pylab.colorbar()               
            pylab.show()
       
            cb = pylab.matshow(sd['s' + str(s) + 'nt_2d_' + dd[overview_d]])
            pylab.title('spin' + str(s) + 'density at bias=' + str(bias))
            pylab.colorbar()
            pylab.show()
        if self.nspins == 2:
            cb = pylab.matshow(sd['s' + str(0) + 'vt_2d_' + dd[overview_d]] * Hartree -
                           sd['s' + str(1) + 'vt_2d_' + dd[overview_d]] * Hartree)
            pylab.title('spin_diff' + 'potential(eV) at bias=' + str(bias))
            pylab.colorbar()                
            pylab.show()
       
            cb = pylab.matshow(sd['s' + str(0) + 'nt_2d_' + dd[overview_d]] -
                               sd['s' + str(1) + 'nt_2d_' + dd[overview_d]])
            pylab.title('spin_diff' + 'density at bias=' + str(bias))
            pylab.colorbar()                
            pylab.show()
 
    def plot_boundary_info(self, step_data):
        import pylab
        bsd = step_data
        pylab.plot(bsd['dmm'], 'b--o')
        pylab.title('dmm')
        pylab.show()
        pylab.plot(bsd['hmm'], 'b--o')
        pylab.title('hmm')        
        pylab.show()
        pylab.plot(bsd['den'], 'b--o')
        pylab.title('den')        
        pylab.show()
        pylab.plot(bsd['ham'], 'b--o')
        pylab.title('ham')        
        pylab.show()
        if 'atomic_ham' in bsd:
            pylab.plot(bsd['atomic_ham'], 'b--o')
            pylab.title('atomic_ham')        
            pylab.show()        
        pylab.plot(bsd['boundary']['nt_sG'], 'b--o')
        pylab.title('boundary_nt_sG')        
        pylab.show()
        pylab.plot(bsd['boundary']['vt_sG'], 'b--o')
        pylab.title('boundary_vt_sG')        
        pylab.show()        
        pylab.plot(bsd['boundary']['vt_sg'], 'b--o')
        pylab.title('boundary_vt_sg')        
        pylab.show()
        pylab.plot(bsd['boundary']['vt_sg1'], 'b--o')
        pylab.title('boundary_vt_sg1')        
        pylab.show()
        pylab.plot(bsd['boundary']['rhot_g'], 'b--o')
        pylab.title('boundary_rhot_g')        
        pylab.show()
        pylab.plot(bsd['boundary']['rhot_g1'], 'b--o')
        pylab.title('boundary_rhot_g1')        
        pylab.show()        
        pylab.plot(bsd['boundary']['vHt_g1'], 'b--o')
        pylab.title('boundary_vHt_g1')        
        pylab.show()          

        pylab.plot(bsd['boundary']['rhot_g2'], 'b--o')
        pylab.title('boundary_rhot_g2')        
        pylab.show()        
        pylab.plot(bsd['boundary']['vHt_g2'], 'b--o')
        pylab.title('boundary_vHt_g2')        
        pylab.show()
        
        
    def plot_iv(self, i_v):
        v, i = i_v
        import pylab
        pylab.plot(v, i, 'b--o')
        pylab.xlabel('bias(V)')
        pylab.ylabel('current(au.)')
        pylab.show()
      
    def recover_kpts(self, calc):
        wfs = calc.wfs
        hamiltonian = calc.hamiltonian
        occupations = calc.occupations
        wfs.eigensolver.iterate(hamiltonian, wfs)
        occupations.calculate(wfs)
     
    def spy(self, var_name, var):
        if not hasattr(self, 'spy_data'):
            self.spy_data = {}
        if var_name not in self.spy_data:
            self.spy_data[var_name] = []
        self.spy_data[var_name].append(var)
        if self.master:
            fd = file('spy_data', 'wb')
            pickle.dump(self.spy_data, fd, 2)
            fd.close()
          
