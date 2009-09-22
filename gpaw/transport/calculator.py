from ase.transport.tools import function_integral, fermidistribution
from ase import Atoms, Atom, Hartree, Bohr
from gpaw import GPAW, debug, dry_run, Mixer, MixerDif, PoissonSolver
from gpaw.poisson import FixedBoundaryPoissonSolver

from gpaw import restart as restart_gpaw
from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.lcao.tools import get_realspace_hs
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.memory import maxrss

from gpaw.transport.tools import tri2full, dot, Se_Sparse_Matrix, PathInfo,\
          get_atom_indices, substract_pk, get_lcao_density_matrix, \
          get_pk_hsd, diag_cell, get_matrix_index, aa1d, aa2d

from gpaw.transport.tools import Tp_Sparse_HSD, Banded_Sparse_HSD, \
                                                                 CP_Sparse_HSD
from gpaw.transport.intctrl import IntCtrl
from gpaw.transport.surrounding import Surrounding, collect_D_asp2, \
                                              collect_D_asp3, distribute_D_asp
from gpaw.transport.selfenergy import LeadSelfEnergy
from gpaw.transport.analysor import Transport_Analysor

import gpaw
import numpy as np
import pickle

class Lead_Calc(GPAW):
    def dry_run(self):
        pass
        
class Transport(GPAW):
    
    def __init__(self, **transport_kwargs):
        self.set_transport_kwargs(**transport_kwargs)
        if self.scat_restart:
            GPAW.__init__(self, self.restart_file + '.gpw')
            self.set_positions()
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
                       
                       'lead_atoms', 'nleadlayers', 'mol_atoms',
                       
                       'use_env', 'env_atoms', 'env_cells', 'env_kpts',
                       'env_use_buffer', 'env_buffer_atoms', 'env_edge_atoms',
                       'env_bias', 'env_pbc', 'env_restart',                     
                       
                       'LR_leads', 'gate',  'cal_loc', 'align',                       
                       'recal_path', 'min_energy',
                       'use_qzk_boundary', 'use_linear_vt_mm',
                       'use_linear_vt_array',
                       'scat_restart', 'save_file', 'restart_file',
                       'non_sc', 'fixed_boundary', 'guess_steps', 'foot_print',
                        'align_har', 'use_fd_poisson']:
                
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
                
            if key in ['lead_atoms']:
                p['lead_atoms'] = kw['lead_atoms']
            if key in ['nleadlayers']:
                p['nleadlayers'] = kw['nleadlayers']
            if key in ['mol_atoms']:
                p['mol_atoms'] = kw['mol_atoms']
                
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
            if key in ['non_sc']:
                p['non_sc'] = kw['non_sc']
            if key in ['foot_print']:
                p['foot_print'] = kw['foot_print']
            if key in ['use_fd_poisson']:
                p['use_fd_poisson'] = kw['use_fd_poisson']

            #----descript the scattering region----     
            if key in ['LR_leads']:         
                p['LR_leads'] = kw['LR_leads']
            if key in ['gate']:
                p['gate'] = kw['gate']
            if key in ['cal_loc']:
                p['cal_loc'] = kw['cal_loc']
            if key in ['recal_path']:
                p['recal_path'] = kw['recal_path']
            if key in ['min_energy']:
                p['min_energy'] = kw['min_energy']
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
            if key in ['guess_steps']:
                p['guess_steps'] = kw['guess_steps']
            if key in ['fixed_boundary']:
                p['fixed_boundary'] = kw['fixed_boundary']
            if key in ['align_har']:
                p['align_har'] = kw['align_har']
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
            
            self.lead_atoms = p['lead_atoms']
            if self.lead_atoms == None:
                self.lead_atoms = self.pl_atoms
            self.nleadlayers = p['nleadlayers']
            self.mol_atoms = p['mol_atoms']
            
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
        self.min_energy = p['min_energy']
        self.use_qzk_boundary = p['use_qzk_boundary']
        self.align =  p['align']
        self.use_linear_vt_mm = p['use_linear_vt_mm']
        self.use_linear_vt_array = p['use_linear_vt_array']        
        self.scat_restart = p['scat_restart']
        self.guess_steps = p['guess_steps']
        self.foot_print = p['foot_print']
        self.save_file = p['save_file']
        self.restart_file = p['restart_file']
        self.fixed = p['fixed_boundary']
        self.non_sc = p['non_sc']
        self.align_har = p['align_har']
        self.use_fd_poisson = p['use_fd_poisson']
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
            raise RuntimeError('wrong way to use keyword LR_leads')
       
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
        p['identical_leads'] = False
        p['pl_atoms'] = []
        p['pl_cells'] = []
        p['pl_kpts'] = []
        p['use_buffer'] = False
        p['buffer_atoms'] = None
        p['edge_atoms'] = None
        p['bias'] = [0, 0]
        p['d'] = 2
        p['lead_restart'] = False

        p['lead_atoms'] = None
        p['nleadlayers'] = [1, 1]

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
        p['use_fd_poisson'] = False
        
        p['LR_leads'] = True
        p['gate'] = 0
        p['cal_loc'] = False
        p['recal_path'] = False
        p['min_energy'] = -100
        p['guess_steps'] = 20
        p['foot_print'] = True
        p['use_qzk_boundary'] = False
        p['align_har'] = 0
        p['align'] = False
        p['use_linear_vt_mm'] = False
        p['use_linear_vt_array'] = False        
        p['scat_restart'] = False
        p['save_file'] = False
        p['restart_file'] = None
        p['fixed_boundary'] = True
        p['non_sc'] = False
        p['spinpol'] = False
        p['verbose'] = False
        return p     

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        
    def initialize_transport(self):
        if self.use_lead:
            if self.LR_leads:
                self.dimt_lead = []
                self.dimt_buffer = []
            self.nblead = []
            self.edge_index = [[None] * self.lead_num, [None] * self.lead_num]

        if self.use_env:
            self.nbenv = []
            self.env_edge_index = [[None] * self.env_num, [None] * self.env_num]

        for i in range(self.lead_num):
            self.atoms_l[i] = self.get_lead_atoms(i)
            calc = self.atoms_l[i].calc
            atoms = self.atoms_l[i]
            if not calc.initialized:
                calc.initialize(atoms)
                if not dry_run:
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
                if not dry_run:
                    calc.set_positions(atoms)
            self.nbenv.append(calc.wfs.setups.nao)
        
        self.initialize()
        self.get_extended_atoms()
        calc = self.extended_atoms.calc
        calc.initialize(self.extended_atoms)
        del calc.density
        self.extended_calc = calc
        self.gd1, self.finegd1 = calc.gd, calc.finegd
        
        self.nspins = self.wfs.nspins
        self.npk = len(self.wfs.ibzk_kc)
        self.my_npk = len(self.wfs.ibzk_qc)
        self.my_nspins = len(self.wfs.kpt_u) / self.my_npk

        self.ntklead = self.pl_kpts[self.d]
 
        bzk_kc = self.wfs.bzk_kc 
        self.gamma = len(bzk_kc) == 1 and not bzk_kc[0].any()
        self.nbmol = self.wfs.setups.nao

        if self.use_lead:
            if self.npk == 1:
                self.lead_kpts = self.atoms_l[0].calc.wfs.bzk_kc
            else:
                self.lead_kpts = self.atoms_l[0].calc.wfs.ibzk_kc                

        if self.use_env:
            self.env_kpts = self.atoms_e[0].calc.wfs.ibzk_kc               
        
        self.initialize_matrix()
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

        for i in range(self.lead_num):
            if self.identical_leads and i > 0:
                self.update_lead_hamiltonian(i, 'lead0')    
            else:
                self.update_lead_hamiltonian(i)

        for i in range(self.env_num):
            self.update_env_hamiltonian(i)
            self.initialize_env(i)

        self.fermi = self.lead_fermi[0]
        world.barrier()
        
        self.timer.start('init surround')
        self.surround = Surrounding(self)  
        self.timer.stop('init surround')
        
        # save memory
        del self.atoms_l
        del self.atoms_e
        
        if not self.non_sc:
            self.timer.start('surround set_position')
            if not self.use_fd_poisson:
                self.inner_poisson = PoissonSolver(nn=self.hamiltonian.poisson.nn)
            else:
                self.inner_poisson = FixedBoundaryPoissonSolver(nn=1)
            self.inner_poisson.set_grid_descriptor(self.finegd)
            self.interpolator = Transformer(self.gd1, self.finegd1,
                                            self.input_parameters.stencils[1],
                                            allocate=False)
            self.interpolator.allocate()
            self.surround.combine()
            self.set_extended_positions()
            self.timer.stop('surround set_position')
        self.get_hamiltonian_initial_guess()
        del self.wfs
        self.wfs = self.extended_calc.wfs

        self.initialized_transport = True
        self.matrix_mode = 'sparse'
        self.plot_option = None
        self.ground = True

    def get_hamiltonian_initial_guess(self):
        atoms = self.atoms.copy()
        atoms.pbc[self.d] = True
        kwargs = self.gpw_kwargs.copy()
        kwargs['poissonsolver'] = PoissonSolver(nn=2)
        kpts = kwargs['kpts']
        kpts = kpts[:2] + (1,)
        kwargs['kpts'] = kpts
        if self.spinpol:
            kwargs['mixer'] = MixerDif(0.1, 5, weight=100.0)
        else:
            kwargs['mixer'] = Mixer(0.1, 5, weight=100.0)
        atoms.set_calculator(gpaw.GPAW(**kwargs))
        calc = atoms.calc
        calc.initialize(atoms)
        calc.set_positions(atoms)
        
        wfs = calc.wfs
        hamiltonian = calc.hamiltonian
        occupations = calc.occupations
        density = calc.density
        scf = calc.scf
        
        if self.non_sc:
            if not self.scat_restart:
                atoms.get_potential_energy()
                if self.save_file:
                    atoms.calc.write('scat.gpw')
                self.hamiltonian = atoms.calc.hamiltonian
                self.density = atoms.calc.density
            else:
                atoms.calc = self
                self.recover_kpts(atoms.calc)                
        else:        
            for iter in range(self.guess_steps):
                wfs.eigensolver.iterate(hamiltonian, wfs)
                occupations.calculate(wfs)
                energy = hamiltonian.get_energy(occupations)
                scf.energies.append(energy)
                scf.check_convergence(density, wfs.eigensolver)
                density.update(wfs)
                density.rhot_g += self.surround.extra_rhot_g
                hamiltonian.update(density)
      
        #atoms.get_potential_energy()
        h_skmm, s_kmm =  self.get_hs(atoms.calc)
        d_skmm = get_lcao_density_matrix(atoms.calc)
        ntk = 1
        kpts = atoms.calc.wfs.ibzk_qc
        h_spkmm = substract_pk(self.d, self.my_npk, ntk, kpts, h_skmm, 'h')
        s_pkmm = substract_pk(self.d, self.my_npk, ntk, kpts, s_kmm)
        d_spkmm = substract_pk(self.d, self.my_npk, ntk, kpts, d_skmm, 'h')
        h00 = self.lead_hsd[0].H[0][0].recover()[0,0]
        h01 = h_spkmm[0,0,0,0]
        s00 = self.lead_hsd[0].S[0].recover()[0,0]
        e_shift = (h00 - h01) / s00
        h_spkmm += e_shift * s_pkmm
        print 'e_shift', e_shift       
        if self.wfs.dtype == float:
            h_spkmm = np.real(h_spkmm).copy()
            s_pkmm = np.real(s_pkmm).copy()
            d_spkmm = np.real(d_spkmm).copy()
        del atoms
        
        for q in range(self.my_npk):
            self.hsd.reset(0, q, s_pkmm[q], 'S', True)
            for s in range(self.my_nspins):
                self.hsd.reset(s, q, h_spkmm[s, q], 'H', True)            
                self.hsd.reset(s, q, d_spkmm[s, q] * ntk, 'D', True)
            
    def fill_guess_with_leads(self, flag=None):
        if self.hsd.S[0].extended:
            n = -2
        else:
            n = -1
        for s in range(self.my_nspins):
            for pk in range(self.my_npk):
                for l in range(self.lead_num):
                    self.hsd.H[s][pk].diag_h[l][n].reset(
                                          self.lead_hsd[l].H[s][pk].recover()
                            + self.bias[l] * self.lead_hsd[l].S[pk].recover())
                    if flag != 'H':
                        self.hsd.D[s][pk].diag_h[l][n].reset(
                                          self.lead_hsd[l].D[s][pk].recover())

    def append_buffer_hsd(self):
        tp_mat = self.hsd.S[0]
        if tp_mat.extended:
            ex_index = [self.lead_index[0] + tp_mat.nb]
            ex_index.append(self.lead_index[1] +
                                       self.nblead[0] + self.nblead[1] + tp_mat.nb - self.lead_index[1][-1] - 1)
            self.hsd.append_lead_as_buffer(self.lead_hsd,
                                           self.lead_couple_hsd, ex_index)                      
                 
    def get_basis_indices(self):
        setups = self.wfs.setups
        
        edge_index = []
        lead_edge_index = []
        
        for i in range(self.lead_num):
            lead_setups = self.atoms_l[i].calc.wfs.setups
            self.lead_index[i] = get_atom_indices(self.pl_atoms[i], setups)
            edge_index.append(get_atom_indices([self.edge_atoms[1][i]], setups))
            lead_edge_index.append(get_atom_indices([self.edge_atoms[0][i]],
                                                                 lead_setups))            
            self.edge_index[0][i] = lead_edge_index[i][0]
            self.edge_index[1][i] = edge_index[i][0]
        
        edge_index = []
        env_edge_index = []
        
        for i in range(self.env_num):
            env_setups = self.atoms_e[i].calc.wfs.setups
            self.env_index[i] = get_atom_indices(self.env_atoms[i], setups)
            edge_index.append(get_atom_indices([self.env_edge_atoms[1][i]],
                                                                     setups))           
            env_edge_index.append(get_atom_indices([self.env_edge_atoms[0][i]],
                                                                  env_setups))
            self.env_edge_index[0][i] = env_edge_index[i][0]
            self.env_edge_index[1][i] = edge_index[i][0]
      
        
        if not self.use_buffer:
            self.dimt_buffer = [0] * self.lead_num
        else:
            for i in range(self.lead_num):
                self.buffer_index[i] = get_atoms_indices(self.buffer_atoms[i],
                                                         setups)
            for i in range(self.env_num):
                self.env_buffer_index[i] = get_atoms_indices(
                                             self.env_buffer_atoms[i], setups)
        
        for i in range(self.lead_num):
            n_layer_atoms = len(self.lead_atoms[i]) / self.nleadlayers[i]
            self.lead_layer_index[i][0] = get_atom_indices(self.mol_atoms, setups)
            begin = 0
            for j in range(1, self.nleadlayers[i] + 1):
                atoms_index = self.lead_atoms[i][begin: begin + n_layer_atoms]
                self.lead_layer_index[i][j] = get_atom_indices(atoms_index, setups)
                begin += n_layer_atoms
      
    def initialize_matrix(self):
        if self.use_lead:
            self.lead_hsd = []
            self.lead_couple_hsd = []
            self.lead_index = []
            self.inner_lead_index = []
            self.buffer_index = []
            self.lead_layer_index = []
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
        ns = self.my_nspins
        dtype = self.wfs.dtype
       
        for i in range(self.lead_num):
            nb = self.nblead[i]
            self.lead_hsd.append(Banded_Sparse_HSD(dtype, ns, npk))
            self.lead_couple_hsd.append(CP_Sparse_HSD(dtype, ns, npk))
            self.lead_index.append([])
            self.inner_lead_index.append([])
            self.buffer_index.append([])
            self.lead_layer_index.append([])
            for j in range(self.nleadlayers[i] + 1):
                self.lead_layer_index[i].append([])
      
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
        self.get_basis_indices()
        
        extended = True
        self.hsd = Tp_Sparse_HSD(dtype, self.my_nspins, self.my_npk,
                                              self.lead_layer_index, extended)              

    def distribute_energy_points(self):
        self.energy_comm = self.gd.comm
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
        self.timer.start('update lead hamiltonian' + str(l))
        
        if not self.lead_restart and restart_file==None:
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            if self.save_file:
                atoms.calc.write('lead' + str(l) + '.gpw', db=True,
                                keywords=['transport', 'electrode', 'lcao']) 
        else:
            if restart_file == None:
                restart_file = 'lead' + str(l)
            atoms, calc = restart_gpaw(restart_file +'.gpw')
            calc.set_positions()
            self.recover_kpts(calc)
            self.atoms_l[l] = atoms
            
        hl_skmm, sl_kmm = self.get_hs(atoms.calc)
        self.lead_fermi[l] = atoms.calc.get_fermi_level()
        dl_skmm = get_lcao_density_matrix(atoms.calc)
            
        hl_spkmm, sl_pkmm, dl_spkmm,  \
        hl_spkcmm, sl_pkcmm, dl_spkcmm = get_pk_hsd(self.d, self.ntklead,
                                                atoms.calc.wfs.ibzk_qc,
                                                hl_skmm, sl_kmm, dl_skmm,
                                                self.text, self.wfs.dtype,
                                                direction=l)
        
        self.timer.stop('update lead hamiltonian' + str(l))
        
        self.timer.start('init lead' + str(l))
        for pk in range(self.my_npk):
            self.lead_hsd[l].reset(0, pk, sl_pkmm[pk], 'S', init=True)
            self.lead_couple_hsd[l].reset(0, pk, sl_pkcmm[pk], 'S',
                                                              init=True)
            for s in range(self.my_nspins):
                self.lead_hsd[l].reset(s, pk, hl_spkmm[s, pk], 'H', init=True)     
                self.lead_hsd[l].reset(s, pk, dl_spkmm[s, pk], 'D', init=True)
                
                self.lead_couple_hsd[l].reset(s, pk, hl_spkcmm[s, pk],
                                                               'H', init=True)     
                self.lead_couple_hsd[l].reset(s, pk, dl_spkcmm[s, pk],
                                                               'D', init=True)                    
        self.timer.stop('init lead' + str(l))

    def update_env_hamiltonian(self, l):
        if not self.env_restart:
            atoms = self.atoms_e[l]
            atoms.get_potential_energy()
            self.he_skmm[l], self.se_kmm[l] = self.get_hs(atoms.calc)
            self.env_fermi[l] = atoms.calc.get_fermi_level()
            self.de_skmm[l] = get_density_matrix(atoms.calc)
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
        if atoms is None:
            atoms = self.atoms
        else:
            self.atoms = atoms.copy()
            self.initialize()
            self.atoms_l = []
            for i in range(self.lead_num):
                self.atoms_l.append(self.get_lead_atoms[i])
            self.get_extended_atoms()
            calc = self.extended_atoms.calc
            calc.initialize(self.extended_atoms)
            del calc.density
            self.extended_calc = calc
            self.gd1, self.finegd1 = calc.gd, calc.finegd
            self.set_extended_positions()                
                
        #if self.scat_restart:
        #    self.recover_kpts(self)
       
        self.append_buffer_hsd()
        #self.fill_guess_with_leads()           
        self.scat_restart = False

    def get_hs(self, calc):
        wfs = calc.wfs
        eigensolver = wfs.eigensolver
        ham = calc.hamiltonian
        S_qMM = wfs.S_qMM.copy()
        for S_MM in S_qMM:
            tri2full(S_MM)
        H_sqMM = np.empty((self.my_nspins,) + S_qMM.shape, wfs.dtype)
        for kpt in wfs.kpt_u:
            eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
            H_MM = eigensolver.H_MM
            tri2full(H_MM)
            H_MM *= Hartree
            if self.my_nspins == 2:
                H_sqMM[kpt.s, kpt.q] = H_MM
            else:
                H_sqMM[0, kpt.q] = H_MM
        return H_sqMM, S_qMM
       
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
        p['kpts'] = self.pl_kpts
        if 'mixer' in p:
            if not self.spinpol:
                p['mixer'] = Mixer(0.1, 5, weight=100.0)
            else:
                p['mixer'] = MixerDif(0.1, 5, weight=100.0)
        p['poissonsolver'] = PoissonSolver(nn=2)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return Lead_Calc(**p)
    
    def get_env_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        #p['usesymm'] = True
        p['kpts'] = self.env_kpts
        if 'mixer' in p:
            p['mixer'] = Mixer(0.1, 5, weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'env%i_' % (l + 1) + p['txt']
        return Lead_Calc(**p)

    def negf_prepare(self, atoms=None):
        if not self.initialized_transport:
            self.initialize_transport()
        self.update_scat_hamiltonian(atoms)
        self.boundary_check()
    
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

    def boundary_check(self):
        tol = 5.e-4
           
    def get_selfconsistent_hamiltonian(self):
        self.timer.start('init scf')
        self.initialize_scf()
        self.timer.stop('init scf')
        
        ##temperary lines
        self.hamiltonian.S = 0
        self.hamiltonian.Etot = 0
        ##temp
        
        while not self.cvgflag and self.step < self.max_steps:
            self.iterate()
            self.cvgflag = self.d_cvg and self.h_cvg
            self.step +=  1
        
        if self.foot_print:
            self.analysor.save_bias_step()
            self.analysor.save_data_to_file()            
         
        self.scf.converged = self.cvgflag
        
        ## these temperary lines is for storage the transport object
        for kpt in self.wfs.kpt_u:
            kpt.rho_MM = None
            kpt.eps_n = np.zeros((self.nbmol))
            kpt.f_n = np.zeros((self.nbmol))
        ##
        self.ground = False
        self.linear_mm = None
        if not self.scf.converged:
            raise RuntimeError('Transport do not converge in %d steps' %
                                                              self.max_steps)
        
    def non_sc_analysis(self):
        if not hasattr(self, 'analysor'):
            self.analysor = Transport_Analysor(self, True)
        self.analysor.save_bias_step()
        self.analysor.save_data_to_file()
  
    def get_hamiltonian_matrix(self):
        self.timer.start('HamMM')            
        self.den2fock()
        self.timer.stop('HamMM')
        if self.master:
            self.text('HamMM', self.timer.gettime('HamMM'), 'second')        
  
    def get_density_matrix(self):
        self.timer.start('DenMM')
        if self.use_qzk_boundary:
            self.fill_lead_with_scat()
            for i in range(self.lead_num):
                self.selfenergies[i].set_bias(0)
        
        if self.recal_path:
            self.initialize_path()
            
        for s in range(self.my_nspins):
            for k in range(self.my_npk):
                if self.recal_path:
                    d_mm = self.get_eqintegral_points(s, k) + \
                                              self.get_neintegral_points(s, k)
                else:
                    d_mm = self.fock2den(s, k)
                d_mm = self.spin_coff * (d_mm + d_mm.T.conj()) / (2 * self.npk)
                self.hsd.reset(s, k, d_mm, 'D') 
        self.timer.stop('DenMM')
        self.print_boundary_charge()
        if self.master:
            self.text('DenMM', self.timer.gettime('DenMM'), 'second')

    def iterate(self):
        if self.master:
            self.text('----------------step %d -------------------'
                                                                % self.step)
        self.h_cvg = self.check_convergence('h')
        self.get_density_matrix()
        self.get_hamiltonian_matrix()
        self.d_cvg = self.check_convergence('d')
        self.txt.flush()
        
    def check_convergence(self, var):
        cvg = False
        if var == 'h':
            diag_ham = np.zeros([self.nbmol], self.wfs.dtype)
            for s in range(self.my_nspins):
                for q in range(self.my_npk):
                    diag_ham += np.diag(self.hsd.H[s][q].recover())
            self.wfs.kpt_comm.sum(diag_ham)
            diag_ham /= self.npk
            
            self.diff_h = 1.         
            if self.step > 0:
                self.diff_h = np.max(abs(diag_ham - self.diag_ham_old))
                if self.master:
                    self.text('hamiltonian: diff = %f  tol=%f' % (self.diff_h,
                                                  self.diag_ham_tol))
                if self.diff_h < self.diag_ham_tol:
                    cvg = True
            self.diag_ham_old = np.copy(diag_ham)
        if var == 'd':
            if self.step > 0:
                self.diff_d = self.density.mixer.get_charge_sloshing()
                if self.ground:
                    tol =  self.scf.max_density_error
                else:
                    tol =  self.scf.max_density_error
 
                if self.master:
                    self.text('density: diff = %f  tol=%f' % (self.diff_d,
                                            tol))
                if self.ground:
                    if self.diff_d < tol:
                      cvg = True
                else:
                    if self.diff_d < tol:
                      cvg = True                    
        return cvg
 
    def initialize_scf(self):
        self.intctrl = IntCtrl(self.occupations.kT * Hartree,
                                self.lead_fermi, self.bias,
                                self.env_bias, self.min_energy)            
        self.surround.reset_bias(self.bias) 
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

        if not hasattr(self, 'analysor'):
            self.analysor = Transport_Analysor(self)
        
        if not self.fixed:
            self.get_linear_hartree_potential()
        #------for check convergence------
        #self.ham_vt_old = np.empty(self.hamiltonian.vt_sG.shape)
        self.ham_vt_diff = None
        self.ham_vt_tol = 1e-2
        self.diag_ham_tol = 5e-3
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
            for k in range(self.my_npk):
                self.eqpathinfo[s].append(PathInfo('eq', self.lead_num + self.env_num))
                self.nepathinfo[s].append(PathInfo('ne', self.lead_num + self.env_num))    
                if self.cal_loc:
                    self.locpathinfo[s].append(PathInfo('eq',
                                                         self.lead_num + self.env_num))
                    
    def calculate_integral_path(self):
        self.initialize_path()
        for s in range(self.my_nspins):
            for k in range(self.my_npk):      
                self.get_eqintegral_points(s, k)
                self.get_neintegral_points(s, k)
                if self.cal_loc:
                    self.get_neintegral_points(s, k, 'locInt')
        ne = self.eqpathinfo[0][0].num + self.nepathinfo[0][0].num
        self.text('energy point' + str(ne))           
        
    def get_eqintegral_points(self, s, k):
        if self.recal_path:
            self.timer.start('eq fock2den')
        maxintcnt = 50
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], self.wfs.dtype)
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
   
        self.hsd.s = s
        self.hsd.pk = k
        
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
        
        if self.recal_path:
            self.timer.stop('eq fock2den')
            
        del self.fint, self.tgtint, self.zint
        return den 
    
    def get_neintegral_points(self, s, k, calcutype='neInt'):
        if self.recal_path:
            self.timer.start('ne fock2den')        
        intpathtol = 1e-8
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], self.wfs.dtype)
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
                
        self.hsd.s = s
        self.hsd.pk = k

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

        if self.recal_path:
            self.timer.stop('ne fock2den')
            
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
            sigma = []
            if self.use_env:
                env_gamma = np.zeros([self.env_num, nbmol, nbmol], complex)
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt

            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in range(self.lead_num):
                tgt = self.selfenergies[j](zp[i])
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
                tgt = self.tgtint[j][self.cntint]
                sigma.append(tgt)
            
            if self.use_env:
                nl = self.lead_num
                for j in range(self.env_num):
                    ind = self.inner_env_index[j]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += self.tgtint[j + nl][self.cntint]
                    env_gamma[j, ind.T, ind] += \
                                    self.env_selfenergies[j].get_lambda(zp[i])

            gr = self.hsd.calculate_eq_green_function(zp[i], sigma, False)
            # --ne-Integral---
            kt = intctrl.kt
            ff = []
            if calcutype == 'neInt':
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
                    ff.append(self.fint[n][0][self.cntint])
                    
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

                gfunc[i] = self.hsd.calculate_ne_green_function(zp[i],
                                                                sigma, ff, False)
                

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
        
        self.hsd.s = s
        self.hsd.pk = k

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
        if self.wfs.dtype == float:
            den = np.real(den).copy()
        return den    

    def ne_fock2den(self, s, k, ov='occ'):
        pathinfo = self.nepathinfo[s][k]
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        begin = self.par_energy_index[s, k, 1, 0]
        end = self.par_energy_index[s, k, 1, 1]        
        zp = pathinfo.energy

        self.timer.start('ne fock2den')
        for i in range(begin, end):
            sigma = []
            for n in range(self.lead_num):
                sigma.append(pathinfo.sigma[n][i])
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]

            ff = []
            for n in range(self.lead_num):
                if ov == 'occ':
                    fermifactor = np.real(pathinfo.fermi_factor[n][0][i])
                    ff.append(fermifactor)
                elif ov == 'vir':
                    fermifactor = np.real(pathinfo.fermi_factor[n][1][i])
                    ff.append(fermifactor)                    
           
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma = pathinfo[n + nl][i]
                    if ov == 'occ':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][0][i])
                    elif ov == 'vir':
                        fermifactor = np.real(
                                         pathinfo.fermi_factor[n + nl][1][i])
                    sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                             sigma - sigma.T.conj())
            
            glesser = self.hsd.calculate_ne_green_function(zp[i], sigma,
                                                           ff, False)
            weight = pathinfo.weight[i]            
            den += glesser * weight / np.pi / 2
        self.energy_comm.sum(den)
        self.timer.stop('ne fock2den')
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
        self.timer.start('eq fock2den')
        for i in range(begin, end):
            sigma = []
            for n in range(self.lead_num):
                sigma.append(pathinfo.sigma[n][i])
            if self.use_env:
                nl = self.lead_num
                for n in range(self.env_num):
                    ind = self.inner_env_index[n]
                    dim = len(ind)
                    ind = np.resize(ind, [dim, dim])
                    sigma[ind.T, ind] += pathinfo.sigma[n + nl][i]
            
            gr = self.hsd.calculate_eq_green_function(zp[i], sigma, False)
            fermifactor = pathinfo.fermi_factor[i]
            weight = pathinfo.weight[i]
            den += gr * fermifactor * weight
        self.energy_comm.sum(den)
        den = 1.j * (den - den.T.conj()) / np.pi / 2
        self.timer.stop('eq fock2den')
        return den

    def den2fock(self):
        self.update_density()
        self.update_hamiltonian()
        
        if self.use_linear_vt_array:
            self.hamiltonian.vt_sG += self.get_linear_potential()
        
        self.timer.start('record')        
        if self.foot_print:
            self.analysor.save_ele_step()
            self.analysor.save_data_to_file('ele')
        self.timer.stop('record')
        self.record_time_cost = self.timer.gettime('record')
        
        self.timer.start('project hamiltonian')
        h_spkmm, s_pkmm = self.get_hs(self.extended_calc)

        self.timer.stop('project hamiltonian')                  
       
        for q in range(self.my_npk):
            for s in range(self.my_nspins):
                self.hsd.reset(s, q, h_spkmm[s, q], 'H')
  
    def get_forces(self, atoms):
        if (atoms.positions != self.atoms.positions).any():
            self.scf.converged = False
        if hasattr(self.scf, 'converged') and self.scf.converged:
            pass
        else:
            self.negf_prepare(atoms)
            self.get_selfconsistent_hamiltonian()
            self.analysor.save_ion_step()
            self.analysor.save_data_to_file()
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
       
    def update_density(self):
        self.timer.start('dmm recover')
        #self.fill_guess_with_leads()
        for kpt in self.wfs.kpt_u:
            if self.my_nspins == 2:
                kpt.rho_MM = self.hsd.D[kpt.s][kpt.q].recover(True)
            else:
                kpt.rho_MM = self.hsd.D[0][kpt.q].recover(True)
        self.timer.stop('dmm recover')        
        
        density = self.density
        self.timer.start('construct density')

        nt_sG = self.gd1.zeros(self.nspins)
        self.wfs.calculate_density_contribution(nt_sG)
        nn = self.surround.nn[0]
        density.nt_sG = self.surround.uncapsule(nn, nt_sG, self.gd1,
                                                    self.gd)
        density.nt_sG += density.nct_G

        self.timer.stop('construct density')
        self.timer.start('atomic density')

        D_asp = self.extended_D_asp
        self.wfs.calculate_atomic_density_matrices(D_asp)
        all_D_asp = collect_D_asp2(D_asp, self.wfs.setups, self.nspins,
                            self.gd.comm, self.extended_calc.wfs.rank_a)
            
        D_asp = all_D_asp[:len(self.atoms)]
        distribute_D_asp(D_asp, density)

        self.timer.stop('atomic density')
        comp_charge = density.calculate_multipole_moments()

        density.normalize(comp_charge)

        density.mix(comp_charge)
        
        #if not density.mixer.mix_rho:
        #    density.mixer.mix(density)
            
        #if density.nt_sg is None:
        #    density.nt_sg = density.finegd.empty(density.nspins)

        #nt_sG = self.surround.combine_nt_sG(density.nt_sG)
        #nt_sg = self.finegd1.zeros(self.nspins)
        #for s in range(density.nspins):
        #    self.interpolator.apply(nt_sG[s], nt_sg[s])
        
        #nn = self.surround.nn[0] * 2
        #density.nt_sg = self.surround.uncapsule(nn, nt_sg, self.finegd1,
        #                                            self.finegd)
        #pseudo_charge = -(density.charge + comp_charge)
        #if abs(pseudo_charge) > 1.0e-14:
        #    x = pseudo_charge / self.finegd.integrate(density.nt_sg).sum()
        #    density.nt_sg *= x
        #density.calculate_pseudo_charge(comp_charge)

        #if density.mixer.mix_rho:
        #    density.mixer.mix(density)        
            
    def update_hamiltonian(self):
        # only used by fixed bc
        
        ham = self.extended_calc.hamiltonian
        density = self.density
        self.timer.start('Hamiltonian')
        #if ham.vt_sg is None:
        #    ham.vt_sg = ham.finegd.empty(ham.nspins)
        #    ham.vHt_g = ham.finegd.zeros()
        #    ham.vt_sG = ham.gd.empty(ham.nspins)
        #    self.inner_poisson.initialize()
 
        nn = self.surround.nn[0] * 2
        nt_sg = self.surround.capsule(nn, density.nt_sg, self.surround.nt_sg,
                                                    self.finegd1, self.finegd)
  
        Ebar = ham.finegd.integrate(ham.vbar_g, np.sum(nt_sg, axis=0),
                                     global_integral=False) 
        vt_g = ham.vt_sg[0]
        vt_g[:] = ham.vbar_g
        Eext = 0.0

        if ham.vext_g is not None:
            vt_g += ham.vext_g.get_potential(ham.finegd)
            Eext = np.vdot(vt_g, np.sum(nt_sg, axis=0)) * ham.finegd.dv - Ebar

        if ham.nspins == 2:
            ham.vt_sg[1] = vt_g
       
        if ham.nspins == 2:
            Exc = ham.xc.get_energy_and_potential(
                 nt_sg[0], ham.vt_sg[0],
                 nt_sg[1], ham.vt_sg[1])
        else:
            Exc = ham.xc.get_energy_and_potential(
                 nt_sg[0], ham.vt_sg[0])

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        if self.fixed and not self.use_fd_poisson:
            density.rhot_g -= self.surround.extra_rhot_g
        if self.hamiltonian.vHt_g is None:
            self.hamiltonian.vHt_g = self.finegd.zeros()

        actual_charge = self.finegd.integrate(density.rhot_g)
        self.text('actual_charge' + str(actual_charge))
        if self.fixed and self.use_fd_poisson:
            ham.npoisson = self.inner_poisson.solve(self.hamiltonian.vHt_g,
                                                       density.rhot_g)
        
        else:
            ham.npoisson = self.inner_poisson.solve(self.hamiltonian.vHt_g,
                                                  density.rhot_g,
                                                  charge=-density.charge)

        if not self.fixed and hasattr(self, 'step'):
            self.hamiltonian.vHt_g += self.linear_vHt_g
        
        vHt_g = self.extended_calc.finegd.collect(ham.vHt_g, True)    
        vHt_g0 = self.finegd.collect(self.hamiltonian.vHt_g, True)

        if hasattr(self, 'step') and self.ground:
            if self.align_har == -1:
                ham_diff = np.sum(vHt_g[:,:,0]) - np.sum(vHt_g0[:,:,0])
            elif self.align_har == 1:
                ham_diff = np.sum(vHt_g[:,:,1]) - np.sum(vHt_g0[:,:,1])
            else:
                ham_diff = np.sum(vHt_g[:,:,0]) - np.sum(vHt_g0[:,:,0])            
                ham_diff += np.sum(vHt_g[:,:,1]) - np.sum(vHt_g0[:,:,1])
                ham_diff /= 2
            ham_diff /= np.product(vHt_g.shape[:2])            
            self.ham_diff = ham_diff
            self.text('Hartree_diff', str(ham_diff))
        
        if hasattr(self, 'step') and self.atoms.pbc.all() and not self.use_fd_poisson:    
            self.hamiltonian.vHt_g += self.ham_diff

        self.surround.combine_vHt_g(self.hamiltonian.vHt_g)
           
        self.text('poisson interations :' + str(ham.npoisson))
        self.timer.stop('Poisson')
      
      
        Epot = 0.5 * self.hamiltonian.finegd.integrate(self.hamiltonian.vHt_g,
                                                       density.rhot_g,
                                                        global_integral=False)
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(ham.vt_sg, ham.vt_sG, density.nt_sG):
            vt_g += ham.vHt_g
            ham.restrict(vt_g, vt_G)
            #Ekin -= ham.gd.integrate(vt_G, nt_G - density.nct_G,
            #                                           global_integral=False)            
        
    
        self.surround.refresh_vt_sG()
        self.timer.start('atomic hamiltonian')
        
        ham = self.hamiltonian
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

            ham.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, dH_sp, a)
            dH_sp += dH_p

            Ekin -= (D_sp * dH_sp).sum()

        self.timer.stop('atomic hamiltonian')

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

        dH_asp = collect_D_asp3(ham, self.wfs.rank_a)
        self.surround.combine_dH_asp(dH_asp)
        self.timer.stop('Hamiltonian')      

    
    def print_boundary_charge(self):
        nb = self.nblead[0]
        qr_mm = np.zeros([self.lead_num, nb, nb])
        boundary_charge = []
        print_info = ''
        if self.hsd.S[0].extended:
            n = -2
        else:
            n = -1
        for i in range(self.lead_num):
            for s in range(self.my_nspins):
                for pk in range(self.my_npk):
                    D = self.hsd.D[s][pk]
                    S = self.hsd.S[pk]
                    qr_mm[i] += dot(D.diag_h[i][n].recover(),
                                                     S.diag_h[i][n].recover())
                    qr_mm[i] += dot(D.dwnc_h[i][n], S.upc_h[i][n])
                    if S.extended:
                        qr_mm[i] += dot(D.dwnc_h[i][n + 1], S.upc_h[i][n + 1])
                    else:
                        qr_mm[i] += dot(D.dwnc_h[i][n], S.upc_h[i][n])                        
            self.wfs.kpt_comm.sum(qr_mm[i])
            boundary_charge.append(np.real(np.trace(qr_mm[i])))
            if i != 0:
                print_info += '******'
            print_info += str(boundary_charge[i])
        self.text(print_info)

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

    def fill_lead_with_scat(self):
        for  i in range(self.lead_num):
            ind = self.inner_lead_index[i]
            dim = len(ind)
            ind = np.resize(ind, [dim, dim])
            self.hl_spkmm[i] = self.h_skmm[:, :, ind.T, ind]
            self.sl_pkmm[i] = self.s_kmm[:, ind.T, ind]
        
        nblead = self.nblead[0]   
        self.hl_spkcmm[0] = self.h_skmm[:, :, :nblead, nblead:2 * nblead]
        self.sl_pkcmm[0] = self.s_kmm[:, :nblead, nblead:2 * nblead]
        
        self.hl_spkcmm[1] = self.h_skmm[:, :, -nblead:, -nblead*2 : -nblead]
        self.sl_pkcmm[1] = self.s_kmm[:, -nblead:, -nblead*2 : -nblead]

    def estimate_transport_matrix_memory(self):
        sum = 0
        ns = self.wfs.nspins
        if self.use_lead:
            nk = len(self.atoms_l[0].calc.wfs.ibzk_qc)
            nb = max(self.nblead)
            npk = len(self.wfs.ibzk_qc)
            unit_real = np.array(1,float).itemsize
            unit_complex = np.array(1, complex).itemsize
            
            gamma = len(self.wfs.bzk_kc) == 1 and not self.wfs.bzk_kc[0].any()          
            if gamma:
                unit = unit_real
            else:
                unit = unit_complex
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
            nk = len(self.atoms_e[0].calc.ibzk_qc)
            nb = self.nbenv
            sum += self.env_num * (2 * ns + 1) * nk * nb ** 2 * unit_complex
            sum += self.env_num * (2 * ns + 1) * nb ** 2 * unit_real
            
            sum += self.env_num * ns * ntgt * nb**2 * unit_complex
            
        if gamma:
            unit = unit_real
        else:
            unit = unit_complex
        nk = len(self.wfs.ibzk_qc)
        nb = self.wfs.setups.nao
        sum += (2*ns + 1) * nk * nb**2 * unit
        return tmp, (sum - tmp)
           
    def reset_lead_hs(self, s, k):
        if self.use_lead:    
            sg = self.selfenergies
            for i in range(self.lead_num):
                sg[i].s = s
                sg[i].pk = k

    def initialize_green_function(self):
        self.selfenergies = []
        if self.use_lead:
            for i in range(self.lead_num):
                self.selfenergies.append(LeadSelfEnergy(self.lead_hsd[i],
                                                      self.lead_couple_hsd[i]))
    
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
        del self.analysor
        del self.surround
 
    def recover_kpts(self, calc):
        wfs = calc.wfs
        wfs.eigensolver.iterate(calc.hamiltonian, wfs)
        calc.occupations.calculate(wfs)

    def estimate_memory(self, mem):
        """Estimate memory use of this object."""
  
        mem_init = maxrss() # XXX initial overhead includes part of Hamiltonian
        mem.subnode('Initial overhead', mem_init)
        for name, obj in [('Density', self.density),
                          ('Hamiltonian', self.hamiltonian),
                          ('Wavefunctions', self.wfs)]:
            obj.estimate_memory(mem.subnode(name))
        for i, atoms in enumerate(self.atoms_l):
            atoms.calc.estimate_memory(mem.subnode('Leads' + str(i), 0))
        se_mem, mat_mem = self.estimate_transport_matrix_memory()
        
        mem.subnode('Matrix', mat_mem)
        mem.subnode('Selfenergy', se_mem)

    def get_extended_atoms(self):
        # for LR leads only
        atoms = self.atoms.copy()
        cell = diag_cell(atoms.cell)
        ex_cell = cell.copy()
        di = 2
        for i in range(self.lead_num):
            atoms_l = self.atoms_l[i].copy()
            cell_l = diag_cell(atoms_l.cell)
            ex_cell[di] += cell_l[di]
            for atom in atoms_l:
                if i == 0:
                    atom.position[di] -= cell_l[di]
                else:
                    atom.position[di] += cell[di]
            atoms += atoms_l
        atoms.set_cell(ex_cell)
        atoms.set_pbc(self.atoms._pbc)
        self.extended_atoms = atoms
        self.extended_atoms.center()
        p = self.gpw_kwargs.copy()
        p['h'] = None
        N_c = self.gd.N_c.copy()
        for i in range(self.lead_num):
            N_c[2] += self.atoms_l[i].calc.gd.N_c[2]
        p['gpts'] = N_c
        if 'mixer' in p:
            if not self.spinpol:
                p['mixer'] = Mixer(0.1, 5, weight=100.0)
            else:
                p['mixer'] = MixerDif(0.1, 5, weight=100.0)
        p['poissonsolver'] = PoissonSolver(nn=2)        
        self.extended_atoms.set_calculator(Lead_Calc(**p))

    def get_linear_hartree_potential(self):
        global_linear_vHt = self.finegd.zeros(global_array=True)
        dim = self.finegd.N_c[2]
        vt = np.linspace(self.bias[0]/Hartree, self.bias[1]/Hartree, dim)
        for i in range(dim):
            global_linear_vHt[:, :, i] = vt[i]
        self.linear_vHt_g = self.finegd.zeros()
        self.finegd.distribute(global_linear_vHt, self.linear_vHt_g)


    def find_pot_first_min_and_max(self, vHt_g):
        vHt = aa1d(vHt_g)
        nn = self.surround.nn[0] * 2 
        g_vHt = np.gradient(vHt[:nn], self.finegd.h_c[2])
        minf = False
        maxf = False
        for i, tmp in enumerate(g_vHt):
            if tmp >= 0 and i > 3 and not minf:
                lmin = i
                minf = True
            if minf and not maxf and tmp <= 0:
                lmax = i
                maxf = True
        lmin = vHt[lmin]
        lmax = vHt[lmax] 

        g_vHt = np.gradient(vHt[-nn:], self.finegd.h_c[2])
        minf = False
        maxf = False
        for i in range(nn):
            tmp = g_vHt[-i - 1]
            if tmp <= 0 and i > 3 and not minf:
                rmin = i
                minf = True
            if minf and not maxf and tmp >= 0:
                rmax = i
                maxf = True
        rmin = vHt[-rmin - 1]
        rmax = vHt[-rmax - 1]
        return lmin, lmax, rmin, rmax

    def get_linear_hartree_potential2(self, vHt_g, vHt_g0):
        global_linear_vHt = self.finegd.zeros(global_array=True)
        dim = self.finegd.N_c[2]
        
        vHt = aa1d(vHt_g)
        vHt0 = aa1d(vHt_g0)
        
        lshift = vHt[0] - vHt0[0]
        rshift = vHt[-1] - vHt0[-1]
        #lmin, lmax, rmin, rmax = self.find_pot_first_min_and_max(vHt_g)
        #lmin0, lmax0, rmin0, rmax0 = self.find_pot_first_min_and_max(vHt_g0)
        
        #print lmin, lmax, rmin, rmax, 'vHt'
        #print lmin0, lmax0, rmin0, rmax0, 'vHt0'

        #lshift = (lmin - lmin0 + lmax - lmax0) / 2
        #rshift = (rmin - rmin0 + rmax - rmax0) / 2
        self.text('hartree potential shift', str(lshift), str(rshift))
        vt = np.linspace(lshift, rshift, dim)
        for i in range(dim):
            global_linear_vHt[:, :, i] = vt[i]
        linear_vHt_g = self.finegd.zeros()
        self.finegd.distribute(global_linear_vHt, linear_vHt_g)
        return linear_vHt_g

    def set_extended_positions(self):
        spos_ac0 = self.atoms.get_scaled_positions() % 1.0
        spos_ac = self.extended_atoms.get_scaled_positions() % 1.0

        self.extended_calc.wfs.set_positions(spos_ac)
        self.wfs.set_positions(spos_ac0)
        self.density.set_positions(spos_ac0, self.wfs.rank_a)
        self.extended_calc.hamiltonian.set_positions(spos_ac,
                                                self.extended_calc.wfs.rank_a)

        density = self.density
        wfs = self.extended_calc.wfs
        gd = self.gd
        gd1 = self.extended_atoms.calc.gd

        magmom_a = self.extended_atoms.get_initial_magnetic_moments()
        if density.nt_sG is None:
            if wfs.kpt_u[0].f_n is None or wfs.kpt_u[0].C_nM is None:
                f_sM = np.empty((self.nspins, wfs.basis_functions.Mmax))
                self.extended_D_asp = {}
                density.D_asp = {}
                f_asi = {}
                c = density.charge / len(density.setups)  
                for a in wfs.basis_functions.atom_indices:
                    f_si = wfs.setups[a].calculate_initial_occupation_numbers(
                             magmom_a[a], density.hund, charge=c)
                    
                    if a in wfs.basis_functions.my_atom_indices:
                        self.extended_D_asp[a] = \
                                wfs.setups[a].initialize_density_matrix(f_si)
                    f_asi[a] = f_si

                for a in self.wfs.basis_functions.atom_indices:
                    setup = self.wfs.setups[a]
                    f_si = setup.calculate_initial_occupation_numbers(
                                  density.magmom_a[a], density.hund, charge=c)
                    if a in self.wfs.basis_functions.my_atom_indices:
                        density.D_asp[a] = setup.initialize_density_matrix(
                                                                         f_si)                    

                all_D_asp = []
                for a, setup in enumerate(wfs.setups):
                    D_sp = self.extended_D_asp.get(a)
                    if D_sp is None:
                        ni = setup.ni
                        D_sp = np.empty((density.nspins, ni * (ni + 1) // 2))
                    if density.gd.comm.size > 1:
                        density.gd.comm.broadcast(D_sp,
                                     self.extended_calc.hamiltonian.rank_a[a])
                    all_D_asp.append(D_sp)      
                
                D_asp = all_D_asp[:len(self.atoms)]
                distribute_D_asp(D_asp, density)

                nt_sG = gd1.zeros(self.nspins)
                wfs.basis_functions.add_to_density(nt_sG, f_asi)
                nn = self.surround.nn[0]
                density.nt_sG = self.surround.uncapsule(nn, nt_sG, gd1, gd)
                density.nt_sG += density.nct_G
                density.normalize()
                comp_charge = density.calculate_multipole_moments()
                density.interpolate(comp_charge)
                density.calculate_pseudo_charge(comp_charge)
            else:
                density.nt_sG = self.gd.empty(self.nspins)
                density.calculate_pseudo_density(wfs)
                density.nt_sG += density.nct_G
                comp_charge = density.calculate_multipole_moments()
                density.interpolate(comp_charge)
                density.calculate_pseudo_charge(comp_charge)

        self.update_hamiltonian()
        self.scf.reset()
        self.forces.reset()
        self.print_positions()
        
