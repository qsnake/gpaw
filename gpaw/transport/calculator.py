import pickle

from ase.transport.selfenergy import LeadSelfEnergy
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import function_integral, fermidistribution
from ase import Atoms, Atom, monkhorst_pack, Hartree
import ase
import numpy as np

import gpaw
from gpaw import GPAW
from gpaw import Mixer
from gpaw import restart as restart_gpaw
from gpaw.transport.tools import get_realspace_hs, get_kspace_hs, \
     tri2full, remove_pbc
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.transport.intctrl import IntCtrl
from gpaw.utilities.timing import Timer

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
        else:
            GPAW.__init__(self, **self.gpw_kwargs)            
            
    def set_transport_kwargs(self, **transport_kwargs):
        kw = transport_kwargs  
        p =  self.set_default_transport_parameters()
        if kw.get('pl_atoms') == None or kw.get('pl_cells') == None:
            raise KeyError('wrong lead information')
        
        p['pl_atoms'] = kw['pl_atoms']
        p['pl_cells'] = kw['pl_cells']
        
        self.gpw_kwargs = kw.copy()
        for key in kw:
            if key in ['pl_atoms', 'pl_cells', 'pl_kpts', 
                           'lead_restart', 'scat_restart', 'save_file',
                           'restart_file', 'cal_loc', 'recal_path',
                           'use_qzk_boundary', 'use_linear_vt_mm',
                           'use_buffer', 'buffer_atoms', 'LR_leads',
                           'bias', 'gate']:
                del self.gpw_kwargs[key]
            if key in ['pl_kpts']:
                p['pl_kpts'] = kw['pl_kpts']
            if key in ['lead_restart']:
                p['lead_restart'] = kw['lead_restart']
            if key in ['scat_restart']:
                p['scat_restart'] = kw['scat_restart']
            if key in ['save_file']:
                p['save_file'] = kw['save_file']
            if key in ['restart_file']:
                p['restart_file'] = kw['restart_file']
            if key in ['cal_loc']:
                p['cal_loc'] = kw['cal_loc']
            if key in ['recal_path']:
                p['recal_path'] = kw['recal_path']
            if key in ['use_qzk_boundary']:
                p['use_qzk_boundary'] = kw['use_qzk_boundary']
            if key in ['use_linear_vt_mm']:
                p['use_linear_vt_mm'] = kw['use_linear_vt_mm']
            if key in ['use_buffer']:
                p['use_buffer'] = kw['use_buffer']
            if key in ['buffer_atoms']:
                p['buffer_atoms'] = kw['buffer_atoms']
            if key in ['LR_leads']:
                p['LR_leads'] = kw['special_lead']
            if key in ['bias']:
                p['bias'] = kw['bias']
            if key in ['gate']:
                p['gate'] = kw['gate']

        self.transport_parameters = p
        self.pl_atoms = p['pl_atoms']
        self.pl_cells = p['pl_cells']
        self.lead_num = len(self.pl_atoms)
        self.pl_kpts = p['pl_kpts']
        self.d = p['d']
        self.lead_restart = p['lead_restart']
        self.scat_restart = p['scat_restart']
        self.save_file = p['save_file']
        self.restart_file = p['restart_file']
        self.cal_loc = p['cal_loc']
        self.recal_path = p['recal_path']
        self.use_qzk_boundary = p['use_qzk_boundary']
        self.use_linear_vt_mm = p['use_linear_vt_mm']
        self.use_buffer = p['use_buffer']
        self.buffer_atoms = p['buffer_atoms']
        self.LR_leads = p['LR_leads']
        self.bias = p['bias']
        self.gate = p['gate']
        
        if self.scat_restart and self.restart_file == None:
            self.restart_file = 'scat'
        
        self.master = (world.rank==0)
        self.cal_loc = self.cal_loc and max(abs(self.bias)) != 0
        if self.use_linear_vt_mm:
            self.use_buffer = False
        
        if not self.LR_leads and self.buffer_atoms == None:
            raise RuntimeError('need to point out the buffer_atoms')
        if self.LR_leads and self.lead_num != 2:
            raise RuntimeErrir('wrong way to use keyword LR_leads')
        if not self.LR_leads and self.lead_num != len(self.buffer_atoms):
            raise RuntimeError('worng buffer_atoms information')
          
        self.initialized_transport = False
      
        nl = self.lead_num
        assert nl >=2 and nl == len(self.pl_cells) and nl == len(self.bias)
        
        self.atoms_l = [None] * self.lead_num
        
        kpts = kw['kpts']
        if kpts[0] == 1 and kpts[1] == 1:
            self.gpw_kwargs['usesymm'] = None
        else:
            self.gpw_kwargs['usesymm'] = False            
 
    def set_default_transport_parameters(self):
        p = {}
        p['pl_atoms'] = []
        p['pl_cells'] = []
        p['pl_kpts'] = []
        p['d'] = 2
        p['lead_restart'] = False
        p['scat_restart'] = False
        p['save_file'] = True
        p['restart_file'] = None
        p['cal_loc'] = False
        p['recal_path'] = False
        p['use_qzk_boundary'] = False 
        p['use_linear_vt_mm'] = False
        p['use_buffer'] = True
        p['buffer_atoms'] = None
        p['LR_leads'] = True
        p['bias'] = []
        p['gate'] = 0
        p['verbose'] = False
        return p     

    def set_atoms(self, atoms):
        self.atoms = atoms
        
    def initialize_transport(self):
        if not self.initialized:
            self.initialize()
            self.set_positions()
        self.nspins = self.wfs.nspins
        self.ntkmol = self.gpw_kwargs['kpts'][self.d]
        self.ntklead = self.pl_kpts[self.d]
        if self.ntkmol == len(self.wfs.bzk_kc):
            self.npk = 1
            self.kpts = self.wfs.bzk_kc
        else:
            self.npk = len(self.wfs.ibzk_kc) / self.ntkmol
            self.kpts = self.wfs.ibzk_kc
        self.gamma = len(self.kpts) == 1
        self.nbmol = self.wfs.setups.nao
        self.dimt_lead = []
        self.nblead = []

        for i in range(self.lead_num):
            self.atoms_l[i] = self.get_lead_atoms(i)
            calc = self.atoms_l[i].calc
            atoms = self.atoms_l[i]
            if not calc.initialized:
                calc.initialize(atoms)
                calc.set_positions(atoms)
            self.dimt_lead.append(calc.gd.N_c[self.d])
            self.nblead.append(calc.wfs.setups.nao)
        if self.npk == 1:    
            self.lead_kpts = self.atoms_l[0].calc.wfs.bzk_kc
        else:
            self.lead_kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        self.kpt_comm = world.new_communicator(np.arange(world.size))
        self.allocate_cpus()
        self.initialize_matrix()
        self.get_lead_index()
        self.get_buffer_index()
        
        if self.nbmol <= np.sum(self.nblead):
            self.use_buffer = False
            if self.master:
                self.text('Moleucle is too small, force not to use buffer')
            
        if self.use_buffer: 
            self.buffer = self.nblead
            self.print_index = self.buffer_index
        else:
            self.buffer = [0] * self.lead_num
            self.print_index = self.lead_index
        self.set_buffer()
            
        self.fermi = 0
        self.current = 0
        self.linear_mm = None
        
        for i in range(self.lead_num):
            self.update_lead_hamiltonian(i)
            self.initialize_lead(i)
        world.barrier()
        self.check_edge()
        self.get_edge_density()
        del self.atoms_l
                
        self.initialized_transport = True

    def get_lead_index(self):
        basis_list = [setup.niAO for setup in self.wfs.setups]
        for i in range(self.lead_num):
            for j in self.pl_atoms[i]:
                begin = np.sum(np.array(basis_list[:j], int))
                for n in range(basis_list[j]):
                    self.lead_index[i].append(begin + n) 
            self.lead_index[i] = np.array(self.lead_index[i], int)
            
    def get_buffer_index(self):
        if not self.use_buffer:
            pass
        elif self.LR_leads:
            for i in range(self.lead_num):
                if i == 0:
                    self.buffer_index[i] = self.lead_index[i] - self.nblead[i]
                if i == 1:
                    self.buffer_index[i] = self.lead_index[i] + self.nblead[i]        
        else:
            basis_list = [setup.niAO for setup in self.wfs.setups]
            for i in range(self.lead_num):
                for j in self.buffer_atoms[i]:
                    begin = np.sum(np.array(basis_list[:j], int))
                    for n in range(basis_list[j]):
                        self.buffer_index[i].append(begin + n) 
                self.buffer_index[i] = np.array(self.buffer_index[i], int)            
            
    def initialize_matrix(self):
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

        npk = self.my_npk
        
        if npk == 1:
            dtype = float
        else:
            dtype = complex
            
        for i in range(self.lead_num):
            ns = self.atoms_l[i].calc.wfs.nspins        
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
            self.ed_pkmm.append(np.empty((ns, npk, nb, nb)))
            self.lead_index.append([])
            self.inner_lead_index.append([])
            self.buffer_index.append([])
        
        self.ec = np.empty([self.lead_num, ns])        
        if self.gamma:
            dtype = float
        else:
            dtype = complex
 
        ns = self.nspins
        nk = len(self.my_kpts)
        nb = self.nbmol
        self.h_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.d_skmm = np.empty((ns, nk, nb, nb), dtype)
        self.s_kmm = np.empty((nk, nb, nb), dtype)
        
        if npk == 1:
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
        rank = world.rank
        size = world.size
        npk = self.npk
        npk_each = npk / size
        r0 = rank * npk_each
        self.my_pk = np.arange(r0, r0 + npk_each)
        self.my_npk = npk_each
    
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

    def update_lead_hamiltonian(self, l):
        if not self.lead_restart:
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            kpts = self.lead_kpts 
            self.hl_skmm[l], self.sl_kmm[l] = self.get_hs(atoms.calc)
            self.dl_skmm[l] = self.initialize_density_matrix('lead', l)
            if self.save_file:
                atoms.calc.write('lead' + str(l) + '.gpw')                    
                self.pl_write('lead' + str(l) + '.mat',
                                                  (self.hl_skmm[l],
                                                   self.dl_skmm[l],
                                                   self.sl_kmm[l]))            
        else:
            atoms, calc = restart_gpaw('lead' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            (self.hl_skmm[l],
             self.dl_skmm[l],
             self.sl_kmm[l]) = self.pl_read('lead' + str(l) + '.mat')
  
    def update_scat_hamiltonian(self):
        if not self.scat_restart:
            atoms = self.atoms
            atoms.calc = self
            atoms.get_potential_energy()
            self.atoms = atoms
            rank = world.rank
            self.h_skmm, self.s_kmm = self.get_hs(atoms.calc)
            self.d_skmm = self.initialize_density_matrix('scat')
            if self.save_file:
                self.write('scat.gpw')
                self.pl_write('scat.mat', (self.h_skmm,
                                           self.d_skmm,
                                           self.s_kmm))
            self.save_file = False
        else:
            self.set_positions()
            self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(
                                                     self.restart_file + '.mat')
            self.set_text('restart.txt', self.verbose)
            self.scat_restart = False

            
    def get_hs(self, calc):
        wfs = calc.wfs
        Ef = calc.get_fermi_level()
        eigensolver = wfs.eigensolver
        ham = calc.hamiltonian
        S_qMM = wfs.S_qMM.copy()
        for S_MM in S_qMM:
            tri2full(S_MM)
        H_sqMM = np.empty((wfs.nspins,) + S_qMM.shape, complex)
        for kpt in wfs.kpt_u:
            eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
            H_MM = eigensolver.H_MM
            tri2full(H_MM)
            H_MM *= Hartree
            H_MM -= Ef * S_qMM[kpt.q]
            H_sqMM[kpt.s, kpt.q] = H_MM
        return H_sqMM, S_qMM

    def get_lead_atoms(self, l):
        """Here is a multi-terminal version """
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl.center()
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l):
        p = self.gpw_kwargs.copy()
        p['nbands'] = None
        kpts = list(p['kpts'])
        if not hasattr(self, 'pl_kpts') or self.pl_kpts==None:
            kpts[self.d] = 2 * int(25.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts[self.d] = self.pl_kpts[self.d]
        p['kpts'] = kpts
        if 'mixer' in p: # XXX Works only if spin-paired
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return gpaw.GPAW(**p)

    def negf_prepare(self):
        if not self.initialized_transport:
            self.initialize_transport()
        self.update_scat_hamiltonian()
        world.barrier()
        self.initialize_mol()
        self.boundary_check()
    
    def initialize_lead(self, l):
        nspins = self.nspins
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

    def initialize_mol(self):
        ntk = self.ntkmol
        kpts = self.my_kpts
        self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
        self.s_pkmm = self.substract_pk(ntk, kpts, self.s_kmm)
        self.d_spkmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h')
        #This line only for two_terminal
        self.s_pkcmm , self.d_spkcmm = self.fill_density_matrix()

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
        tkpts = self.pick_out_tkpts(ntk, kpts)
        for i in range(npk):
            n = i * ntk
            for j in range(ntk):
                if hors == 'h':
                    tk_mm[:, j] = np.copy(k_mm[:, n + j])
                elif hors == 's':
                    tk_mm[j] = np.copy(k_mm[n + j])
            if hors == 'h':
                pk_mm[:, i] = get_realspace_hs(tk_mm, None,
                                               tkpts, weight, position)
            elif hors == 's':
                pk_mm[i] = get_realspace_hs(None, tk_mm,
                                                   tkpts, weight, position)
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
            for i in range(self.nspins):
                for j in range(self.my_npk):
                    self.ed_pkmm[n][i, j] = np.dot(self.dl_spkcmm[n][i, j],
                                                 self.sl_pkcmm[n][j].T.conj())
                    self.ec[n, i] += np.trace(self.ed_pkmm[n][i, j])   
        self.kpt_comm.sum(self.ec)
        self.ed_pkmm *= 3 - self.nspins
        self.ec *= 3 - self.nspins
        if self.master:
            for i in range(self.nspins):
                for n in range(self.lead_num):
                    total_edge_charge  = self.ec[n, i] / self.npk
                self.text('edge_charge[%d]=%f' % (i, total_edge_charge))

    def pick_out_tkpts(self, ntk, kpts):
        npk = self.npk
        tkpts = np.zeros([ntk, 3])
        for i in range(ntk):
            tkpts[i, self.d] = kpts[i, self.d]
        return tkpts

    def initialize_density_matrix(self, region, l=0):
        npk = self.npk
        if region == 'lead':
            ntk = self.ntklead
            calc = self.atoms_l[l].calc
            d_skmm = np.empty(self.hl_skmm[l].shape, self.hl_skmm[l].dtype)
        
        if region == 'scat':
            ntk = self.ntkmol
            calc = self
            d_skmm = np.empty(self.h_skmm.shape, self.h_skmm.dtype)
            
        for kpt in calc.wfs.kpt_u:
            C_nm = kpt.C_nM
            f_nn = np.diag(kpt.f_n)
            d_skmm[kpt.s, kpt.q] = np.dot(C_nm.T.conj(),
                                          np.dot(f_nn, C_nm)) * ntk * npk
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
        ham_diff = np.empty([self.nspins, self.lead_num])
        den_diff = np.empty([self.nspins, self.lead_num])
        for s in range(self.nspins):
            for i in range(self.lead_num):
                ind = self.print_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                ham_diff[s, i] = np.max(abs(self.h_spkmm[:, :, ind.T, ind] -
                                                            self.hl_spkmm[i]))
                den_diff[s, i] = np.max(abs(self.d_spkmm[:, :, ind.T, ind] -
                                                            self.dl_spkmm[i]))
        self.edge_ham_diff = np.max(ham_diff)
        self.edge_den_diff = np.max(den_diff)
        
        if self.edge_ham_diff > tol and self.master:
            self.text('Warning*: hamiltonian boundary difference %f' %
                                                           self.edge_ham_diff)
        if self.edge_den_diff > tol and self.master:
            self.text('Warning*: density boundary difference %f' % 
                                                           self.edge_den_diff)
            
    def get_selfconsistent_hamiltonian(self):
        self.negf_prepare()
        self.initialize_scf()
        while not self.cvgflag and self.step < self.max_steps:
            self.iterate()
            self.cvgflag = self.d_cvg and self.h_cvg
            self.step +=  1
        self.scf.converged = self.cvgflag
    
    def get_hamiltonian_matrix(self):
        self.timer.start('HamMM')            
        self.den2fock()
        self.timer.stop('HamMM')
        self.h_spkmm = self.substract_pk(self.ntkmol, self.kpts,
                                         self.h_skmm, 'h')
        if self.verbose and self.master:
            self.text('HamMM', self.timer.gettime('HamMM'), 'second')        
  
    def get_density_matrix(self):
        self.timer.start('DenMM')
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        if self.use_qzk_boundary:
            self.fill_lead_with_scat()
            for i in range(self.lead_num):
                self.selfenergies[i].set_bias(0)
        if self.recal_path:
            for s in range(self.nspins):
                for k in range(self.my_npk):
                    den[s, k] = self.get_eqintegral_points(s, k)
                    denocc[s, k] = self.get_neintegral_points(s, k)
                    if self.cal_loc:
                        denloc[s, k] = self.get_neintegral_points(s, k,
                                                                  'locInt')
                    self.d_spkmm[s, k, ind.T, ind] = self.spin_coff * (
                                                              den[s, k] +
                                                              denocc[s, k])
        else:
            for s in range(self.nspins):
                for k in range(self.my_npk):
                    self.d_spkmm[s, k, ind.T, ind] = self.spin_coff *   \
                                                     self.fock2den(s, k)
        self.timer.stop('DenMM')
        if self.verbose and self.master:
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
            if self.step > 0:
                self.diff_h = np.max(abs(self.hamiltonian.vt_sG -
                                    self.ham_vt_old))
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
                    self.output('step')
                if self.master:
                    self.text('density: diff = %f  tol=%f' % (self.diff_d,
                                                  self.scf.max_density_error))
                if self.diff_d < self.scf.max_density_error:
                    cvg = True
        return cvg
 
    def initialize_scf(self):
        self.intctrl = IntCtrl(self.occupations.kT * Hartree,
                                                  self.fermi, self.bias)
        
        self.selfenergies = []
        
        for i in range(self.lead_num):
            self.selfenergies.append(LeadSelfEnergy((self.hl_spkmm[i][0,0],
                                                         self.sl_pkmm[i][0]), 
                                            (self.hl_spkcmm[i][0,0],
                                                         self.sl_pkcmm[i][0]),
                                            (self.hl_spkcmm[i][0,0],
                                                         self.sl_pkcmm[i][0]),
                                             0))

            self.selfenergies[i].set_bias(self.bias[i])

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_spkmm[0,0],
                                           S=self.s_pkmm[0], eta=0)

        self.calculate_integral_path()
    
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
        self.ham_vt_tol = 1e-4
        
        self.step = 0
        self.cvgflag = False
        self.spin_coff = 3. - self.nspins
        self.max_steps = 200
        self.h_cvg = False
        self.d_cvg = False
        
    def initialize_path(self):
        self.eqpathinfo = []
        self.nepathinfo = []
        self.locpathinfo = []
       
        for s in range(self.nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            if self.cal_loc:
                self.locpathinfo.append([])                
            if self.cal_loc:
                self.locpathinfo.append([])
            for k in self.my_pk:
                self.eqpathinfo[s].append(PathInfo('eq', self.lead_num))
                self.nepathinfo[s].append(PathInfo('ne', self.lead_num))    
                if self.cal_loc:
                    self.locpathinfo[s].append(PathInfo('eq', self.lead_num))
                    
    def calculate_integral_path(self):
        self.initialize_path()
        nb = self.nbmol_inner
        ns = self.nspins
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
        maxintcnt = 100
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], self.d_spkmm.dtype)
        intctrl = self.intctrl
        
        self.zint = [0] * maxintcnt
        self.fint = []

        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append(np.empty([maxintcnt, nblead, nblead],
                                                           complex))
        self.cntint = -1

        sg = self.selfenergies
        for i in range(self.lead_num):
            sg[i].h_ii = self.hl_spkmm[i][s, k]
            sg[i].s_ii = self.sl_pkmm[i][k]
            sg[i].h_ij = self.hl_spkcmm[i][s, k]
            sg[i].s_ij = self.sl_pkcmm[i][k]
            sg[i].h_im = self.hl_spkcmm[i][s, k]
            sg[i].s_im = self.sl_pkcmm[i][k]
        
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
        for i in range(self.lead_num):
            siglist.append([])
        for i in sgforder:
            flist.append(self.fint[i])
 
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            sigma= np.empty([nblead, nblead], complex)
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
        maxintcnt = 100
        intctrl = self.intctrl

        self.zint = [0] * maxintcnt
        self.tgtint = []
        for i in range(self.lead_num):
            nblead = self.nblead[i]
            self.tgtint.append(np.empty([maxintcnt, nblead, nblead], complex))
            
        sg = self.selfenergies
        for i in range(self.lead_num):
            sg[i].h_ii = self.hl_spkmm[i][s, k]
            sg[i].s_ii = self.sl_pkmm[i][k]
            sg[i].h_ij = self.hl_spkcmm[i][s, k]
            sg[i].s_ij = self.sl_pkcmm[i][k]
            sg[i].h_im = self.hl_spkcmm[i][s, k]
            sg[i].s_im = self.sl_pkcmm[i][k]
        
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        if calcutype == 'neInt' or calcutype == 'neVirInt':
            for n in range(1, len(intctrl.neintpath)):
                self.cntint = -1
                self.fint = []
                for i in range(self.lead_num):
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
                for i in range(self.lead_num):
                    flist.append([[],[]])
                    siglist.append([])

                for l in range(self.lead_num):
                    nblead = self.nblead[l]
                    sigma= np.empty([nblead, nblead], complex)
                    for j in sgforder:
                        for i in [0, 1]:
                            fermi_factor = self.fint[l][i][j]
                            flist[l][i].append(fermi_factor)   
                        sigma = self.tgtint[l][j]
                        siglist[l].append(sigma)
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
            for i in range(self.lead_num):
                siglist.append([])
            for i in sgforder:
                flist.append(self.fint[i])
            sigma= np.empty([nblead, nblead], complex)
            for i in range(self.lead_num):
                for j in sgforder:
                    sigma = self.tgtint[i][j]
                    siglist[i].append(sigma)
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
        stepintcnt = 100
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
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt
                for n in range(self.lead_num):
                    nblead = self.nblead[n]
                    tmp = self.tgtint[n].shape[0]
                    tmptgt = np.copy(self.tgtint[n])
                    self.tgtint[n] = np.empty([tmp + stepintcnt, nblead, nblead],
                                                                  complex)
                    self.tgtint[n][:tmp] = tmptgt
            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in range(self.lead_num):
                self.tgtint[j][self.cntint] = self.selfenergies[j](zp[i])
            
            for j in range(self.lead_num):
                ind = self.inner_lead_index[j]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += self.tgtint[j][self.cntint]             
                gamma[j, ind.T, ind] += self.selfenergies[j].get_lambda(zp[i])

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
                aocc = np.dot(gr, gammaocc)
                aocc = np.dot(aocc, gr.T.conj())
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
                avir = np.dot(gr, gammavir)
                avir = np.dot(avir, gr.T.conj())
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
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        denocc = np.zeros([nbmol, nbmol], complex)
  
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
        self.greenfunction.H = self.h_spkmm[s, k, ind.T, ind]
        self.greenfunction.S = self.s_pkmm[k, ind.T, ind]

        den = self.eq_fock2den(self.eqpathinfo[s][k], den)
        denocc = self.ne_fock2den(self.nepathinfo[s][k], denocc, ov='occ')       
        den += denocc

        if self.cal_loc:
            denloc = np.zeros([nbmol, nbmol], complex)
            denvir = np.zeros([nbmol, nbmol], complex)
            denloc = self.eq_fock2den(self.locpathinfo[s][k], denloc)
            denvir = self.ne_fock2den(self.nepathinfo[s][k], denvir, ov='vir')
            weight_mm = self.integral_diff_weight(denocc, denvir,
                                                                 'transiesta')
            diff = (denloc - (denocc + denvir)) * weight_mm
            den += diff
            percents = np.sum( diff * diff ) / np.sum( denocc * denocc )
            self.text('local percents %f' % percents)
        den = (den + den.T.conj()) / 2
        return den    

    def ne_fock2den(self, pathinfo, den, ov='occ'):
        zp = pathinfo.energy
        for i in range(len(zp)):
            sigma = np.zeros(den.shape, complex)
            sigmalesser = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i]
            gr = self.greenfunction.calculate(zp[i], sigma)

            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])                
                sigmatmp = pathinfo.sigma[n][i]
                if ov == 'occ':
                    fermifactor = np.real(pathinfo.fermi_factor[n][0][i])
                elif ov == 'vir':
                    fermifactor = np.real(pathinfo.fermi_factor[n][1][i])                    
                sigmalesser[ind.T, ind] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())
   
            glesser = np.dot(sigmalesser, gr.T.conj())
            glesser = np.dot(gr, glesser)
            weight = pathinfo.weight[i]            
            den += glesser * weight / np.pi / 2
        return den  

    def eq_fock2den(self, pathinfo, den):
        zp = pathinfo.energy
        for i in range(len(zp)):
            sigma = np.zeros(den.shape, complex)
            for n in range(self.lead_num):
                ind = self.inner_lead_index[n]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                sigma[ind.T, ind] += pathinfo.sigma[n][i]
            gr = self.greenfunction.calculate(zp[i], sigma)
            fermifactor = pathinfo.fermi_factor[i]
            weight = pathinfo.weight[i]
            den += gr * fermifactor * weight
        den = 1.j * (den - den.T.conj()) / np.pi / 2            
        return den

    def den2fock(self):
        self.get_density()
        self.update_kinetic()
        self.hamiltonian.update(self.density)
        if not self.use_linear_vt_mm:
            self.hamiltonian.vt_sG += self.get_linear_potential()
        self.h_skmm, self.s_kmm = self.get_hs(self)
        if self.use_linear_vt_mm:
            if self.linear_mm == None:
                self.linear_mm = self.get_linear_potential_matrix()            
            self.h_skmm += self.linear_mm

    
    def get_forces(self, atoms):
        self.get_selfconsistent_hamiltonian()
        self.forces.F_av = None
        return GPAW.get_forces(self, atoms)
       
    def get_density(self):
        #Calculate pseudo electron-density based on green function.
        ns = self.nspins
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
                qr_mm[s, i] += np.dot(dr_mm[s, i, 1], self.s_pkmm[i]) 
        if ntk != 1:
            for i in range(self.lead_num):
                ind = self.print_index[i]
                dim = len(ind)
                ind = np.resize(ind, [dim, dim])
                qr_mm[:, :, ind.T, ind] += self.ed_pkmm[i]
        self.kpt_comm.sum(qr_mm)
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
                        self.d_skmm[s, j, i] = get_kspace_hs(None,
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
            kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
        self.density.update(self.wfs)

    
    def print_boundary_charge(self, qr_mm):
        qr_mm = np.sum(np.sum(qr_mm, axis=0), axis=0)
        edge_charge = []
        natom_inlead = np.empty([self.lead_num])
        natom_print = np.empty([self.lead_num])
        
        for i in range(self.lead_num):
            natom_inlead[i] = len(self.pl_atoms[i])
            nb_atom = self.nblead[i] / natom_inlead[i]
            if self.use_buffer:
                pl1 = self.buffer[i]
            else:
                pl1 = self.nblead[i]
            natom_print[i] = int(pl1 / nb_atom)
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
        qr_mm = np.empty([self.nspins, self.my_npk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.my_npk):
                qr_mm[i,j] = np.dot(d_spkmm[i, j], self.s_pkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))
        Qmol += np.sum(self.ec)
        Qmol = self.kpt_comm.sum(Qmol) / self.npk
        return Qmol        

    def get_linear_potential(self):
        linear_potential = np.zeros(self.hamiltonian.vt_sG.shape)
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        if self.use_buffer:
            buffer_dim = self.dimt_lead
        else:
            buffer_dim = 0
        scat_dim = dimt - np.sum(buffer_dim)
        bias= []
        for i in range(self.lead_num): 
            bias.append(self.bias[i] / Hartree)
        vt = np.empty([dimt])
        if buffer_dim !=0:
            vt[:buffer_dim[0]] = bias[0]
            vt[-buffer_dim[1]:] = bias[1]         
            vt[buffer_dim[0]: -buffer_dim[1]] = np.linspace(bias[0],
                                                         bias[1], scat_dim)
        else:
            vt = np.linspace(bias[0], bias[1], scat_dim)
        for s in range(self.nspins):
            for i in range(dimt):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dimp) + 1)
        return linear_potential
    
    def output(self, filename):
        self.pl_write(filename + '.mat', (self.h_skmm, self.d_skmm, self.s_kmm))
        world.barrier()
        self.write(filename + '.gpw')
        if world.rank == 0:
            fd = file(filename, 'wb')
            pickle.dump((
                        self.bias,
                        self.gate,
                        self.intctrl,
                        self.eqpathinfo,
                        self.nepathinfo,
                        self.kpts,
                        self.lead_kpts,
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
        fd = file(filename, 'rb')
        (self.bias,
         self.gate,
         self.intctrl,
         self.eqpathinfo,
         self.nepathinfo,
         self.kpts,
         self.lead_kpts,
         self.forces,
         self.current,
         self.step,
         self.cvgflag
         ) = pickle.load(fd)
        fd.close()
        self.h_skmm, self.d_skmm, self.s_kmm = self.pl_read(filename + '.mat')
        (self.h1_skmm,
                 self.d1_skmm,
                 self.s1_kmm) = self.pl_read('lead0.mat', collect=True)
        (self.h2_skmm,
                 self.d2_skmm,
                 self.s2_kmm,
                 self.ntklead) = self.pl_read('lead1.mat', collect=True)
        self.nspins = self.h1_skmm.shape[0]
        self.npk = len(self.lead_kpts) / self.ntklead
        self.ntkmol = len(self.kpts) / self.npk
        self.nblead = self.h1_skmm.shape[-1]
        self.nbmol = self.h_skmm.shape[-1]
        #self.atoms.calc.hamiltonian.vt_sG += self.get_linear_potential()
        world.barrier()
    
    #this should be changed
    def analysis(self, filename):
        self.input(filename)
        self.allocate_cpus()
        self.initialize_lead(0)
        self.initialize_lead(1)
        self.initialize_mol()
        if self.nblead == self.nbmol:
            self.buffer = 0
        else:
            self.buffer = self.nblead
        self.set_buffer()
      
    def set_calculator(self, e_points):
        from ase.transport.calculators import TransportCalculator
        ind = self.inner_mol_index
        dim = len(ind)
        ind = np.resize(ind, [dim, dim])
     
        h_scat = np.sum(self.h_spkmm[0, :, ind.T, ind], axis=0) / self.npk
        h_scat = np.real(h_scat)
        
        h_lead1 = self.double_size(np.sum(self.h1_spkmm[0], axis=0),
                                   np.sum(self.h1_spkmm_ij[0], axis=0))
        h_lead2 = self.double_size(np.sum(self.h2_spkmm[0], axis=0),
                                   np.sum(self.h2_spkmm_ij[0], axis=0))
        h_lead1 /= self.npk
        h_lead2 /= self.npk
        
        h_lead1 = np.real(h_lead1)
        h_lead2 = np.real(h_lead2)
        
        s_scat = np.sum(self.s_pkmm[:, ind.T, ind], axis=0) / self.npk
        s_scat = np.real(s_scat)
        
        s_lead1 = self.double_size(np.sum(self.s1_pkmm, axis=0),
                                   np.sum(self.s1_pkmm_ij, axis=0))
        s_lead2 = self.double_size(np.sum(self.s2_pkmm, axis=0),
                                   np.sum(self.s2_pkmm_ij, axis=0))
        
        s_lead1 /= self.npk
        s_lead2 /= self.npk
        
        s_lead1 = np.real(s_lead1)
        s_lead2 = np.real(s_lead2)
        
        tcalc = TransportCalculator(energies=e_points,
                                    h = h_scat,
                                    h1 = h_lead2,
                                    h2 = h_lead2,
                                    s = s_scat,
                                    s1 = s_lead2,
                                    s2 = s_lead2,
                                    dos = True
                                   )
        return tcalc
    
    def plot_dos(self, E_range, point_num = 30):
        e_points = np.linspace(E_range[0], E_range[1], point_num)
        tcalc = self.set_calculator(e_points)
        tcalc.get_transmission()
        tcalc.get_dos()
        f1 = self.intctrl.leadfermi[0] * (np.zeros([10, 1]) + 1)
        f2 = self.intctrl.leadfermi[1] * (np.zeros([10, 1]) + 1)
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
        
    def plot_v(self, vt=None, tit=None, ylab=None, l_MM=False):
        import pylab
        self.use_linear_vt_mm = l_MM
        if vt == None:
            vt = self.atoms.calc.hamiltonian.vt_sG + self.get_linear_potential()
        dim = vt.shape
        for i in range(3):
            vt = np.sum(vt, axis=0) / dim[i]
        pylab.plot(vt * Hartree, 'b--o')
        if ylab == None:
            ylab = 'energy(eV)'
        pylab.ylabel(ylab)
        if tit == None:
            tit = 'bias=' + str(self.bias)
        pylab.title(tit)
        pylab.show()

    def plot_d(self, nt=None, tit=None, ylab=None):
        import pylab
        if nt == None:
            nt = self.atoms.calc.density.nt_sG
        dim = nt.shape
        for i in range(3):
            nt = np.sum(nt, axis=0) / dim[i]
        pylab.plot(nt, 'b--o')
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
                current[i] += tcalc.T_e[j] * weight[j] * fermi_factor[i][j]
        self.current = current[0] - current[1]
        return self.current
    
    def get_nepath_info(self):
        if hasattr(self, 'nepathinfo'):
            energy = self.nepathinfo[0][0].energy
            weight = self.nepathinfo[0][0].weight
            fermi_factor = self.nepathinfo[0][0].fermi_factor
      
        return energy, weight, fermi_factor
    
    def set_buffer(self):
        self.nbmol_inner = self.nbmol - np.sum(self.buffer)
        ind = np.arange(self.nbmol)
        buffer_ind = []
        for i in range(self.lead_num):
            buffer_ind += list(self.buffer_index[i])
        ind = np.delete(ind, buffer_ind)
        self.inner_mol_index = ind
        for i in range(self.lead_num):
             self.inner_lead_index[i] = np.searchsorted(ind,
                                                           self.lead_index[i])        

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
                    dim = (dim[0],) + (dim[1] * world.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] * world.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_write')
                totalmat = np.empty(dim, dtype=matlist[i].dtype)
                self.kpt_comm.gather(matlist[i], 0, totalmat)
                total_matlist.append(totalmat)
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
                    dim = (dim[0],) + (dim[1] / world.size,) + dim[2:]
                elif len(dim) == 3:
                    dim = (dim[0] / world.size,) + dim[1:]
                else:
                    raise RuntimeError('wrong matrix dimension for pl_read')
                local_mat = np.empty(dim, dtype=total_matlist[i].dtype)
                self.kpt_comm.scatter(total_matlist[i], local_mat, 0)
            elif type(total_matlist[i]) == np.ndarray:
                local_mat = np.empty(total_matlist[i].shape,
                                             dtype= total_matlist[i].dtype)
                local_mat = total_matlist[i]
                self.kpt_comm.broadcast(local_mat, 0)
            else:
                local_mat = np.zeros([1], dtype=int)
                local_mat[0] = total_matlist[i]
                self.kpt_comm.broadcast(local_mat, 0)
                local_mat = local_mat[0]
            matlist.append(local_mat)
        return matlist

    def fill_lead_with_scat(self):
        for  i in range(self.lead_num):
            ind = self.inner_lead_index[i]
            dim = len(dim)
            ind = np.resize(ind, [dim, dim])
            self.hl_spkmm[i] = self.h_spkmm[:, :, ind.T, ind]
            self.sl_pkmm[i] = self.s_pkmm[:, ind.T, ind]
            
        self.h1_spkmm = self.h_spkmm_mol[:, :, :nblead, :nblead]
        self.s1_pkmm = self.s_pkmm_mol[:, :nblead, :nblead]
        self.h1_spkmm_ij = self.h_spkmm_mol[:, :, :nblead, nblead:2 * nblead]
        self.s1_spkmm_ij = self.s_pkmm_mol[:, :nblead, nblead:2 * nblead]
        
        self.h2_spkmm = self.h_spkmm_mol[:, :, -nblead:, -nblead:]
        self.s2_pkmm = self.s_pkmm_mol[:, -nblead:, -nblead:]
        self.h2_spkmm_ij = self.h_spkmm_mol[:, :, -nblead:, -nblead*2 : -nblead]
        self.s2_spkmm_ij = self.s_pkmm_mol[:, -nblead:, -nblead*2 : -nblead]

    def get_lead_layer_num(self):
        tol = 1e-4
        temp = []
        for lead_atom in self.atoms_l[0]:
            for i in range(len(temp)):
                if abs(atom.position[self.d] - temp[i]) < tol:
                    break
                temp.append(atom.position[self.d])
                
    def get_linear_potential_matrix(self):
        #a bad way to get linear_potential of scattering region 
        lead_atoms_num = len(self.pl_atoms[0])
        atoms_inner = self.atoms.copy()
        atoms_inner.center()
        atoms_extend = atoms_inner.copy()
        for i in range(lead_atoms_num, 0 ,-1):
            atoms_extend = atoms_inner[i:i+1] + atoms_extend + atoms_inner[-i-1:-i]
        
        atoms_extend.positions[:lead_atoms_num] = atoms_inner.positions[:lead_atoms_num]
        atoms_extend.positions[-lead_atoms_num:] = atoms_inner.positions[-lead_atoms_num:]
        
        atoms_extend.set_pbc(atoms_inner._pbc)
        d = self.d
        cell = np.diag(atoms_inner._cell.copy())
        cell[d] += self.pl_cells[0][d] * 2
        atoms_extend.set_cell(cell)
        for i in range(lead_atoms_num):
            atoms_extend.positions[i, d] -= self.pl_cells[0][d]
        for i in range(-lead_atoms_num, 0):
            atoms_extend.positions[i, d] += self.pl_cells[1][d]
        
        atoms_extend.center()
        
        atoms_extend.set_calculator(GPAW(h=0.3,
                          xc='PBE',
                          basis='szp',
                          kpts=(1,1,1),
                          width=0.2,
                          mode='lcao',
                          usesymm=None,
                          mixer=Mixer(0.1, 5, metric='new', weight=100.0)
                          ))
        
        calc = atoms_extend.calc
        self.initialize_lfc(calc, atoms_extend)
        #calc.set_positions(atoms_extend)
        
        linear_potential = calc.gd.empty(self.nspins)
        
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        dimt_lead = self.dimt_lead

        buffer_dim = dimt_lead
        scat_dim = dimt - buffer_dim * 2
        bias = self.bias / Hartree
        vt = np.empty([dimt])
        if buffer_dim !=0:
            vt[:buffer_dim] = bias / 2.0
            vt[-buffer_dim:] = -bias / 2.0        
            vt[buffer_dim : -buffer_dim] = np.linspace(bias/2.0,
                                                         -bias/2.0, scat_dim)
        else:
            vt = np.linspace(bias/2.0, -bias/2.0, scat_dim)
        for s in range(self.nspins):
            for i in range(dimt):
                linear_potential[s,:,:,i] = vt[i] * (np.zeros(dimp) + 1)

        wfs = calc.wfs
        nq = len(wfs.ibzk_qc)
        nao = wfs.setups.nao
        H_sqMM = np.empty([wfs.nspins, nq, nao, nao])
        H_MM = np.empty([nao, nao])
        for kpt in wfs.kpt_u:
            wfs.basis_functions.calculate_potential_matrix(linear_potential[kpt.s],
                                                       H_MM, kpt.q)
            tri2full(H_MM)
            H_MM *= Hartree
            H_sqMM[kpt.s, kpt.q] = H_MM
        pl1 = self.nblead
        return   H_sqMM[:, :, pl1:-pl1, pl1:-pl1]
    
    def initialize_lfc(self, calc, atoms):
        from ase.units import Bohr
        par = calc.input_parameters
        pos_av = atoms.get_positions() / Bohr
        cell_cv = atoms.get_cell() / Bohr
        pbc_c = atoms.get_pbc()
        Z_a = atoms.get_atomic_numbers()
        magmom_a = atoms.get_initial_magnetic_moments()
        
        from ase.dft import monkhorst_pack
        
        kpts = par.kpts
        if kpts is None:
            bzk_kc = np.zeros((1, 3))
        elif isinstance(kpts[0], int):
            bzk_kc = monkhorst_pack(kpts)
        else:
            bzk_kc = np.array(kpts)
         
        magnetic = magmom_a.any()

        spinpol = par.spinpol
        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        nspins = 1 + int(spinpol)
        if not spinpol:
            assert not par.hund

        fixmom = par.fixmom
        if par.hund:
            fixmom = True
            assert natoms == 1   
            
        if par.gpts is not None and par.h is None:
            N_c = np.array(par.gpts)
        else:
            if par.h is None:
                h = 0.2 / Bohr
            else:
                h = par.h / Bohr
            # N_c should be a multiple of 4:
            N_c = []
            for axis_v in cell_cv:
                L = (axis_v**2).sum()**0.5
                N_c.append(max(4, int(L / h / 4 + 0.5) * 4))
            N_c = np.array(N_c)
        
        gamma = len(bzk_kc) == 1 and not bzk_kc[0].any()

        if hasattr(calc, 'time'):
            dtype = complex
        else:
            if gamma:
                dtype = float
            else:
                dtype = complex

        if isinstance(par.xc, (str, dict)):
            from gpaw.xc_functional import XCFunctional
            xcfunc = XCFunctional(par.xc, nspins)
        else:
            xcfunc = par.xc
        from gpaw.setup import Setups
        setups = Setups(Z_a, par.setups, par.basis, nspins, par.lmax, xcfunc)

        # Brillouin zone stuff:
        if gamma:
            symmetry = None
            weight_k = np.array([1.0])
            ibzk_kc = np.zeros((1, 3))
        else:
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            symmetry, weight_k, ibzk_kc = reduce_kpoints(atoms, bzk_kc,
                                                         setups, par.usesymm)

        width = par.width
        if width is None:
            if gamma:
                width = 0
            else:
                width = 0.1 / Hartree
        else:
            width /= Hartree
            
        nao = setups.nao
        nvalence = setups.nvalence - par.charge
        
        nbands = par.nbands
        if nbands is None:
            nbands = nao
        elif nbands > nao and par.mode == 'lcao':
            raise ValueError('Too many bands for LCAO calculation: ' +
                             '%d bands and only %d atomic orbitals!' %
                             (nbands, nao))
        
        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                par.charge)

        M = magmom_a.sum()

        if nbands <= 0:
            nbands = int(nvalence + M + 0.5) // 2 + (-nbands)
            
            
        from gpaw import parsize
        if parsize is None:
            parsize = par.parsize

        from gpaw import parsize_bands
        if parsize_bands is None:
            parsize_bands = par.parsize_bands

        if nbands % parsize_bands != 0:
            raise RuntimeError('Cannot distribute %d bands to %d processors' %
                               (nbands, parsize_bands))
        mynbands = nbands // parsize_bands
        
        if not calc.wfs:
            domain_comm, kpt_comm, band_comm = calc.distribute_cpus(
                world, parsize, parsize_bands, nspins, len(ibzk_kc))

            if calc.gd is not None and calc.gd.comm.size != domain_comm.size:
                # Domain decomposition has changed, so we need to
                # reinitialize density and hamiltonian:
                self.density = None
                self.hamiltonian = None

            # Create a Domain object:
            from gpaw.domain import Domain
            calc.domain = Domain(cell_cv, pbc_c)
            calc.domain.set_decomposition(domain_comm, parsize, N_c)

            # Construct grid descriptor for coarse grids for wave functions:
            from gpaw.grid_descriptor import GridDescriptor
            calc.gd = GridDescriptor(calc.domain, N_c)

            # do k-point analysis here? XXX

            args = (calc.gd, nspins, setups,
                    nbands, mynbands,
                    dtype, world, kpt_comm, band_comm,
                    gamma, bzk_kc, ibzk_kc, weight_k, symmetry)
            from gpaw.wavefunctions import LCAOWaveFunctions
            calc.wfs = LCAOWaveFunctions(*args)
            spos_ac = atoms.get_scaled_positions() % 1.0
            from gpaw.lfc import BasisFunctions
            wfs = calc.wfs
            wfs.basis_functions = BasisFunctions(wfs.gd,
                                                  [setup.phit_j
                                                   for setup in wfs.setups],
                                                   wfs.kpt_comm,
                                                  cut=True)
            if not gamma:
                wfs.basis_functions.set_k_points(wfs.ibzk_qc)
            wfs.basis_functions.set_positions(spos_ac)
            
        
            
