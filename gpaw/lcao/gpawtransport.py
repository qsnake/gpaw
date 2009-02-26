import pickle

from ase.transport.selfenergy import LeadSelfEnergy
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import function_integral, fermidistribution
from ase import Atoms, Atom, monkhorst_pack, Hartree
import ase
import numpy as np

from gpaw import GPAW, Mixer
from gpaw import restart as restart_gpaw
from gpaw.lcao.tools import get_realspace_hs, get_kspace_hs, \
     tri2full, remove_pbc
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.lcao.IntCtrl import IntCtrl
from gpaw.lcao.CvgCtrl import CvgCtrl
from gpaw.utilities.timing import Timer

class PathInfo:
    def __init__(self, type):
        self.type = type
        self.num = 0
        self.energy = []
        self.weight = []
        self.nres = 0
        self.sigma = [[], []]
        if type == 'eq':
            self.fermi_factor = []
        elif type == 'ne':
            self.fermi_factor = [[], []]
        else:
            raise TypeError('unkown PathInfo type')

    def add(self, elist, wlist, flist, siglist):
        self.num += len(elist)
        self.energy += elist
        self.weight += wlist
        if self.type == 'eq':
            self.fermi_factor += flist
        elif self.type == 'ne':
            for i in [0, 1]:
                self.fermi_factor[i] += flist[i]
        else:
            raise TypeError('unkown PathInfo type')
        for i in [0, 1]:
            self.sigma[i] += siglist[i]

    def set_nres(self, nres):
        self.nres = nres
    
class GPAWTransport:
    
    def __init__(self, atoms, pl_atoms, pl_cells, d=0, extend=False):
        self.atoms = atoms
        self.pl_atoms = pl_atoms
        self.pl_cells = pl_cells
        self.d = d
        self.atoms_l = [None,None]
        self.h_skmm = None
        self.s_kmm = None
        self.h1_skmm = None
        self.s1_kmm = None
        self.h2_skmm = None
        self.s2_kmm = None
        self.extend = extend

    def write_left_lead(self,filename):
        self.update_lead_hamiltonian(0)

    def write(self, filename):
        self.update_lead_hamiltonian(0)

        pl1 = self.h1_skmm.shape[-1]
        h1 = np.zeros((2*pl1, 2 * pl1), complex)
        s1 = np.zeros((2*pl1, 2 * pl1), complex)

        atoms1 = self.atoms_l[0]
        calc1 = atoms1.calc
        R_c = [0,0,0] 
        h1_sii, s1_ii = get_realspace_hs(self.h1_skmm,
                                         self.s1_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)
        R_c = [0,0,0]
        R_c[self.d] = 1.0
        h1_sij, s1_ij = get_realspace_hs(self.h1_skmm,
                                         self.s1_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)

        h1[:pl1, :pl1] = h1_sii[0]
        h1[pl1:2 * pl1, pl1:2 * pl1] = h1_sii[0]
        h1[:pl1, pl1:2 * pl1] = h1_sij[0]
        tri2full(h1, 'U')
        
        s1[:pl1,:pl1] = s1_ii
        s1[pl1:2*pl1,pl1:2*pl1] = s1_ii
        s1[:pl1,pl1:2*pl1] = s1_ij
        tri2full(s1, 'U')
        
        if calc1.wfs.world.rank == 0:
            print "Dumping lead 1 hamiltonian..."
            fd = file('lead1_' + filename, 'wb')
            pickle.dump((h1, s1), fd, 2)
            fd.close()

        world.barrier()
        
        self.update_lead_hamiltonian(1) 
        pl2 = self.h2_skmm.shape[-1]
        h2 = np.zeros((2 * pl2, 2 * pl2), complex)
        s2 = np.zeros((2 * pl2, 2 * pl2), complex)

        atoms2 = self.atoms_l[1]
        calc2 = atoms2.calc
        
        h2_sii, s2_ii = get_realspace_hs(self.h2_skmm,
                                         self.s2_kmm,
                                         calc2.ibzk_kc, 
                                         calc2.weight_k,
                                         R_c=(0,0,0))
        R_c = [0,0,0]
        R_c[self.d] = 1.0

        h2_sij, s2_ij = get_realspace_hs(self.h2_skmm,
                                         self.s2_kmm,
                                         calc1.ibzk_kc, 
                                         calc1.weight_k,
                                         R_c=R_c)


        h2[:pl2,:pl2] = h2_sii[0]
        h2[pl2:2*pl2,pl2:2*pl2] = h2_sii[0]
        h2[:pl2,pl2:2*pl2] = h2_sij[0]
        tri2full(h2,'U')
        
        s2[:pl2, :pl2] = s2_ii
        s2[pl2:2*pl2, pl2:2*pl2] = s2_ii
        s2[:pl2, pl2:2*pl2] = s2_ij
        tri2full(s2, 'U')

        if calc2.wfs.world.rank == 0:
            print "Dumping lead 2 hamiltonian..."
            fd = file('lead2_'+filename,'wb')
            pickle.dump((h2,s2),fd,2)
            fd.close()

        world.barrier()
        
        del self.atoms_l

        self.update_scat_hamiltonian()
        nbf_m = self.h_skmm.shape[-1]
        nbf = nbf_m + pl1 + pl2
        h = np.zeros((nbf, nbf), complex)
        s = np.zeros((nbf, nbf), complex)
        
        h_mm = self.h_skmm[0,0]
        s_mm = self.s_kmm[0]
        atoms = self.atoms
        remove_pbc(atoms, h_mm, s_mm, self.d)

        h[:2*pl1,:2*pl1] = h1
        h[-2*pl2:,-2*pl2:] = h2
        h[pl1:-pl2,pl1:-pl2] = h_mm

        s[:2*pl1,:2*pl1] = s1
        s[-2*pl2:,-2*pl2:] = s2
        s[pl1:-pl2,pl1:-pl2] = s_mm
  
        if atoms.calc.wfs.world.rank == 0:
            print "Dumping scat hamiltonian..."
            fd = file('scat_'+filename,'wb')
            pickle.dump((h,s),fd,2)
            fd.close()
        world.barrier()

    def update_lead_hamiltonian(self, l, restart=False, savefile=True):
        if not restart:
            self.atoms_l[l] = self.get_lead_atoms(l)
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            kpts = self.atoms_l[0].calc.wfs.ibzk_kc  
            self.npk = kpts.shape[0] / self.ntklead
            self.dimt_lead = self.atoms_l[0].calc.hamiltonian.vt_sG.shape[-1]
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                self.d1_skmm = self.generate_density_matrix('lead_l')        
                if savefile:
                    atoms.calc.write('lead0.gpw')                    
                    self.pl_write('leadhs0', (self.h1_skmm,
                                          self.s1_kmm,
                                          self.d1_skmm,
                                          self.ntklead,
                                          self.dimt_lead))            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                self.d2_skmm = self.generate_density_matrix('lead_r')
                if savefile:
                    atoms.calc.write('lead1.gpw')    
                    self.pl_write('leadhs1', (self.h2_skmm,
                                          self.s2_kmm,
                                          self.d2_skmm,
                                          self.ntklead,
                                          self.dimt_lead))            
        else:
            atoms, calc = restart_gpaw('lead' + str(l) + '.gpw')
            calc.set_positions()
            self.atoms_l[l] = atoms
            if l == 0:        
                (self.h1_skmm,
                 self.s1_kmm,
                 self.d1_skmm,
                 self.ntklead,
                 self.dimt_lead) = self.pl_read('leadhs0')
            elif l == 1:
                (self.h2_skmm,
                 self.s2_kmm,
                 self.d2_skmm,
                 self.ntklead,
                 self.dimt_lead) = self.pl_read('leadhs1')
            kpts = self.atoms_l[0].calc.wfs.ibzk_kc  
            self.npk = kpts.shape[0] / self.ntklead 
        self.nblead = self.h1_skmm.shape[-1]
        
    def update_scat_hamiltonian(self, restart=False, savefile=True):
        if not restart:
            atoms = self.atoms
            atoms.get_potential_energy()
            calc = atoms.calc
            rank = world.rank
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            self.d_skmm = self.generate_density_matrix('scat')
            if savefile:
                calc.write('scat.gpw')
                self.pl_write('scaths', (self.h_skmm,
                                         self.s_kmm,
                                         self.d_skmm))                        
        else:
            atoms, calc = restart_gpaw('scat.gpw')
            calc.set_positions()
            self.atoms = atoms
            self.h_skmm, self.s_kmm, self.d_skmm = self.pl_read('scaths')
        kpts = calc.wfs.ibzk_kc
        self.nbmol = self.h_skmm.shape[-1]
            
    def get_hs(self, atoms):
        calc = atoms.calc
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
        """l: 0, 1 correpsonding to left, right """
        atoms = self.atoms.copy()
        atomsl = atoms[self.pl_atoms[l]]
        atomsl.cell = self.pl_cells[l]
        atomsl.center()
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l, nkpts=35):
        p = self.atoms.calc.input_parameters.copy()
        p['nbands'] = None
        kpts = list(p['kpts'])
        if nkpts == 0:
            kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        else:
            kpts[self.d] = nkpts
        self.ntklead = kpts[self.d]
        p['kpts'] = kpts
        if 'mixer' in p: # XXX Works only if spin-paired
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)
        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return GPAW(**p)

    def read(self, filename):
        h, s = pickle.load(file('scat_' + filename))
        h1, s1 = pickle.load(file('lead1_' + filename))
        h2, s2 = pickle.load(file('lead2_' + filename))
        pl1 = len(h1) / 2 
        pl2 = len(h2) / 2
        self.h_skmm = h[:]
        self.s_kmm = s[:]
        self.h1_skmm = h1[:]
        self.s1_kmm = s1[:]
        self.h2_skmm = h2[:]
        self.s2_kmm = s2[:]
        self.atoms_l[0] = self.get_lead_atoms(0)
        self.atoms_l[1] = self.get_lead_atoms(1)
        
    def negf_prepare(self, scat_restart=False,
                                          lead_restart=False, savefile=True):
        p = self.atoms.calc.input_parameters.copy()           
        self.ntkmol = p['kpts'][self.d]
        self.update_lead_hamiltonian(0, lead_restart, savefile)
        world.barrier()
        self.update_lead_hamiltonian(1, lead_restart, savefile)
        world.barrier()
        if self.extend:
            self.extend_scat()
        self.update_scat_hamiltonian(scat_restart, savefile)
        world.barrier()
        self.nspins = self.h1_skmm.shape[0]
        self.kpts = self.atoms.calc.wfs.ibzk_kc
        self.lead_kpts = self.atoms_l[0].calc.wfs.ibzk_kc
        self.allocate_cpus()
        self.print_info = self.atoms.calc.text

        self.check_edge()
        self.initial_lead(0)
        self.initial_lead(1)
        self.initial_mol()        

        self.edge_density_mm = self.calc_edge_density(self.d_spkmm_ij,
                                                              self.s_pkmm_ij)
        self.edge_charge = np.zeros([self.nspins])
        for i in range(self.nspins):
            for j in range(self.my_npk):
                self.edge_charge[i] += np.trace(self.edge_density_mm[i, j])
        self.kpt_comm.sum(self.edge_charge)
        if world.rank == 0:
            for i in range(self.nspins):  
                total_edge_charge  = self.edge_charge[i] / self.npk
                self.print_info('edge_charge[%d]=%f' % (i, total_edge_charge))
        self.boundary_check()
        del self.atoms_l

    def initial_lead(self, lead):
        nspins = self.nspins
        ntk = self.ntklead
        npk = self.my_npk
        nblead = self.nblead
        kpts = self.my_lead_kpts
        position = [0, 0, 0]
        spk = self.substract_pk
        if lead == 0:
            position[self.d] = 1.0
            self.h1_spkmm = spk(ntk, kpts, self.h1_skmm, 'h')
            self.s1_pkmm = spk(ntk, kpts, self.s1_kmm)
            self.h1_spkmm_ij = spk(ntk, kpts, self.h1_skmm, 'h', position)
            self.s1_pkmm_ij = spk(ntk, kpts, self.s1_kmm, 's', position)
            self.d1_spkmm = spk(ntk, kpts, self.d1_skmm, 'h')
            self.d1_spkmm_ij = spk(ntk, kpts, self.d1_skmm, 'h', position)
        elif lead == 1:
            position[self.d] = -1.0
            self.h2_spkmm = spk(ntk, kpts, self.h2_skmm, 'h')
            self.s2_pkmm = spk(ntk, kpts, self.s2_kmm)
            self.h2_spkmm_ij = spk(ntk, kpts, self.h2_skmm, 'h', position)
            self.s2_pkmm_ij = spk(ntk, kpts, self.s2_kmm, 's', position)            
        else:
            raise TypeError('unkown lead index')

    def initial_mol(self):
        ntk = self.ntkmol
        kpts = self.my_kpts
        position = [0,0,0]
        position[self.d] = 1.0
        self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
        self.s_pkmm = self.substract_pk(ntk, kpts, self.s_kmm)
        self.d_spkmm = self.substract_pk(ntk, kpts, self.d_skmm, 'h')
        self.s_pkmm_ij , self.d_spkmm_ij = self.fill_density_matrix()

    def substract_pk(self, ntk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        npk = self.my_npk
        weight = np.array([1.0 / ntk] * ntk )
        if hors not in 'hs':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / ntk,) + dim[2:]
            pk_mm = np.empty(dim, complex)
            dim = (dim[0],) + (ntk,) + dim[2:]
            tk_mm = np.empty(dim, complex)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / ntk,) + dim[1:]
            pk_mm = np.empty(dim, complex)
            dim = (ntk,) + dim[1:]
            tk_mm = np.empty(dim, complex)
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
        npk = self.npk
        kpts = self.my_lead_kpts
        s_pkmm = self.substract_pk(ntk, kpts, self.s1_kmm, 's', position)
        matmax = np.max(abs(s_pkmm))
        if matmax > tolx:
            self.print_info('Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax)
    
    def calc_edge_density(self, d_spkmm_ij, s_pkmm_ij):
        nspins = self.nspins
        npk = self.my_npk
        nbf = s_pkmm_ij.shape[-1]
        edge_charge_mm = np.zeros([nspins, npk, nbf, nbf])
        for i in range(nspins):
            for j in range(npk):
                edge_charge_mm[i, j] += np.dot(d_spkmm_ij[i, j],
                                               s_pkmm_ij[j].T.conj())
                edge_charge_mm[i, j] += np.dot(d_spkmm_ij[i, j].T.conj(),
                                               s_pkmm_ij[j])
        return edge_charge_mm

    def pick_out_tkpts(self, ntk, kpts):
        npk = self.npk
        tkpts = np.zeros([ntk, 3])
        for i in range(ntk):
            tkpts[i, self.d] = kpts[i, self.d]
        return tkpts

    def generate_density_matrix(self, region):
        ntk = self.ntklead
        npk = self.npk
        if region == 'lead_l':
            calc = self.atoms_l[0].calc
            dim = self.h1_skmm.shape
        elif region == 'lead_r':
            calc = self.atoms_l[1].calc
            dim = self.h2_skmm.shape
        elif region == 'scat':
            calc = self.atoms.calc
            dim = self.h_skmm.shape
            ntk = self.ntkmol
        else:
            raise KeyError('invalid lead index')
        d_skmm = np.empty(dim, complex)
        for kpt in calc.wfs.kpt_u:
            C_nm = kpt.C_nM
            f_nn = np.diag(kpt.f_n)
            d_skmm[kpt.s, kpt.q] = np.dot(C_nm.T.conj(),
                                          np.dot(f_nn, C_nm)) * ntk * npk
        return d_skmm
    
    def fill_density_matrix(self):
        nblead = self.nblead
        s_pkmm_ij = np.zeros(self.s_pkmm.shape, complex)
        s_pkmm_ij[:, -nblead:, :nblead] = self.s1_pkmm_ij
        d_spkmm_ij = np.zeros(self.d_spkmm.shape, complex)
        d_spkmm_ij[:, :, -nblead:, :nblead] = self.d1_spkmm_ij                    
        return s_pkmm_ij, d_spkmm_ij

    def boundary_check(self):
        tol = 5.e-4
        pl1 = self.h1_skmm.shape[-1]
        self.do_shift = False
        matdiff = self.h_spkmm[0, :, :pl1, :pl1] - self.h1_spkmm[0]
        if self.nspins == 2:
             matdiff1 = self.h_spkmm[1, :, :pl1, :pl1] - self.h1_spkmm[1]
             float_diff = np.max(abs(matdiff - matdiff1)) 
             if float_diff > 1e-6:
                  self.print_info('Warning!, float_diff between spins %f' %
                                                             float_diff)
        e_diff = matdiff[0,0,0] / self.s1_pkmm[0,0,0]
        if abs(e_diff) > tol:
            self.print_info('Warning*: hamiltonian boundary difference %f' %
                                                     e_diff)
            self.do_shift = True
            for i in range(self.nspins):
                self.h_spkmm[i] -= e_diff * self.s_pkmm
        self.zero_shift = e_diff 
        matdiff = self.d_spkmm[:, :, :pl1, :pl1] - self.d1_spkmm
        print_diff = np.max(abs(matdiff))
        if print_diff > tol:
            self.print_info('Warning*: density boundary difference %f' % 
                                                      print_diff)
            
    def extend_scat(self):
        lead_atoms_num = len(self.pl_atoms[0])
        atoms_inner = self.atoms.copy()
        atoms_inner.center()
        atoms = self.atoms_l[0] + atoms_inner + self.atoms_l[1]
        atoms.set_pbc(atoms_inner._pbc)
        d = self.d
        cell = self.atoms._cell.copy()
        cell[d] += self.pl_cells[0][d] * 2
        atoms.set_cell(cell)
        for i in range(lead_atoms_num):
            atoms.positions[i, d] -= self.pl_cells[0][d]
        for i in range(-lead_atoms_num, 0):
            atoms.positions[i, d] += self.atoms._cell[d, d]
        atoms.calc = self.atoms.calc
        self.atoms = atoms
        self.atoms.center()

    def get_selfconsistent_hamiltonian(self, bias=0, gate=0,
                                      cal_loc=False, recal_path=False,
                                      verbose=False,  scat_lead=False):
        self.initialize_scf(bias, gate, cal_loc, verbose)  
        self.move_buffer()
        nbmol = self.nbmol_inner
        nspins = self.nspins
        npk = self.my_npk
        den = np.empty([nspins, npk, nbmol, nbmol], complex)
        denocc = np.empty([nspins, npk, nbmol, nbmol], complex)
        if self.cal_loc:
            denloc = np.empty([nspins, npk, nbmol, nbmol], complex)            
        world.barrier()
        #-------get the path --------    
        for s in range(nspins):
            for k in range(npk):      
                den[s, k] = self.get_eqintegral_points(self.intctrl, s, k)
                denocc[s, k] = self.get_neintegral_points(self.intctrl, s, k)
                if self.cal_loc:
                    denloc[s, k] = self.get_neintegral_points(self.intctrl,
                                                              s, k, 'locInt')                    
        #-------begin the SCF ----------         
        self.step = 0
        self.cvgflag = 0
        calc = self.atoms.calc    
        timer = Timer()
        ntk = self.ntkmol
        kpts = self.my_kpts
        spin_coff = 3 - nspins
        max_steps = 200
        while self.cvgflag == 0 and self.step < max_steps:
            if self.master:
                self.print_info('----------------step %d -------------------'
                                                                % self.step)
            self.move_buffer()
            f_spkmm_mol = np.copy(self.h_spkmm_mol)
            temp = self.fcvg.matcvg(self.atoms.calc.hamiltonian.vt_sG,
                                                        self.print_info)
            timer.start('Fock2Den')
            if scat_lead == True:
                self.fill_lead_with_scat()
                self.selfenergies[0].set_bias(0)
                self.selfenergies[1].set_bias(0)
            if recal_path:
                for s in range(nspins):
                    for k in range(npk):
                        den[s, k] = self.get_eqintegral_points(self.intctrl,
                                                                s, k)
                        denocc[s, k] = self.get_neintegral_points(
                                                               self.intctrl,
                                                                s, k)
                        if self.cal_loc:
                            denloc[s, k] = self.get_neintegral_points(
                                                               self.intctrl,
                                                               s, k,
                                                               'locInt')     
                        self.d_spkmm_mol[s, k] = spin_coff * (den[s, k] +
                                                              denocc[s, k])
            else:
                for s in range(nspins):
                    for k in range(npk):
                        self.d_spkmm_mol[s, k] = spin_coff * self.fock2den(
                                                            self.intctrl,
                                                            f_spkmm_mol,
                                                            s, k)
            timer.stop('Fock2Den')
            if self.verbose and self.master:
                self.print_info('Fock2Den', timer.gettime('Fock2Den'),
                                                                    'second')
            self.cvgflag = self.fcvg.bcvg and self.dcvg.bcvg
            timer.start('Den2Fock')            
            self.h_skmm = self.den2fock(self.d_spkmm)
            self.cvgflag = self.fcvg.bcvg and self.dcvg.bcvg
            timer.stop('Den2Fock')
            if self.verbose and self.master:
                self.print_info('Den2Fock', timer.gettime('Den2Fock'),
                                                                   'second')
         
            self.h_spkmm = self.substract_pk(ntk, kpts, self.h_skmm, 'h')
            if self.do_shift:
                for i in range(nspins):
                    self.h_spkmm[i] -= (self.zero_shift -
                                                     self.gate) * self.s_pkmm
            self.step +=  1
        self.FF = self.atoms.calc.get_forces(self.atoms)
        return 1
 
    def initialize_scf(self, bias, gate, cal_loc, verbose):
        self.verbose = verbose
        self.master = (world.rank==0)
        self.bias = bias
        self.gate = gate
        self.cal_loc = cal_loc and self.bias != 0
        self.kt = self.atoms.calc.occupations.kT * Hartree
        self.fermi = 0
        self.current = 0
        if self.nblead == self.nbmol:
            self.buffer = 0
        else:
            self.buffer = self.nblead
        self.intctrl = IntCtrl(self.kt, self.fermi, self.bias)
        self.fcvg = CvgCtrl(self.master)
        self.dcvg = CvgCtrl(self.master)
        inputinfo = {'fasmethodname':'SCL_None', 'fmethodname':'CVG_None',
                     'falpha': 0.1, 'falphascaling':0.1, 'ftol':1e-5,
                     'fallowedmatmax':1e-4, 'fndiis':10, 'ftolx':1e-5,
                     'fsteadycheck': False,
                     'dasmethodname':'SCL_None', 'dmethodname':'CVG_Broydn',
                     'dalpha': 0.1, 'dalphascaling':0.1, 'dtol':1e-4,
                     'dallowedmatmax':1e-4, 'dndiis': 6, 'dtolx':1e-5,
                     'dsteadycheck': False}
        self.fcvg(inputinfo, 'f', self.dcvg)
        self.dcvg(inputinfo, 'd', self.fcvg)
        
        self.selfenergies = [LeadSelfEnergy((self.h1_spkmm[0,0],
                                                            self.s1_pkmm[0]), 
                                            (self.h1_spkmm_ij[0,0],
                                                         self.s1_pkmm_ij[0]),
                                            (self.h1_spkmm_ij[0,0],
                                                         self.s1_pkmm_ij[0]),
                                             0),
                             LeadSelfEnergy((self.h2_spkmm[0,0],
                                                            self.s2_pkmm[0]), 
                                            (self.h2_spkmm_ij[0,0],
                                                         self.s2_pkmm_ij[0]),
                                            (self.h2_spkmm_ij[0,0],
                                                         self.s2_pkmm_ij[0]),
                                             0)]

        self.greenfunction = GreenFunction(selfenergies=self.selfenergies,
                                           H=self.h_spkmm[0,0],
                                           S=self.s_pkmm[0], eta=0)

        self.selfenergies[0].set_bias(self.bias / 2.0)
        self.selfenergies[1].set_bias(-self.bias / 2.0)
       
        self.eqpathinfo = []
        self.nepathinfo = []
        if self.cal_loc:
            self.locpathinfo = []
       
        for s in range(self.nspins):
            self.eqpathinfo.append([])
            self.nepathinfo.append([])
            if self.cal_loc:
                self.locpathinfo.append([])                
            if self.cal_loc:
                self.locpathinfo.append([])
            for k in self.my_pk:
                self.eqpathinfo[s].append(PathInfo('eq'))
                self.nepathinfo[s].append(PathInfo('ne'))    
                if self.cal_loc:
                    self.locpathinfo[s].append(PathInfo('eq'))
        if self.master:
            self.print_info('------------------Transport SCF-----------------------')
            self.print_info('Mixer: %s,  Mixing factor: %s,  tol_Ham=%f, tol_Den=%f ' % (
                                      inputinfo['dmethodname'],
                                      inputinfo['dalpha'],
                                      inputinfo['ftol'],
                                      inputinfo['dtol']))
            self.print_info('bias = %f (V), gate = %f (V)' % (bias, gate))

     
    def get_eqintegral_points(self, intctrl, s, k):
        maxintcnt = 100
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol])
        
        self.zint = [0] * maxintcnt
        self.fint = []
        self.tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)
        self.cntint = -1

        self.selfenergies[0].h_ii = self.h1_spkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_pkmm[k]
        self.selfenergies[0].h_ij = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_pkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_pkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_spkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_pkmm[k]
        self.selfenergies[1].h_ij = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_pkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_pkmm_ij[k]
        
        self.greenfunction.H = self.h_spkmm_mol[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]
       
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
                self.print_info('Warning: SGF not Found. eqzgp[%d]= %f %f'
                                                        %(i, elist[i],sgferr))
        flist = self.fint[:]
        siglist = [[],[]]
        for i, num in zip(range(fcnt), sgforder):
            flist[i] = self.fint[num]
 
        sigma= np.empty([nblead, nblead], complex)
        for i in [0, 1]:
            for j, num in zip(range(fcnt), sgforder):
                sigma = self.tgtint[i, num]
                siglist[i].append(sigma)
        self.eqpathinfo[s][k].add(elist, wlist, flist, siglist)    
   
        del self.fint, self.tgtint, self.zint
        return den 
    
    def get_neintegral_points(self, intctrl, s, k, calcutype='neInt'):
        intpathtol = 1e-8
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        maxintcnt = 100

        self.zint = [0] * maxintcnt
        self.tgtint = np.empty([2, maxintcnt, nblead, nblead], complex)

        self.selfenergies[0].h_ii = self.h1_spkmm[s, k]
        self.selfenergies[0].s_ii = self.s1_pkmm[k]
        self.selfenergies[0].h_ij = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_ij = self.s1_pkmm_ij[k]
        self.selfenergies[0].h_im = self.h1_spkmm_ij[s, k]
        self.selfenergies[0].s_im = self.s1_pkmm_ij[k]

        self.selfenergies[1].h_ii = self.h2_spkmm[s, k]
        self.selfenergies[1].s_ii = self.s2_pkmm[k]
        self.selfenergies[1].h_ij = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_ij = self.s2_pkmm_ij[k]
        self.selfenergies[1].h_im = self.h2_spkmm_ij[s, k]
        self.selfenergies[1].s_im = self.s2_pkmm_ij[k]
        
        self.greenfunction.H = self.h_spkmm_mol[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]

        if calcutype == 'neInt' or calcutype == 'neVirInt':
            for n in range(1, len(intctrl.neintpath)):
                self.cntint = -1
                self.fint = [[],[]]
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
                            self.print_info('--Warning: SGF not found, \
                                        nezgp[%d]=%f %f' % (i, zgp[i],sgferr))
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
                flist = [[],[]]
                siglist = [[],[]]
                sigma= np.empty([nblead, nblead], complex)
                for i in [0, 1]:
                    for j, num in zip(range(nefcnt), sgforder):
                        fermi_factor = self.fint[i][num]
                        sigma = self.tgtint[i, num]
                        flist[i].append(fermi_factor)    
                        siglist[i].append(sigma)
                if calcutype == 'neInt':
                    self.nepathinfo[s][k].add(zgp, wgp, flist, siglist)
                elif calcutype == 'neVirInt':
                    self.virpathinfo[s][k].add(zgp, wgp, flist, siglist)
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
                    self.print_info('Warning: SGF not Found. eqzgp[%d]= %f'
                                                               %(i, elist[i]))
            flist = self.fint[:]
            siglist = [[],[]]
            for i, num in zip(range(fcnt), sgforder):
                flist[i] = self.fint[num]
            sigma= np.empty([nblead, nblead], complex)
            for i in [0, 1]:
                for j, num in zip(range(fcnt), sgforder):
                    sigma = self.tgtint[i, num]
                    siglist[i].append(sigma)
            self.locpathinfo[s][k].add(elist, wlist, flist, siglist)           
        neq = np.trace(np.dot(den, self.greenfunction.S))
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
        nlead = 2
        nblead = self.nblead
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
            gamma = np.zeros([nlead, nbmol, nbmol], complex)
            if self.cntint + 1 >= len(self.zint):
                self.zint += [0] * stepintcnt
                tmp = self.tgtint.shape[1]
                tmptgtint = np.copy(self.tgtint)
                self.tgtint = np.empty([2, tmp + stepintcnt, nblead, nblead],
                                                                      complex)
                self.tgtint[:, :tmp] = tmptgtint
                self.tgtint[:, tmp:tmp + stepintcnt] = np.zeros([2,
                                                 stepintcnt, nblead, nblead])
            self.cntint += 1
            self.zint[self.cntint] = zp[i]

            for j in [0, 1]:
                self.tgtint[j, self.cntint] = self.selfenergies[j](zp[i])
            
            sigma[:nblead, :nblead] += self.tgtint[0, self.cntint]
            sigma[-nblead:, -nblead:] += self.tgtint[1, self.cntint]
            gamma[0, :nblead, :nblead] += self.selfenergies[0].get_lambda(
                                                                        zp[i])
            gamma[1, -nblead:, -nblead:] += self.selfenergies[1].get_lambda(
                                                                        zp[i])
            gr = self.greenfunction.calculate(zp[i], sigma)       
        
            # --ne-Integral---
            if calcutype == 'neInt':
                gammaocc = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    self.fint[n].append(  fermidistribution(zp[i] -
                                         intctrl.leadfermi[n], intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                             intctrl.minfermi, intctrl.kt) )
                    gammaocc += gamma[n] * self.fint[n][self.cntint]
                aocc = np.dot(gr, gammaocc)
                aocc = np.dot(aocc, gr.T.conj())
               
                gfunc[i] = aocc
            elif calcutype == 'neVirInt':
                gammavir = np.zeros([nbmol, nbmol], complex)
                for n in [0, 1]:
                    self.fint[n].append(fermidistribution(zp[i] -
                                         intctrl.maxfermi, intctrl.kt) - 
                                         fermidistribution(zp[i] -
                                            intctrl.leadfermi[n], intctrl.kt))
                    gammavir += gamma[n] * self.fint[n][self.cntint]
                avir = np.dot(gr, gammavir)
                avir = np.dot(avir, gr.T.conj())
                gfunc[i] = avir
            # --local-Integral--
            elif calcutype == 'locInt':
                # fmax-fmin
                self.fint.append( fermidistribution(zp[i] -
                                    intctrl.maxfermi, intctrl.kt) - \
                                    fermidistribution(zp[i] -
                                    intctrl.minfermi, intctrl.kt) )
                gfunc[i] = gr * self.fint[self.cntint]
 
            # --res-Integral --
            elif calcutype == 'resInt':
                self.fint.append(-2.j * np.pi * intctrl.kt)
                gfunc += gr * self.fint[self.cntint]
            #--eq-Integral--
            else:
                if intctrl.kt <= 0:
                    self.fint.append(1.0)
                else:
                    self.fint.append(fermidistribution(zp[i] -
                                                intctrl.minfermi, intctrl.kt))
                gfunc[i] = gr * self.fint[self.cntint]    
        return gfunc        
    
    def fock2den(self, intctrl, f_spkmm, s, k):
        nblead = self.nblead
        nbmol = self.nbmol_inner
        den = np.zeros([nbmol, nbmol], complex)
        denocc = np.zeros([nbmol, nbmol], complex)
        if self.cal_loc:
            denvir = np.zeros([nbmol, nbmol], complex)
            denloc = np.zeros([nbmol, nbmol], complex)
        sigmatmp = np.zeros([nblead, nblead], complex)

        eqzp = self.eqpathinfo[s][k].energy
        self.greenfunction.H = f_spkmm[s, k]
        self.greenfunction.S = self.s_pkmm_mol[k]
        
        for i in range(len(eqzp)):
            sigma = np.zeros([nbmol, nbmol], complex)  
            sigma[:nblead, :nblead] += self.eqpathinfo[s][k].sigma[0][i]
            sigma[-nblead:, -nblead:] += self.eqpathinfo[s][k].sigma[1][i]
            gr = self.greenfunction.calculate(eqzp[i], sigma)
            fermifactor = self.eqpathinfo[s][k].fermi_factor[i]
            weight = self.eqpathinfo[s][k].weight[i]
            den += gr * fermifactor * weight
        den = 1.j * (den - den.T.conj()) / np.pi / 2

        if self.cal_loc:
            eqzp = self.locpathinfo[s][k].energy
            for i in range(len(eqzp)):
                sigma = np.zeros([nbmol, nbmol], complex)  
                sigma[:nblead, :nblead] += self.locpathinfo[s][k].sigma[0][i]
                sigma[-nblead:, -nblead:] += self.locpathinfo[s][k].sigma[1][i]
                gr = self.greenfunction.calculate(eqzp[i], sigma)
                fermifactor = self.locpathinfo[s][k].fermi_factor[i]
                weight = self.locpathinfo[s][k].weight[i]
                denloc += gr * fermifactor * weight
            denloc = 1.j * (denloc - denloc.T.conj()) / np.pi / 2

        nezp = self.nepathinfo[s][k].energy
        
        intctrl = self.intctrl
        for i in range(len(nezp)):
            sigma = np.zeros([nbmol, nbmol], complex)
            sigmalesser = np.zeros([nbmol, nbmol], complex)
            sigma[:nblead, :nblead] += self.nepathinfo[s][k].sigma[0][i]
            sigma[-nblead:, -nblead:] += self.nepathinfo[s][k].sigma[1][i]    
            gr = self.greenfunction.calculate(nezp[i], sigma)
            fermifactor = np.real(self.nepathinfo[s][k].fermi_factor[0][i])
           
            sigmatmp = self.nepathinfo[s][k].sigma[0][i]
            sigmalesser[:nblead, :nblead] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())
            fermifactor = np.real(self.nepathinfo[s][k].fermi_factor[1][i])

            sigmatmp = self.nepathinfo[s][k].sigma[1][i] 
            sigmalesser[-nblead:, -nblead:] += 1.0j * fermifactor * (
                                          sigmatmp - sigmatmp.T.conj())       
            glesser = np.dot(sigmalesser, gr.T.conj())
            glesser = np.dot(gr, glesser)
            weight = self.nepathinfo[s][k].weight[i]            
            denocc += glesser * weight / np.pi / 2
            if self.cal_loc:
                sigmalesser = np.zeros([nbmol, nbmol], complex)
                fermifactor = fermidistribution(nezp[i] -
                                              intctrl.maxfermi, intctrl.kt)-\
                              fermidistribution(nezp[i] -
                                             intctrl.leadfermi[0], intctrl.kt)
                fermifactor = np.real(fermifactor)
                sigmatmp = self.nepathinfo[s][k].sigma[0][i]
                sigmalesser[:nblead, :nblead] += 1.0j * fermifactor * (
                                                   sigmatmp - sigmatmp.T.conj())
                fermifactor = fermidistribution(nezp[i] -
                                         intctrl.maxfermi, intctrl.kt) -  \
                              fermidistribution(nezp[i] -
                                         intctrl.leadfermi[1], intctrl.kt)
                fermifactor = np.real(fermifactor)
                sigmatmp = self.nepathinfo[s][k].sigma[1][i] 
                sigmalesser[-nblead:, -nblead:] += 1.0j * fermifactor * (
                                                sigmatmp - sigmatmp.T.conj())       
                glesser = np.dot(sigmalesser, gr.T.conj())
                glesser = np.dot(gr, glesser)
                weight = self.nepathinfo[s][k].weight[i]            
                denvir += glesser * weight / np.pi / 2
        
        if self.cal_loc:
            weight_mm = self.integral_diff_weight(denocc, denvir,
                                                                 'transiesta')
            diff = (denloc - (denocc + denvir)) * weight_mm
            den += denocc + diff
            percents = np.sum( diff * diff ) / np.sum( denocc * denocc )
            self.print_info('local percents %f' % percents)
        else:
            den += denocc
        den = (den + den.T.conj()) / 2
        return den    

    def den2fock(self, d_pkmm):
        self.get_density(d_pkmm)
        calc = self.atoms.calc
        calc.update_kinetic()
        density = calc.density
        calc.hamiltonian.update(density)
        linear_potential = self.get_linear_potential()
        calc.hamiltonian.vt_sG += linear_potential
        xcfunc = calc.hamiltonian.xc.xcfunc
        calc.Enlxc = xcfunc.get_non_local_energy()
        calc.Enlkin = xcfunc.get_non_local_kinetic_corrections() 
        h_skmm, s_kmm = self.get_hs(self.atoms)
 
        return h_skmm
    
    def get_density(self,d_spkmm):
        #Calculate pseudo electron-density based on green function.
        calc = self.atoms.calc
        density = calc.density
        wfs = calc.wfs

        nspins = self.nspins
        ntk = self.ntkmol
        npk = self.my_npk
        nbmol = self.nbmol
        relate_layer_num = 3
        dr_mm = np.zeros([nspins, npk, relate_layer_num,
                                                 nbmol, nbmol], complex)
        qr_mm = np.zeros([nspins, npk, nbmol, nbmol])
        
        ind = (relate_layer_num - 1) / 2
        for s in range(nspins):
            for i in range(relate_layer_num):
                for j in range(self.my_npk):
                    if i == ind - 1:
                        dr_mm[s, j, i] = self.d_spkmm_ij[s, j].T.conj()
                    elif i == ind:
                        dr_mm[s, j, i] = np.copy(d_spkmm[s, j])
                        qr_mm[s, j] += np.dot(dr_mm[s, j, i],
                                                    self.s_pkmm[j]) 
                    elif i == ind + 1:
                        dr_mm[s, j, i]= np.copy(self.d_spkmm_ij[s, j])
        qr_mm += self.edge_density_mm
        world.barrier()
        self.kpt_comm.sum(qr_mm)
        qr_mm /= self.npk
     
        if self.master:
            qr_mm = np.sum(np.sum(qr_mm, axis=0), axis=0)
            natom_inlead = len(self.pl_atoms[0])
            nb_atom = self.nblead / natom_inlead
            pl1 = self.buffer + self.nblead
            natom_print = pl1 / nb_atom 
            edge_charge0 = np.diag(qr_mm[:pl1,:pl1])
            edge_charge1 = np.diag(qr_mm[-pl1:, -pl1:])
            edge_charge0.shape = (natom_print, nb_atom)
            edge_charge1.shape = (natom_print, nb_atom)
            edge_charge0 = np.sum(edge_charge0,axis=1)
            edge_charge1 = np.sum(edge_charge1,axis=1)
            self.print_info('***charge distribution at edges***')
            if self.verbose:
                info = []
                for i in range(natom_print):
                    info.append('--' +  str(edge_charge0[i])+'--')
                self.print_info(info)
                info = []
                for i in range(natom_print):
                    info.append('--' +  str(edge_charge1[i])+'--')
                self.print_info(info)
            else:
                edge_charge0.shape = (natom_print / natom_inlead, natom_inlead)
                edge_charge1.shape = (natom_print / natom_inlead, natom_inlead)                
                edge_charge0 = np.sum(edge_charge0,axis=1)
                edge_charge1 = np.sum(edge_charge1,axis=1)
                info = ''
                for i in range(natom_print / natom_inlead):
                    info += '--' +  str(edge_charge0[i]) + '--'
                info += '---******---'
                for i in range(natom_print / natom_inlead):
                    info += '--' +  str(edge_charge1[i]) + '--'
                self.print_info(info)
            self.print_info('***total charge***')
            self.print_info(np.trace(qr_mm))            

        rvector = np.zeros([relate_layer_num, 3])
        tkpts = self.pick_out_tkpts(ntk, self.my_kpts)
        for i in range(relate_layer_num):
            rvector[i, self.d] = i - (relate_layer_num - 1) / 2
        
        self.d_skmm.shape = (nspins, npk, ntk, nbmol, nbmol)
        for s in range(nspins):
            if ntk != 1:
                for i in range(ntk):
                    for j in range(npk):
                        self.d_skmm[s, j, i] = get_kspace_hs(None, dr_mm[s, j, :],
                                                             rvector, tkpts[i])
                        self.d_skmm[s, j, i] /=  ntk * self.npk 
            else:
                for j in range(npk):
                    self.d_skmm[s, j, 0] =  dr_mm[s, j, 1]
                    self.d_skmm[s, j, 0] /= self.npk 
        self.d_skmm.shape = (nspins, ntk * npk, nbmol, nbmol)

        for kpt in calc.wfs.kpt_u:
            kpt.rho_MM = self.d_skmm[kpt.s, kpt.q]
        density.update(wfs)
        if self.step > 0:
            self.diff = density.mixer.get_charge_sloshing()
            if self.step == 1:
                self.min_diff = self.diff
            elif self.diff < self.min_diff:
                self.min_diff = self.diff
                self.output('step.dat')
            self.print_info('dcvg: dmatmax = %f   tol=%f' % (self.diff,
                                                  calc.scf.max_density_error))
            if self.diff < calc.scf.max_density_error:
                self.dcvg.bcvg = True
        return density

    def calc_total_charge(self, d_spkmm):
        nbmol = self.nbmol 
        qr_mm = np.empty([self.nspins, self.my_npk, nbmol, nbmol])
        for i in range(self.nspins):  
            for j in range(self.my_npk):
                qr_mm[i,j] = np.dot(d_spkmm[i, j],self.s_pkmm[j])
        Qmol = np.trace(np.sum(np.sum(qr_mm, axis=0), axis=0))
        Qmol += np.sum(self.edge_charge)
        Qmol = self.kpt_comm.sum(Qmol) / self.npk
        return Qmol        

    def get_linear_potential(self):
        calc = self.atoms.calc
        linear_potential = np.zeros(calc.hamiltonian.vt_sG.shape)
        
        dimt = linear_potential.shape[-1]
        dimp = linear_potential.shape[1:3]
        dimt_lead = self.dimt_lead
        if self.nblead == self.nbmol:
            buffer_dim = 0
        else:
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
        return linear_potential
    
    def output(self, filename):
       out_matrix={}
       matrix = (self.h_skmm, self.d_skmm, self.s_kmm)
       out_matrix[str(world.rank)] = matrix 
       world.barrier()

       if world.rank == 0:
           fd = file(filename, 'wb')
           pickle.dump((out_matrix,
                        self.kpts,
                        self.bias,
                        self.gate,
                        self.intctrl,
                        self.eqpathinfo,
                        self.nepathinfo,
                        self.kpts,
                        self.lead_kpts,
                        self.current,
                        self.step,
                        self.cvgflag
                        ), fd, 2)
           fd.close()
      
    def input(self, filename):
        if world.rank == 0:
            fd = file(filename, 'r')
            (in_matrix,
             self.kpts,
             self.bias,
             self.gate,
             self.intctrl,
             self.eqpathinfo,
             self.nepathinfo,
             self.kpts,
             self.lead_kpts,
             self.current,
             self.step,
             self.cvgflag) = pickle.load(fd)
            an = len(in_matrix)
            dimh = in_matrix['0'][0].shape
            dims = in_matrix['0'][2].shape
            npq = dims[0]
            dimh = (dimh[0],) + (dimh[1] * an,) + dimh[2:]
            dims = (dims[0] * an,) + dims[1:]
            self.h_skmm = np.empty(dimh, complex)
            self.d_skmm = np.empty(dimh, complex)
            self.s_kmm = np.empty(dims, complex)
            for i in range(an):
                temp = i * npq
                self.h_skmm[:, temp:temp + npq] = in_matrix[str(i)][0]
                self.d_skmm[:, temp:temp + npq] = in_matrix[str(i)][1]
                self.s_kmm[temp:temp + npq] = in_matrix[str(i)][2]
            fd.close()
        world.barrier()
    
    def initial_analysis(self, filename):
        self.input(filename)
        (self.h1_skmm,
                 self.s1_kmm,
                 self.d1_skmm,
                 self.ntklead,
                 self.dimt_lead) = self.pl_read('leadhs0', collect=True)
        (self.h2_skmm,
                 self.s2_kmm,
                 self.d2_skmm,
                 self.ntklead,
                 self.dimt_lead) = self.pl_read('leadhs1', collect=True)
        self.nspins = self.h1_skmm.shape[0]
        self.npk = self.h1_skmm.shape[1] / self.ntklead
        self.ntkmol = self.h_skmm.shape[1] / self.npk
        self.nblead = self.h1_skmm.shape[-1]
        self.nbmol = self.h_skmm.shape[-1]
        self.allocate_cpus()
        self.initial_lead(0)
        self.initial_lead(1)
        self.initial_mol()
      
    def set_calculator(self, e_points):
        from ase.transport.calculators import TransportCalculator
     
        h_scat = self.h_spkmm[0,0]
        h_lead1 = self.double_size(self.h1_spkmm[0,0],
                                   self.h1_spkmm_ij[0,0])
        h_lead2 = self.double_size(self.h2_spkmm[0,0],
                                   self.h2_spkmm_ij[0,0])
       
        s_scat = self.s_pkmm[0]
        s_lead1 = self.double_size(self.s1_pkmm[0], self.s1_pkmm_ij[0])
        s_lead2 = self.double_size(self.s2_pkmm[0], self.s2_pkmm_ij[0])
        
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
        
    def plot_v(self, vt=None, tit=None, ylab=None):
        import pylab
        if vt == None:
            if hasattr(self, 'vt_sG'):
                vt = self.vt_sG
            else:
                vt = self.atoms.calc.hamiltonian.vt_sG
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
            if hasattr(self, 'nt_sG'):
                nt = self.nt_sG
            else:
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
    
    def move_buffer(self):
        self.nbmol_inner = self.nbmol - 2 * self.buffer
        pl1 = self.buffer
        if pl1 == 0:
            self.h_spkmm_mol = self.h_spkmm
            self.d_spkmm_mol = self.d_spkmm
            self.s_pkmm_mol = self.s_pkmm
        else:
            self.h_spkmm_mol = self.h_spkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.d_spkmm_mol = self.d_spkmm[:, :, pl1: -pl1, pl1:-pl1]
            self.s_pkmm_mol = self.s_pkmm[:, pl1: -pl1, pl1:-pl1]
            
    def allocate_cpus(self):
        rank = world.rank
        size = world.size
        npk = self.npk
        npk_each = npk / size
        r0 = rank * npk_each
        self.my_pk = np.arange(r0, r0 + npk_each)
        self.my_npk = npk_each
        self.kpt_comm = world.new_communicator(np.arange(size))

        self.my_kpts = np.empty((npk_each * self.ntkmol, 3), complex)
        kpts = self.kpts
        for i in range(self.ntkmol):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_kpts[j * self.ntkmol + i] = kpts[k * self.ntkmol + i]        

        self.my_lead_kpts = np.empty((npk_each * self.ntklead, 3), complex)
        kpts = self.lead_kpts
        for i in range(self.ntklead):
            for j, k in zip(range(npk_each), self.my_pk):
                self.my_lead_kpts[j * self.ntklead + i] = kpts[
                                                        k * self.ntklead + i]         

    def integral_diff_weight(self, denocc, denvir, method='transiesta'):
        if method=='transiesta':
            eta = 1e-16
            weight = denocc * denocc.conj() / (denocc * denocc.conj() +
                                               denvir * denvir.conj() + eta)
        return weight

    def pl_write(self, filename, matlist):
        if type(matlist)!= 'tuple':
            matlist = (matlist,)
            nmat = 1
        else:
            nmat = len(matlist)
        total_matlist = []
        for i in range(nmat):
            if type(matlist[i]) == 'ndarray':  
                dim = matlist[i].shape
                dim = (world.size,) + dim[:] 
                total_mat = np.zeros(dim, dtype=mat.dtype)
                total_mat[world.rank] = matlist[i]
                self.kpt_comm.sum(total_mat)
                total_matlist.append(total_mat)
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
        total_matlist = total_matlist[0]
        matlist = []
        nmat = len(total_matlist)        
        for i in range(nmat):
            if type(total_matlist[i]) == 'ndarray':
                if not collect:
                    matlist.append(total_matlist[i][world.rank])
                else:
                    matlist.append(np.sum(total_matlist[i], axis=0))
            else:
                matlist.append(total_matlist[i])
        return matlist

    def adjust_citeria(self):
        calc = self.atoms.calc
        p = calc.input_parameters.copy()
        if world.size == 1:
            k_queue = range(3, 10)
        else:
            k_queue = [4, 8, 16, 20, 24, 28, 32]
        tol = 1e-5
        h_diff = 1
        d_diff = 1
        num = 0
        
        while h_diff > tol and num < 7:
            kpts = list(p['kpts'])
            kpts[0] = k_queue[num]
            kpts[1] = k_queue[num]
            p['kpts'] = kpts
            p['usesymm'] = True
            self.atoms.calc = GPAW(**p)
            
            self.update_lead_hamiltonian(0, restart=False, savefile=False)
            self.update_scat_hamiltonian(restart=False, savefile=False)
            calc = self.atoms.calc
            calc1 = self.atoms_l[0].calc
            ## attention parallel
            h, s = get_realspace_hs(self.h_skmm, self.s_kmm,
                                calc.wfs.ibzk_kc, calc.wfs.weight_k, [0,0,0])
            d, s = get_realspace_hs(self.d_skmm, self.s_kmm,
                                calc.wfs.ibzk_kc, calc.wfs.weight_k, [0,0,0])
            h1, s1 = get_realspace_hs(self.h1_skmm, self.s1_kmm,
                              calc1.wfs.ibzk_kc, calc1.wfs.weight_k, [0,0,0])
            d1, s1 = get_realspace_hs(self.d1_skmm, self.s1_kmm,
                              calc1.wfs.ibzk_kc, calc1.wfs.weight_k, [0,0,0])
            nblead = self.nblead    
            h_diff = np.max(abs(h[:, :nblead, :nblead] - h1))
            s_diff = np.max(abs(s[:nblead, :nblead] - s1))
            d_diff = np.max(abs(d[:, :nblead, :nblead] -d1))
            print 'h, s, d ', h_diff, s_diff, d_diff
            num += 1
        if num == 7:
            print 'Warning!, the K sampling is not enough  diff= %f' % h_diff
        
        p['usesymm'] = None
        p1['usesymm'] = None
        
        calc = GPAW(**p)
        calc1 = GPAW(**p1)
    
    def fill_lead_with_scat(self):
        nblead = self.nblead
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
                
