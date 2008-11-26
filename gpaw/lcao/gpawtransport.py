from ase import Atoms, Atom, monkhorst_pack, Hartree
import ase
from gpaw import GPAW, Mixer, restart
from gpaw.lcao.tools import get_realspace_hs, get_kspace_hs,tri2full, remove_pbc
import pickle
import numpy as npy
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize
from ase.transport.tools import mytextwrite1, dagger

class GPAWTransport:
    
    def __init__(self, atoms, pl_atoms, pl_cells, d=0):
        self.atoms = atoms
        if not self.atoms.calc.initialized:
            self.atoms.calc.initialize(atoms)
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

    def write_left_lead(self,filename):
        self.update_lead_hamiltonian(0)

    def write(self, filename):
        self.update_lead_hamiltonian(0)

        pl1 = self.h1_skmm.shape[-1]
        h1 = npy.zeros((2*pl1, 2 * pl1), complex)
        s1 = npy.zeros((2*pl1, 2 * pl1), complex)

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
        
        if calc1.master:
            print "Dumping lead 1 hamiltonian..."
            fd = file('lead1_' + filename, 'wb')
            pickle.dump((h1, s1), fd, 2)
            fd.close()

        world.barrier()
        
        self.update_lead_hamiltonian(1) 
        pl2 = self.h2_skmm.shape[-1]
        h2 = npy.zeros((2 * pl2, 2 * pl2), complex)
        s2 = npy.zeros((2 * pl2, 2 * pl2), complex)

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
        
        s2[:pl2,:pl2] = s2_ii
        s2[pl2:2*pl2,pl2:2*pl2] = s2_ii
        s2[:pl2,pl2:2*pl2] = s2_ij
        tri2full(s2,'U')

        if calc2.master:
            print "Dumping lead 2 hamiltonian..."
            fd = file('lead2_'+filename,'wb')
            pickle.dump((h2,s2),fd,2)
            fd.close()

        world.barrier()
        
        del self.atoms_l

        self.update_scat_hamiltonian()
        nbf_m = self.h_skmm.shape[-1]
        nbf = nbf_m + pl1 + pl2
        h = npy.zeros((nbf, nbf), complex)
        s = npy.zeros((nbf, nbf), complex)
        
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
  
        if atoms.calc.master:
            print "Dumping scat hamiltonian..."
            fd = file('scat_'+filename,'wb')
            pickle.dump((h,s),fd,2)
            fd.close()
        world.barrier()

    def update_lead_hamiltonian(self, l, flag=0):
        # flag: 0 for calculation, 1 for read from file
        if flag == 0:
            self.atoms_l[l] = self.get_lead_atoms(l)
            atoms = self.atoms_l[l]
            atoms.get_potential_energy()
            atoms.calc.write('lead' + str(l) + '.gpw')
            if l == 0:
                self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
                fd = file('leadhs0','wb')
                pickle.dump((self.h1_skmm, self.s1_kmm), fd, 2)            
                fd.close()            
            elif l == 1:
                self.h2_skmm, self.s2_kmm = self.get_hs(atoms)
                fd = file('leadhs1','wb')
                pickle.dump((self.h2_skmm, self.s2_kmm), fd, 2)
                fd.close()
        else:
            atoms, calc = restart('lead' + str(l) + '.gpw')
            self.atoms_l[l] = atoms
            if l == 0:        
                fd = file('leadhs0','r') 
                self.h1_skmm, self.s1_kmm = pickle.load(fd)
                fd.close()
            elif l == 1:
                fd = file('leadhs1','r') 
                self.h2_skmm, self.s2_kmm = pickle.load(fd)
                fd.close()

    def update_scat_hamiltonian(self, flag=0):
        # flag: 0 for calculation, 1 for read from file
        if flag == 0:
            atoms = self.atoms
            atoms.get_potential_energy()
            atoms.calc.write('scat.gpw')
            self.h_skmm, self.s_kmm = self.get_hs(atoms)
            fd = file('scaths', 'wb')
            pickle.dump((self.h_skmm, self.s_kmm), fd, 2)
            fd.close()
            calc0 = atoms.calc
            fd = file('nct_G.dat', 'wb')
            pickle.dump(calc0.density.nct_G, fd, 2)
            fd.close()
            self.nct_G = npy.copy(calc0.density.nct_G)
                        
        else :
            atoms, calc = restart('scat.gpw')
            self.atoms = atoms
            fd = file('scaths', 'r')
            self.h_skmm, self.s_kmm = pickle.load(fd)
            fd.close()
            fd = file('nct_G.dat', 'r')
            self.nct_G = pickle.load(fd)
            fd.close()
            

    def get_hs(self, atoms):
        calc = atoms.calc
        Ef = calc.get_fermi_level()
        print'fermi_level'
        print Ef
        eigensolver = calc.eigensolver
        ham = calc.hamiltonian
        Vt_skmm = eigensolver.Vt_skmm
        ham.calculate_effective_potential_matrix(Vt_skmm)
        ibzk_kc = calc.ibzk_kc
        nkpts = len(ibzk_kc)
        nspins = calc.nspins
        weight_k = calc.weight_k
        nao = calc.nao
        h_skmm = npy.zeros((nspins, nkpts, nao, nao), complex)
        s_kmm = npy.zeros((nkpts, nao, nao), complex)
        for k in range(nkpts):
            s_kmm[k] = ham.S_kmm[k]
            tri2full(s_kmm[k])
            for s in range(nspins):
                h_skmm[s,k] = calc.eigensolver.get_hamiltonian_matrix(ham,
                                                                      k=k,
                                                                      s=s)
                tri2full(h_skmm[s, k])
                h_skmm[s,k] *= Hartree
                #h_skmm[s,k] -= Ef * s_kmm[k]

        return h_skmm, s_kmm

    def get_lead_atoms(self, l):
        """l: 0, 1 correpsonding to left, right """
        atoms = self.atoms.copy()
        atomsl = Atoms(pbc=atoms.pbc, cell=self.pl_cells[l])
    
        for a in self.pl_atoms[l]:
            atomsl.append(atoms[a])
       
        atomsl.center()
        atomsl.set_calculator(self.get_lead_calc(l))
        return atomsl

    def get_lead_calc(self, l):
        p = self.atoms.calc.input_parameters.copy()
        p['nbands'] = None
        kpts = list(p['kpts'])
        #kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        kpts[self.d] = 3
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
        #self.atoms_l[0] = self.get_lead_atoms(0)
        #self.atoms_l[1] = self.get_lead_atoms(1)
        
    def prepare(self, filename, flag=0):
        # flag: 0 for calculation, 1 for read from file
        self.update_lead_hamiltonian(0, flag)
        atoms1 = self.atoms_l[0]
        calc1 = atoms1.calc
        kpts = calc1.ibzk_kc
        p1 = calc1.input_parameters.copy()
        p = self.atoms.calc.input_parameters.copy()
    
        pl1 = self.h1_skmm.shape[-1]
        nspins = self.h1_skmm.shape[0]

        #self.nxklead = p1['kpts'][0]  
        #self.nxkmol = p['kpts'][0]
        self.nxklead = 3
        self.nxkmol = 3
        
        #self.nyzk = kpts.shape[0] / self.nxklead        
        self.nyzk = 1
        
        nxk = self.nxklead
        weight = npy.array([1.0 / nxk] * nxk )
                 
        xkpts = self.pick_out_xkpts(nxk, kpts)
        self.check_edge(self.s1_kmm, xkpts, weight)
        
        self.h1_syzkmm = self.substract_yzk(nxk, kpts, self.h1_skmm, 'h')
        self.s1_yzkmm = self.substract_yzk(nxk, kpts, self.s1_kmm)

        self.h1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h1_skmm, 'h', [1.0,0,0])
        self.s1_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s1_kmm, 's', [1.0,0,0])

        #self.d1_skmm = self.generate_density_matrix(0)
             
        #self.d1_syzkmm = self.substract_yzk(nxk, kpts, self.d1_skmm, 'h')
        #self.d1_syzkmm_ij = self.substract_yzk(nxk, kpts, self.d1_skmm, 'h', [1.0,0,0])
                     
        self.update_lead_hamiltonian(1, flag)

        self.h2_syzkmm = self.substract_yzk(nxk, kpts, self.h2_skmm, 'h')
        self.s2_yzkmm = self.substract_yzk(nxk, kpts, self.s2_kmm)

        self.h2_syzkmm_ij = self.substract_yzk(nxk, kpts, self.h2_skmm, 'h', [-1.0,0,0])
        self.s2_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s2_kmm, 's', [-1.0,0,0])

        #self.d2_skmm = self.generate_density_matrix(1)
    
        self.update_scat_hamiltonian(flag)
        
        nxk = self.nxkmol
        kpts = self.atoms.calc.ibzk_kc
        self.h_syzkmm = self.substract_yzk(nxk, kpts, self.h_skmm, 'h')
        self.s_yzkmm = self.substract_yzk(nxk, kpts, self.s_kmm)
        self.s_yzkmm_ij = self.substract_yzk(nxk, kpts, self.s_kmm, 's', [1.0,0,0])
        
        #self.d_kmm = self.generate_density_matrix(2)
        #self.d_kmm = self.d_kmm[0]
        #self.d_yzkmm = self.substract_yzk(nxk, kpts, self.d_kmm)
        
        #self.d_yzkmm_ij = self.fill_density_matrix()
        
        #self.edge_density_mm = self.calc_edge_charge(self.d_yzkmm_ij,
        #                                                      self.s_yzkmm_ij)
        #self.edge_charge = npy.empty([self.nyzk])
        
        #for i in range(self.nyzk):
        #    self.edge_charge[i] = npy.trace(self.edge_density_mm[i])
        #    print 'edge_charge[',i,']=', self.edge_charge[i]
            
        
        
        
        
       
    def substract_yzk(self, nxk, kpts, k_mm, hors='s', position=[0, 0, 0]):
        nyzk = self.nyzk
        weight = npy.array([1.0 / nxk] * nxk )
        if hors != 's' and hors != 'h':
            raise KeyError('hors should be h or s!')
        if hors == 'h':
            dim = k_mm.shape[:]
            dim = (dim[0],) + (dim[1] / nxk,) + dim[2:]
            yzk_mm = npy.empty(dim, complex)
            dim = (dim[0],) + (nxk,) + dim[2:]
            xk_mm = npy.empty(dim, complex)
        elif hors == 's':
            dim = k_mm.shape[:]
            dim = (dim[0] / nxk,) + dim[1:]
            yzk_mm = npy.empty(dim, complex)
            dim = (nxk,) + dim[1:]
            xk_mm = npy.empty(dim, complex)
        n = 0
        xkpts = self.pick_out_xkpts(nxk, kpts)
        for i in range(nyzk):
            n = i
            for j in range(nxk):
                if hors == 'h':
                    xk_mm[:, j] = npy.copy(k_mm[:, n])
                elif hors == 's':
                    xk_mm[j] = npy.copy(k_mm[n])
                n += nyzk
            if hors == 'h':
                yzk_mm[:, i] = get_realspace_hs(xk_mm, None,
                                               xkpts, weight, position)
            elif hors == 's':
                yzk_mm[i] = get_realspace_hs(None, xk_mm,
                                                   xkpts, weight, position)
        return yzk_mm   
            
    def check_edge(self, k_mm, xkpts, weight):
        tolx = 1e-4
        position = [2,0,0]
        nbf = k_mm.shape[-1]
        nxk = self.nxklead
        nyzk = self.nyzk
        xk_mm = npy.empty([nxk, nbf, nbf], complex)
      
        num = 0       
        for i in range(nxk):
            xk_mm[i] = npy.copy(k_mm[num])
            num += nyzk

        r_mm = npy.empty([nbf, nbf], complex)
        r_mm = get_realspace_hs(None, xk_mm, xkpts, weight, position)
        matmax = npy.max(abs(r_mm))

        if matmax > tolx:
            print 'Warning*: the principle layer should be lagger, \
                                                          matmax=%f' % matmax
    
    def calc_edge_charge(self, d_yzkmm_ij, s_yzkmm_ij):
        nkpts = s_yzkmm_ij.shape[0]
        nbf = s_yzkmm_ij.shape[-1]
        edge_charge_mm = npy.zeros([nkpts, nbf, nbf])
        for i in range(nkpts):
            edge_charge_mm[i] += npy.dot(d_yzkmm_ij[i],
                                                   dagger(s_yzkmm_ij[i]))
            edge_charge_mm[i] += npy.dot(dagger(d_yzkmm_ij[i]),
                                                          s_yzkmm_ij[i])
        return edge_charge_mm
    def pick_out_xkpts(self, nxk, kpts):
        nyzk = self.nyzk
        xkpts = npy.zeros([nxk, 3])
        num = 0
        for i in range(nxk):
            xkpts[i, 0] = kpts[num, 0]
            num += nyzk
        return xkpts
    def generate_density_matrix(self, lead):
        nxk = self.nxklead
        nyzk = self.nyzk
        if lead == 0:
            calc = self.atoms_l[0].calc
            dim = self.h1_skmm.shape
        elif lead == 1:
            calc = self.atoms_l[1].calc
            dim = self.h2_skmm.shape
        elif lead == 2:
            calc = self.atoms.calc
            dim = self.h_skmm.shape
            nxk = self.nxkmol
        else:
            raise KeyError('invalid lead index in generate_density_matrix')
        d_skmm = npy.empty(dim)
        for kpt in calc.kpt_u:
            C_nm = kpt.C_nm
            f_nn = npy.diag(kpt.f_n)
            d_skmm[kpt.s, kpt.k] = npy.dot(dagger(C_nm),
                                          npy.dot(f_nn, C_nm)) * nxk * nyzk
        return d_skmm
    def fill_density_matrix(self):
        pl1 = self.h1_skmm.shape[-1]
        nbmol = self.h_skmm.shape[-1]
        nyzk = self.nyzk
        d_yzkmm_ij = npy.zeros([nyzk, nbmol, nbmol])
        d_yzkmm_ij[:, -pl1:, :pl1] = npy.copy(self.d1_syzkmm_ij[0])
        return d_yzkmm_ij
    def boundary_check(self):
        tol = 1e-4
        pl1 = self.h1_skmm.shape[-1]
        matdiff = self.h_syzkmm[0, :, :pl1, :pl1] - self.h1_syzkmm[0]
        ediff = npy.mean(matdiff)
        if npy.max(abs(matdiff - ediff * self.s1_yzkmm)) > tol:
            print 'Warning*: hamiltonian boundary difference %f' % ediff
        nspins = len(self.h_syzkmm)
        for i in range(nspins):
            self.h_syzkmm[i] -= ediff * self.s_yzkmm
        matdiff = self.d_yzkmm[:, :pl1, :pl1] - self.d1_syzkmm[0]
        ediff = npy.mean(matdiff)
        if npy.max(abs(matdiff - ediff * self.s1_yzkmm)) > tol:
            print 'Warning*: density boundary difference %f' % ediff
        
        
        
