from ase import Atoms, Atom, monkhorst_pack, Hartree
from gpaw import GPAW, Mixer
from gpaw.lcao.tools import get_realspace_hs, tri2full, remove_pbc
import pickle
import numpy as npy
from gpaw.mpi import world

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

    def update_lead_hamiltonian(self, l):
        self.atoms_l[l] = self.get_lead_atoms(l)
        atoms = self.atoms_l[l]
        atoms.get_potential_energy()
        if l == 0:
            self.h1_skmm, self.s1_kmm = self.get_hs(atoms)
        elif l == 1:
            self.h2_skmm, self.s2_kmm = self.get_hs(atoms)

    def update_scat_hamiltonian(self):
        atoms = self.atoms
        atoms.get_potential_energy()
        self.h_skmm, self.s_kmm = self.get_hs(atoms)

    def get_hs(self, atoms):
        calc = atoms.calc
        Ef = calc.get_fermi_level()
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
                h_skmm[s,k] -= Ef * s_kmm[k]

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

        kpts = [1, 1, 1]
        kpts[self.d] = 2 * int(17.0 / self.pl_cells[l][self.d]) + 1
        p['kpts'] = kpts
        
        if 'mixer' in p: # XXX Works only if spin-paired
            p['mixer'] = Mixer(0.1, 5, metric='new', weight=100.0)

        if 'txt' in p and p['txt'] != '-':
            p['txt'] = 'lead%i_' % (l + 1) + p['txt']
        return GPAW(**p)
        
