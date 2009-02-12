import numpy as np
from numpy import linalg as la
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack
from gpaw.utilities.tools import dagger, lowdin
from gpaw.lcao.tools import get_realspace_hs
from ase import Hartree


def dots(*args):
    x = args[0]
    for M in args[1:]:
        x = np.dot(x, M)
    return x        


def normalize(U, U2=None):
    if U2==None:
        for col in U.T:
            col /= la.norm(col)
    else:
         for col1, col2 in zip(U.T, U2.T):
            L2 = np.vdot(col1, col1) + np.vdot(col2, col2)
            N = 1 / np.sqrt(L2)
            col1 *= N
            col2 *= N
       

def normalize2(C, S):
    C /= np.sqrt(np.dot(np.dot(dagger(C), S), C).diagonal())


def get_bfs(calc):
    bfs = BasisFunctions(calc.gd, [setup.phit_j for setup in calc.wfs.setups],
                         calc.wfs.kpt_comm, cut=True)
    if not calc.wfs.gamma:
        bfs.set_k_points(calc.wfs.ibzk_qc)
    bfs.set_positions(calc.atoms.get_scaled_positions())
    return bfs


def get_phs(calc, s=0):
    dtype = calc.wfs.dtype
    spos_ac = calc.atoms.get_scaled_positions()
    setups = calc.wfs.setups
    domain = calc.domain
    tci = TwoCenterIntegrals(domain, setups, calc.wfs.gamma, calc.wfs.ibzk_qc)
    tci.set_positions(spos_ac)

    nq = len(calc.wfs.ibzk_qc)
    nao = calc.wfs.setups.nao
    S_qMM = np.zeros((nq, nao, nao), dtype)
    T_qMM = np.zeros((nq, nao, nao), dtype)

    #setup basis functions
    bfs = get_bfs(calc)
    
    P_aqMi = {}
    for a in bfs.my_atom_indices:
        ni = calc.wfs.setups[a].ni
        P_aqMi[a] = np.zeros((nq, nao, ni), dtype)

    tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi, dtype)

    vt_G = calc.hamiltonian.vt_sG[s]
    H_qMM = np.zeros((nq, nao, nao), dtype)
    for q, H_MM in enumerate(H_qMM):
        bfs.calculate_potential_matrix(vt_G, H_MM, q)

    # Non-local corrections
    for a, P_qMi in P_aqMi.items():
        dH_ii = unpack(calc.hamiltonian.dH_asp[a][s])
        for P_Mi, H_MM in zip(P_qMi, H_qMM):
            H_MM +=  np.dot(P_Mi, np.inner(dH_ii, P_Mi).conj())

    H_qMM += T_qMM # kinetic energy
    #fill in the upper triangle
    tri = np.tri(nao, dtype=dtype)
    tri.flat[::nao + 1] = 0.5
    for H_MM, S_MM in zip(H_qMM, S_qMM):
        H_MM *= tri 
        H_MM += dagger(H_MM)
        H_MM *= Hartree
        S_MM *= tri 
        S_MM += dagger(S_MM)

    # Calculate projections
    V_qnM = np.zeros((nq, calc.wfs.nbands, nao), dtype)

    # Non local corrections
    for q in range(nq):
        for a, P_ni in calc.wfs.kpt_u[q].P_ani.items():
            dS_ii = calc.wfs.setups[a].O_ii
            P_Mi = P_aqMi[a][q]
            V_qnM[q] += np.dot(P_ni, np.inner(dS_ii, P_Mi).conj())

    #Hack XXX, not needed when BasisFunctions get
    #an integrate method.
    spos_Ac = []
    spline_Aj = []
    for a, spos_c in enumerate(spos_ac):
        for phit in calc.wfs.setups[a].phit_j:
            spos_Ac.append(spos_c)
            spline_Aj.append([phit])
            
    bfs = LFC(calc.gd, spline_Aj, calc.wfs.kpt_comm, cut=True, dtype=dtype)
    if not calc.wfs.gamma:
        bfs.set_k_points(calc.wfs.ibzk_qc)
    bfs.set_positions(np.array(spos_Ac))

    V_qAni = [bfs.dict(calc.wfs.nbands) for q in range(nq)]
    #XXX a copy is made of psit_nG in case it is a tar-reference.
    for q, V_Ani in enumerate(V_qAni):
        bfs.integrate(calc.wfs.kpt_u[q].psit_nG[:], V_Ani, q)
        M1 = 0
        for A in range(len(spos_Ac)):
            V_ni = V_Ani[A]
            M2 = M1 + V_ni.shape[1]
            V_qnM[q, :, M1:M2] += V_ni
            M1 = M2

    return V_qnM, H_qMM, S_qMM, P_aqMi


class ProjectedWannierFunctions:
    def __init__(self, projections, h_lcao, s_lcao, eigenvalues, kpoints, 
                 L_k=None, M_k=None, N=None, fixedenergy=None):
        """projections[n,i] = <psi_n|f_i>
           h_lcao[i1, i2] = <f_i1|h|f_i2>
           s_lcao[[i1, i2] = <f_i1|f_i2>
           eps_n: Exact eigenvalues
           L: Number of extra degrees of freedom
           M: Number of states to exactly span
           N: Total number of bands in the calculation
           
           Methods:
           -- get_hamiltonian_and_overlap_matrix --
           will return the hamiltonian and identity operator
           in the projected wannier function basis. 
           The following steps are performed:
            
           1) calculate_edf       -> self.b_il
           2) calculate_rotations -> self.Uo_mi and self.Uu_li
           3) calculate_overlaps  -> self.S_ii
           4) calculate_hamiltonian_matrix -> self.H_ii

           -- get_eigenvalues --
           gives the eigenvalues of of the hamiltonian in the
           projected wannier function basis.

           -- indices --
           i localized function index
           n eigenstate index
           l edf index
           m fixed eigenstate index
           k k-point index
           """
         
        self.eps_kn = eigenvalues
        self.ibzk_kc = kpoints
        self.nk = len(self.ibzk_kc)
        self.V_kni = projections   #<psi_n1|f_i1>
        self.dtype = self.V_kni.dtype
        self.Nw = self.V_kni.shape[2]
        self.s_lcao_kii = s_lcao #F_ii[i1,i2] = <f_i1|f_i2>
        self.h_lcao_kii = h_lcao

        if N is None:
            N = self.V_kni.shape[1]
        self.N = N
        
        if fixedenergy is None:
            raise NotImplementedError,'Only fixedenergy is implemented for now'
        else:
            self.fixedenergy = fixedenergy
            self.M_k = [sum(eps_n <= fixedenergy) for eps_n in self.eps_kn]
            self.L_k = [self.Nw - M for M in self.M_k]
            print "fixedenergy =", self.fixedenergy

        print 'N =', self.N
        print 'skpt_kc = '
        print self.ibzk_kc
        print 'M_k =', self.M_k
        print 'L_k =', self.L_k
        print 'Nw =', self.Nw

    def get_hamiltonian_and_overlap_matrix(self, useibl=True):
        self.calculate_edf(useibl=useibl)
        self.calculate_rotations()
        self.calculate_overlaps()
        self.calculate_hamiltonian_matrix(useibl=useibl)
        return self.H_kii, self.S_kii

    def calculate_edf(self, useibl=True):
        """Calculate the coefficients b_il in the expansion of the EDF.

        ``|phi_l> = sum_i b_il |f^u_i>``, in terms of ``|f^u_i> = P^u|f_i>``.

        To use the infinite band limit set useibl=True.
        N is the total number of bands to use.
        """
        
        for k, L in enumerate(self.L_k):
            if L==0:
                assert L!=0, 'L_k=0 for k=%i. Not implemented' % k
        
        self.Vo_kni = [V_ni[:M] for V_ni, M in zip(self.V_kni, self.M_k)]
        
        self.Fo_kii = np.asarray([np.dot(dagger(Vo_ni), Vo_ni) 
                                  for Vo_ni in self.Vo_kni])
        
        if useibl:
            self.Fu_kii = self.s_lcao_kii - self.Fo_kii
        else:
            self.Vu_kni = [V_ni[M:self.N] 
                           for V_ni, M in zip(self.V_kni, self.M_k)]
            self.Fu_kii = np.asarray([np.dot(dagger(Vu_ni), Vu_ni) 
                                     for Vu_ni in self.Vu_kni])
        self.b_kil = [] 
        for Fu_ii, L in zip(self.Fu_kii, self.L_k):
            b_i, b_ii = la.eigh(Fu_ii)
            ls = b_i.real.argsort()[-L:] 
            b_il = b_ii[:, ls] #pick out the eigenvec with largest eigenvals.
            normalize2(b_il, Fu_ii) #normalize the EDF: <phi_l|phi_l> = 1
            self.b_kil.append(b_il)

    def calculate_rotations(self):
        Uo_kni = [Vo_ni.copy() for Vo_ni in self.Vo_kni]
        Uu_kli = [np.dot(dagger(b_il), Fu_ii) 
                  for b_il, Fu_ii in zip(self.b_kil, self.Fu_kii)]
        #Normalize such that <omega_i|omega_i> = 1
        for Uo_ni, Uu_li in zip(Uo_kni, Uu_kli):
            normalize(Uo_ni, Uu_li)
        self.Uo_kni = Uo_kni
        self.Uu_kli = Uu_kli

    def calculate_overlaps(self):
        Wo_kii = [np.dot(dagger(Uo_ni), Uo_ni) for Uo_ni in self.Uo_kni]
        Wu_kii = [dots(dagger(Uu_li), dagger(b_il), Fu_ii, b_il, Uu_li) 
        for Uu_li, b_il, Fu_ii in zip(self.Uu_kli, self.b_kil, self.Fu_kii)]
        Wo_kii = np.asarray(Wo_kii)
        Wu_kii = np.asarray(Wu_kii)
        self.S_kii = Wo_kii + Wu_kii

    def get_condition_number(self):
        eigs_kn = [la.eigvalsh(S_ii) for S_ii in self.S_kii]
        return np.asarray([abs(eigs_n.max() / eigs_n.min()) 
                          for eigs_n in eigs_kn])

    def calculate_hamiltonian_matrix(self, useibl=True):
        """Calculate H_kij = H^o_i(k)j(k) + H^u_i(k)j(k)
           i(k): Bloch sum of omega_i
        """

        epso_kn = [eps_n[:M] for eps_n, M in zip(self.eps_kn, self.M_k)]
        self.Ho_kii = np.asarray([np.dot(dagger(Uo_ni) * epso_n, Uo_ni) 
                                  for Uo_ni, epso_n in zip(self.Uo_kni, 
                                                           epso_kn)])

        if self.h_lcao_kii!=None and useibl:
            print "Using h_lcao and infinite band limit"
            Vo_kni = self.Vo_kni
            Huf_kii = [h_lcao_ii - np.dot(dagger(Vo_ni) * epso_n, Vo_ni)
                       for h_lcao_ii, Vo_ni, epso_n in zip(self.h_lcao_kii, 
                                                           self.Vo_kni, 
                                                           epso_kn)]
            self.Huf_kii = np.asarray(Huf_kii)
        else:
            print "Using finite band limit (not using h_lcao)"
            epsu_kn = [eps_n[M:self.N] 
                       for eps_n, M in zip(self.eps_kn, self.M_k)]
            Huf_kii = [np.dot(dagger(Vu_ni) * epsu_n, Vu_ni) 
                       for Vu_ni, epsu_n in zip(self.Vu_kni, epsu_kn)]
            self.Huf_kii = np.asarray(Huf_kii)

        Hu_kii = [dots(dagger(Uu_li), dagger(b_il), Huf_ii, b_il, Uu_li)
                  for Uu_li, b_il, Huf_ii in zip(self.Uu_kli, self.b_kil,
                                                 self.Huf_kii)]
        self.Hu_kii = np.asarray(Hu_kii)
        self.H_kii = self.Ho_kii + self.Hu_kii

    def get_eigenvalues(self):
        eigs_kn = [np.sort(la.eigvals(la.solve(S_ii, H_ii)).real)
                   for H_ii, S_ii in zip(self.H_kii, self.S_kii)]
        return np.asarray(eigs_kn)

    def get_lcao_eigenvalues(self):
        eigs_kn = [np.sort(la.eigvals(la.solve(s_ii, h_ii)).real)
                   for h_ii, s_ii in zip(self.h_lcao_kii, self.s_lcao_kii)]
        return np.asarray(eigs_kn)

    def get_norm_of_projection(self):
        norm_kn = np.zeros((self.nk, self.N))
        Sinv_kii = np.asarray([la.inv(S_ii) for S_ii in self.S_kii])

        normo_kn = np.asarray([dots(Uo_ni, Sinv_ii, dagger(Uo_ni)).diagonal()
                    for Uo_ni, Sinv_ii in zip(self.Uo_kni, Sinv_kii)])
        
        Vu_kni = np.asarray([V_ni[M:self.N] 
                             for V_ni, M in zip(self.V_kni, self.M_k)])

        Pu_kni = [dots(Vu_ni, b_il, Uu_li) 
                for Vu_ni, b_il, Uu_li in zip(Vu_kni, self.b_kil, self.Uu_kli)]

        normu_kn = np.asarray([dots(Pu_ni, Sinv_ii, dagger(Pu_ni)).diagonal()
                    for Pu_ni, Sinv_ii in zip(Pu_kni, Sinv_kii)])

        norm_kn = np.concatenate([normo_kn.flat, normu_kn.flat])
        norm_kn.shape = ([normo_kn.shape[0],
                          normo_kn.shape[1]+normu_kn.shape[1]])
        return norm_kn                          

    def get_mlwf_initial_guess(self):
        """calculate initial guess for maximally localized 
        wannier functions. Does not work for the infinite band limit.
        cu_nl: rotation coefficents of unoccupied states
        U_ii: rotation matrix of eigenstates and edf.
        """
        Vu_ni = self.Vu_ni[self.M: self.N]
        cu_nl = np.dot(Vu_ni, self.b_il)
        U_ii = np.vstack((self.Uo_ni, self.Uu_li))
        lowdin(U_ii)
        return U_ii, cu_nl

    def get_function(self, wfs, bfs=None, useibl=True):
        """Returns specified projected wannier function (k, i).

        The Wannier function is constructed by
        a) rotating the pseudo eigen states and pseudo target functions,
        b) rotating the corresponding projector function overlaps.

        bfs must be specified if useibl is True.
        """
        k = 0 # XXX temporary hack
        Gshape = wfs.kpt_u[k].psit_nG.shape[-3:]
        psit_nG = wfs.kpt_u[k].psit_nG[:self.N].reshape(self.N, -1)
        M = self.M_k[k]
        bU_ii = np.dot(self.b_kil[k], self.Uu_kli[k])
        Uo_ni = self.Uo_kni[k]
        V_ni = self.V_kni[k]
        if useibl:
            # Rotate occupied part
            rot_mi = Uo_ni - np.dot(V_ni[:M], bU_ii)
            w_iG = np.dot(rot_mi.T, psit_nG[:M]).reshape((-1,) + Gshape)
            
            # Mix in relevant combination of target functions
            bfs.lcao_to_grid(bU_ii, w_iG, q=-1)
        else:
            rot_ni = np.vstack((Uo_ni, np.dot(V_ni[M:], bU_ii)))
            w_iG = np.dot(rot_ni.T, psit_nG).reshape((-1,) + Gshape)
        return w_iG

        
if __name__=='__main__':
    from ase import Atoms, molecule
    from gpaw import GPAW, Mixer
    import numpy as np
    from gpaw.lcao.tools import get_realspace_hs
    from time import time

    if 0:
        atoms = molecule('C6H6')
        atoms.center(vacuum=2.5)
        calc = GPAW(h=0.2, basis='szp', width=0.05, convergence={'bands':17})
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('C6H6.gpw', 'all')


    if 0:
        calc = GPAW('C6H6.gpw', txt=None, basis='sz')
        ibzk_kc = calc.wfs.ibzk_kc
        nk = len(ibzk_kc)
        Ef = calc.get_fermi_level()
        eps_kn = np.asarray([calc.get_eigenvalues(k) for k in range(nk)])
        eps_kn -= Ef

        V_knM, H_kMM, S_kMM, P_aqMi = get_phs(calc, s=0)
        H_kMM -= Ef * S_kMM 
        
        pwf = ProjectedWannierFunctions(V_knM, 
                                        h_lcao=H_kMM, 
                                        s_lcao=S_kMM, 
                                        eigenvalues=eps_kn,
                                        kpoints=ibzk_kc,
                                        fixedenergy=5.0)
        t1 = time()
        h, s = pwf.get_hamiltonian_and_overlap_matrix(useibl=True)
        t2 = time()
        print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)
        norm_kn = pwf.get_norm_of_projection()
        eps1_kn = pwf.get_eigenvalues()
        print "band | deps/eV |  norm"
        print "-------------------------"
        for n in range(norm_kn.shape[1]):
            norm = norm_kn[0, n]
            if n >= eps1_kn.shape[1]:
                print "%4i |    -    | %.1e " % (n, norm)
            else:
                deps = np.around(abs(eps1_kn[0,n] - eps_kn[0, n]), 13)
                print "%4i | %.1e | %.1e " % (n, deps, norm)

##         import tab
##         bfs = get_bfs(calc)
##         w_iG = pwf.get_function(calc.wfs, bfs=bfs, useibl=True)
##         from ase import write
##         atoms = calc.get_atoms()
##         for i in range(30):
##             write('wf_ibl_%s.cube' % i, atoms, data = w_iG[i])


    if 0:
        atoms = Atoms('Al', cell=(2.42, 7, 7), pbc=True)
        calc = GPAW(h=0.2, basis='dzp', kpts=(12, 1, 1), 
                    convergence={'bands':9},
                    maxiter=200)
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('al.gpw', 'all')

    if 0:
        calc = GPAW('al.gpw', txt=None, basis='sz')
        ibzk_kc = calc.wfs.ibzk_kc
        nk = len(ibzk_kc)
        Ef = calc.get_fermi_level()
        eps_kn = np.asarray([calc.get_eigenvalues(kpt=k) for k in range(nk)])
        eps_kn -= Ef
        
        V_knM, H_kMM, S_kMM = get_phs(calc, s=0)
        H_kMM -= S_kMM*Ef
        
        pwf = ProjectedWannierFunctions(V_knM, 
                                        h_lcao=H_kMM, 
                                        s_lcao=S_kMM, 
                                        eigenvalues=eps_kn, 
                                        fixedenergy=1.0,
                                        kpoints=ibzk_kc)
        
        t1 = time()
        h_kMM, s_kMM = pwf.get_hamiltonian_and_overlap_matrix(useibl=True)
        t2 = time()
        print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)
        print "max condition number:", pwf.get_condition_number().max()
        eigs_kn = pwf.get_eigenvalues()
        fd2 = open('bands_al_sz.dat','w')
        fd1 = open('bands_al_exact.dat', 'w')
        for eps1_n, eps2_n, k in zip(eps_kn, eigs_kn, ibzk_kc[:,0]):
            for e1 in eps1_n:
                print >> fd1, k, e1
            for e2 in eps2_n:
                print >> fd2, k, e2
        fd1.close()            
        fd2.close()
        h_skMM = h_kMM.copy()
        h_skMM.shape=(1, 4, 4, 4)
        n = 2
        w_k = calc.wfs.weight_k
        h_n, s_n = get_realspace_hs(h_skMM, s_kMM, ibzk_kc, w_k, (n, 0, 0))

    if 0:
        atoms = Atoms('Al', cell=(2.42, 7, 7), pbc=True)
        atoms*=(8, 1, 1)
        calc = GPAW(h=0.2, basis='szp', kpts=(1, 1, 1), 
                    convergence={'bands':4*8}, width=0.1,
                    maxiter=200, mixer=Mixer(0.1,7,metric='new', weight=100.))
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write('al8.gpw', 'all')

    if 0:
        calc = GPAW('al8.gpw', txt=None, basis='sz')
        ibzk_kc = calc.wfs.ibzk_kc
        nk = len(ibzk_kc)
        Ef = calc.get_fermi_level()
        eps_kn = np.asarray([calc.get_eigenvalues(kpt=k) for k in range(nk)])
        eps_kn -= Ef
        
        V_knM, H_kMM, S_kMM = get_phs(calc, s=0)
        H_kMM -= S_kMM * Ef
        
        pwf = ProjectedWannierFunctions(V_knM, 
                                        h_lcao=H_kMM, 
                                        s_lcao=S_kMM, 
                                        eigenvalues=eps_kn, 
                                        fixedenergy=0.0,
                                        kpoints=ibzk_kc)
        
        t1 = time()
        h_kMM, s_kMM = pwf.get_hamiltonian_and_overlap_matrix(useibl=True)
        t2 = time()
        print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)


