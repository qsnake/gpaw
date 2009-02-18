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
    if U2 is None:
        for col in U.T:
            col /= la.norm(col)
    else:
         for col1, col2 in zip(U.T, U2.T):
             norm = np.sqrt(np.vdot(col1, col1) + np.vdot(col2, col2))
             col1 /= norm
             col2 /= norm
       

def normalize2(C, S):
    C /= np.sqrt(dots(dagger(C), S, C).diagonal())


def get_rot(F_MM, V_oM, L):
    eps_M, U_MM = la.eigh(F_MM)
    indices = eps_M.real.argsort()[-L:] 
    U_Ml = U_MM[:, indices]
    U_Ml /= np.sqrt(dots(U_Ml.T.conj(), F_MM, U_Ml).diagonal())

    U_ow = V_oM.copy()
    U_lw = np.dot(U_Ml.T.conj(), F_MM)
    for col1, col2 in zip(U_ow.T, U_lw.T):
         norm = np.sqrt(np.vdot(col1, col1) + np.vdot(col2, col2))
         col1 /= norm
         col2 /= norm
    return U_ow, U_lw, U_Ml


def condition_number(S):
    eps = la.eigvalsh(S).real
    return eps.max() / eps.min()


def eigvals(H, S):
    return np.sort(la.eigvals(la.solve(S, H)).real)


def get_bfs(calc):
    bfs = BasisFunctions(calc.gd, [setup.phit_j for setup in calc.wfs.setups],
                         calc.wfs.kpt_comm, cut=True)
    if not calc.wfs.gamma:
        bfs.set_k_points(calc.wfs.ibzk_qc)
    bfs.set_positions(calc.atoms.get_scaled_positions())
    return bfs


def get_lfc(calc):
    spos_Ac = []
    spline_Aj = []
    for a, spos_c in enumerate(calc.atoms.get_scaled_positions()):
        for phit in calc.wfs.setups[a].phit_j:
            spos_Ac.append(spos_c)
            spline_Aj.append([phit])

    lfc = LFC(calc.gd, spline_Aj, calc.wfs.kpt_comm,
              cut=True, dtype=calc.wfs.dtype)
    if not calc.wfs.gamma:
        lfc.set_k_points(calc.wfs.ibzk_qc)
    lfc.set_positions(np.array(spos_Ac))
    return lfc
    

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
    lfc = get_lfc(calc)
    V_qAni = [lfc.dict(calc.wfs.nbands) for q in range(nq)]
    for q, V_Ani in enumerate(V_qAni):
        lfc.integrate(calc.wfs.kpt_u[q].psit_nG[:], V_Ani, q)
        M1 = 0
        for A in range(len(V_Ani)):
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
        return np.asarray([condition_number(S) for S in self.S_kii])

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
        return np.asarray([eigvals(H, S)
                           for H, S in zip(self.H_kii, self.S_kii)])

    def get_lcao_eigenvalues(self):
        return np.asarray([eigvals(H, S)
                           for H, S in zip(self.h_lcao_kii, self.s_lcao_kii)])

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

        return np.hstack((normo_kn, normu_kn))

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
