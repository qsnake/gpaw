from math import pi
from pickle import load, dump

import numpy as npy
from ase.units import Bohr

from _gpaw import localize
from gpaw.utilities.tools import dagger, lowdin


class Wannier:
    def __init__(self, calc=None, spin=0):
        self.spin = spin
        if calc is not None:
            self.cell_c = calc.domain.cell_c * Bohr
            n = calc.get_number_of_bands()
            self.Z_nnc = npy.empty((n, n, 3), complex)
            for c in range(3):
                self.Z_nnc[:, :, c] = calc.get_wannier_integrals(c, spin,
                                                                #k, k1, G
                                                                 0, 0, 1.0)
            self.value = 0.0
            self.U_nn = npy.identity(n)

    def load(self, filename):
        self.cell_c, self.Z_nnc, self.value, self.U_nn = load(open(filename))

    def dump(self, filename):
        dump((self.cell_c, self.Z_nnc, self.value, self.U_nn), filename)
        
    def localize(self, eps=1e-5, iterations=-1):
        i = 0
        while i != iterations:
            value = localize(self.Z_nnc, self.U_nn)
            print i, value
            if value - self.value < eps:
                break
            i += 1
            self.value = value
        return value # / Bohr**6

    def get_centers(self):
        scaled_c = -npy.angle(self.Z_nnc.diagonal()).T / (2 * pi)
        return (scaled_c % 1.0) * self.cell_c

    def get_function(self, calc, n):
        psit_nG = calc.kpt_u[self.spin].psit_nG[:].reshape((calc.nbands, -1))
        return npy.dot(self.U_nn[:, n],
                       psit_nG).reshape(calc.gd.n_c) / Bohr**1.5

    def get_hamiltonian(self, calc):
        # U^T diag(eps_n) U
        eps_n = calc.get_eigenvalues(kpt=0, spin=self.spin)
        npy.dot(self.U_nn.T * eps_n, self.U_nn)


class LocFun(Wannier):
    def localize(self, calc, N=None, projections=None, ortho=True):
        # N is size of Hilbert space to fix. Default is number of occ. bands.
        if N is None:
            N = 0
            f_n = calc.collect_occupations(0, self.spin)
            while f_n[N] > .01:
                N += 1

        if projections is None:
            projections = single_zeta(calc, self.spin)
        
        U_nj, self.S_jj = get_locfun_rotation(projections, N, ortho)
        Z_nnc = npy.empty(S_jj.shape + (3,))
        for c in range(3):
            Z_nnc[:, :, c] = npy.dot(dagger(U_nj),
                                     npy.dot(self.Z_nnc[:, :, c], U_nj))
        self.U_nn = U_nj
        self.Z_nnc = Z_nnc
        self.value = npy.sum(npy.abs(Z_nnc.diagonal())**2)
        return self.value # / Bohr**6
            

def get_locfun_rotation(projections_nj, N=None, ortho=False):
    """Mikkel Strange's localized functions.
    
    projections_nj = <psi_n|p_j>
    psi_n: eigenstates
    p_j: localized function
    N = number of occupied states (or part of Hilbert space to fix)
    """

    Nbands, M = projections_nj.shape
    if N is None:
        N = Nbands
    P = M - N # Extra degrees of freedom
    V = Nbands - N # Virtual states

    a0_nj = projections_nj[:N, :]
    a0_vj = projections_nj[N:N + V, :]
    B_vv = npy.dot(a0_vj, dagger(a0_vj))
    b_v, b_vv = npy.linalg.eigh(B_vv)
    list = npy.argsort(-b_v)
    T_vp = npy.take(b_vv, npy.argsort(-b_v)[:P], axis=1)
    R_vv = npy.dot(T_vp, dagger(T_vp))
    D_jj = npy.dot(dagger(a0_nj), a0_nj) + npy.dot(dagger(a0_vj),
                                                   npy.dot(R_vv, a0_vj))
    D2_j = 1.0 / npy.sqrt(D_jj.diagonal())
    ap_nj = D2_j * a0_nj
    ap_vj = D2_j * npy.dot(R_vv, a0_vj)
    S_jj = npy.dot(dagger(ap_nj), ap_nj) + npy.dot(dagger(ap_vj), ap_vj)
    
    # Check for linear dependencies
    Scd = npy.diagonal(npy.linalg.cholesky(S_jj))
    if Scd.min() < 0.01:
        print "Warning: possibly near linear depedence"

    if ortho:
        ap_nj = lowdin(ap_nj, S_jj)
        ap_vj = lowdin(ap_vj, S_jj)

    U_nj = npy.concatenate([ap_nj.flatten(),
                            ap_vj.flatten()]).reshape([N + V, M])
    return U_nj, S_jj

def single_zeta(paw, spin):
    p_jn = []
    for nucleus in paw.nuclei:
        i = 0
        for l, n in zip(nucleus.setup.l_j, nucleus.setup.n_j):
            if n < 0:
                break
            for j in range(i, i + 2 * l + 1):
                p_jn.append(nucleus.P_uni[spin, :, j])
            i += 2 * l + 1
    projections_nj = dagger(npy.array(p_jn))
    assert projections_nj.shape[0] >= projections_nj.shape[1]
    return projections_nj
