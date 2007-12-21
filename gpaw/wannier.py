from ASE.Utilities.Wannier import Wannier as ASEWannier
import Numeric as num
from math import pi
from cmath import exp

class Wannier(ASEWannier):
    def __init__(self,
                 numberofwannier,
                 calculator,
                 numberofbands=None,
                 occupationenergy=0,
                 numberoffixedstates=None,
                 spin=0,
                 initialwannier=None,
                 seed=None):

        self.CheckNumeric()
        self.SetNumberOfWannierFunctions(numberofwannier)
        self.SetCalculator(calculator)
        if numberofbands is not None:
            self.SetNumberOfBands(numberofbands)
        else:
            self.SetNumberOfBands(calculator.GetNumberOfBands())
        self.SetSpin(spin)
        self.seed = seed
        self.SetIBZKPoints(calculator.GetIBZKPoints())
        self.SetBZKPoints(calculator.GetBZKPoints())
        assert len(self.GetBZKPoints()) == len(self.GetIBZKPoints()),\
               'k-points must not be symmetry reduced'

        # Set eigenvalues relative to the Fermi level
        efermi = calculator.GetFermiLevel()
        self.SetEigenValues([calculator.GetEigenvalues(kpt, spin) - efermi
                             for kpt in range(len(self.GetBZKPoints()))])

        if numberoffixedstates is not None:
            self.SetNumberOfFixedStates(numberoffixedstates)
            numberofextradof = num.array(len(numberoffixedstates) *
                                         [numberofwannier]) \
                                         - num.array(numberoffixedstates)
            self.SetNumberOfExtraDOF(numberofextradof.tolist())
        else:
            # All states below this energy (relative to Fermi level) are fixed.
            self.SetOccupationEnergy(occupationenergy)
            self.InitOccupationParameters()

        if initialwannier is not None:
            self.SetInitialWannierFunctions(initialwannier)

        if calculator.typecode == num.Float:
            self.SetType(float)
        else:
            self.SetType(complex)
    
        # Set unitcell and determine reciprocal weights
        self.SetUnitCell(calculator.GetListOfAtoms().GetUnitCell())
        self.CalculateWeightsFromUnitCell()

        Nb, M_k, L_k = self.GetMatrixDimensions()
        Nw = self.GetNumberOfWannierFunctions()
        print 'Number of bands:', Nb
        print 'Number of Wannier functions:', Nw
        print 'kpt | Fixed | EDF'
        for k in range(len(M_k)):
            print str(k).center(3), '|', str(M_k[k]).center(5), '|', str(L_k[k]).center(3)


    def SetInitialWannierFunctions(self, initialwannier):
        self.initialwannier = initialwannier

    def InitializeRotationAndCoefficientMatrices(self):
        # Set ZIMatrix and ZIkMatrix to zero.
        self.InitZIMatrices()
        if not hasattr(self, 'initialwannier'):
            self.RandomizeMatrices(seed=self.seed)
        else:
            initialwannier = self.initialwannier
            Nb, M_k, L_k = self.GetMatrixDimensions()
            Nk = self.GetNumberOfKPoints()
            spin = self.GetSpin()
            nuclei = self.GetCalculator().nuclei

            V_knj = num.zeros((Nk, Nb, len(initialwannier)), num.Complex)
            ##for k in range(Nk):
            ##    for j in range(len(initialwannier)):
            ##        a, i = initialwannier[j]
            ##        print "%i %i" % (a,i)
            ##        u = k + spin * Nk
            ##        V_knj[k, :, j] = nuclei[a].P_uni[u, :, i]
            V_knj = get_projections(initialwannier, self.GetCalculator())

            c_k, U_k = get_c_k_and_U_k(V_knj, (Nb, M_k, L_k))
            self.SetListOfRotationMatrices(U_k)
            self.SetListOfCoefficientMatrices(c_k)
            self.UpdateListOfLargeRotationMatrices()
            self.UpdateZIMatrix()

    def GetWannierFunctionOnGrid(self, wannierindex, repeat=None):
        """wannierindex can be either a single WF or a coordinate vector
        in terms of the WFs."""

        # The coordinate vector of wannier functions
        if type(wannierindex) == int:
            coords_w = num.zeros(self.GetNumberOfWannierFunctions(),
                                 num.Complex)
            coords_w[wannierindex] = 1.0
        else:   
            coords_w = wannierindex

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.GetKPointGrid()
        else:
            repeat = num.array(repeat) # Ensure that repeat is an array
        N1, N2, N3 = repeat

        dim = self.GetCalculator().GetNumberOfGridPoints()
        largedim = dim * repeat
        
        bzk_kc = self.GetBZKPoints()
        Nkpts = len(bzk_kc)
        kpt_u = self.GetCalculator().kpt_u
        wanniergrid = num.zeros(largedim, typecode=num.Complex)
        for k, kpt_c in enumerate(bzk_kc):
            u = (k + Nkpts * self.spin) % len(kpt_u)
            U_nw = self.GetListOfLargeRotationMatrices()[k]
            vec_n = num.matrixmultiply(U_nw, coords_w)

            psi_nG = num.reshape(kpt_u[u].psit_nG[:], (len(vec_n), -1))
            wan_G = num.dot(vec_n, psi_nG)
            wan_G.shape = tuple(dim)

            # Distribute the small wavefunction over large cell:
            for n1 in range(N1):
                for n2 in range(N2):
                    for n3 in range(N3):
                        e = exp(-2.j * pi * num.dot([n1, n2, n3], kpt_c))
                        wanniergrid[n1 * dim[0]:(n1 + 1) * dim[0],
                                    n2 * dim[1]:(n2 + 1) * dim[1],
                                    n3 * dim[2]:(n3 + 1) * dim[2]] += e * wan_G

        # Normalization
        wanniergrid /= num.sqrt(Nkpts)
        return wanniergrid

    def WriteCube(self, wannierindex, filename, repeat=None, real=False):
        from ASE.IO.Cube import WriteCube
        if repeat is None:
            repeat = self.GetKPointGrid()
        wanniergrid = self.GetWannierFunctionOnGrid(wannierindex, repeat)
        WriteCube(self.calculator.GetListOfAtoms().Repeat(repeat),
                  wanniergrid, filename, real=real)


#Mikkel Strange (2007) 
#See Thygesen et al. PRB
from random import random
from gpaw.utilities.tools import dagger, project, normalize, gram_schmidt_orthonormalize
from gpaw.localized_functions import create_localized_functions
from gpaw.spline import Spline

def get_projections(initialwannier, calc):
    #initialwannier = [[spos_c, ls, a]]
    
    nbf = 0
    for spos_c,ls,a in initialwannier:
        nbf += num.sum([2 * l + 1 for l in ls])
    
    f_kni = num.zeros((len(calc.ibzk_kc),calc.nbands,nbf),num.Complex)
    
    nbf = 0
    for spos_c,ls,a in initialwannier:
        a /= calc.a0
        cutoff = 4 * a
        x = num.arange(0.0,cutoff,cutoff / 500.0)
        rad_g = num.exp(-x*x/a**2)
        rad_g[-1:] = 0.0
        functions = [Spline(l,cutoff,rad_g) for l in ls]
        lf = create_localized_functions(functions,calc.gd,spos_c,
                                        typecode=calc.typecode)
        lf.set_phase_factors(calc.ibzk_kc)
        nlf = num.sum([2 * l + 1 for l in ls])
        nbands = calc.nbands
        nkpts = len(calc.ibzk_kc)
        for k in range(nkpts):
            lf.integrate(calc.kpt_u[k].psit_nG[:],f_kni[k,:,nbf:nbf+nlf],k=k)
        nbf += nlf
   
    return f_kni
   # f_kni = num.conjugate(f_kni)
 

def get_c_k_and_U_k(V_kni, NML):
    """V_kni = <psi_kn|f_i>, where f_i is an initial function """
    nbands, M_k, L_k = NML
    U_k = []
    c_k = []
    for M,L, V_ni in zip(M_k, L_k, V_kni):
        V_ni = normalize(V_ni)
        T = V_ni[M:].copy()
        nbf = T.shape[1] #number of initial functions
        c = num.zeros([nbands - M, L], num.Complex)
        U = num.zeros([M + L, M + L], num.Complex)
        #Calculate the EDF
        w = abs(num.sum(T * num.conjugate(T)))
        for i in xrange(min(L, nbf)):
            t = w.tolist().index(max(w))
            c[:, i] = T[:, t]
            for j in xrange(i):
                c[:,i] = c[:,i] - project(c[:, j], T[:, t])
            c[:,i] /= num.sqrt(num.dot(num.conjugate(c[:, i]), c[:, i]))
            w -= abs(num.dot(num.conjugate(c[:, i]), T))#**2 !?
        if nbf < L:
            print "augmenting with random vectors"
            for i in xrange(nbf, L):
                for j in xrange(nbands - M):
                    c[i, j] = random()
            c = gram_schmidt_orthonormalize(c)
        if L > 0:
            check_ortho(c)
        U[:M, :nbf] = V_ni[:M]
        U[M:, :nbf] = num.dot(dagger(c), V_ni[M:])
        U[:, :nbf] = gram_schmidt_orthonormalize(U[:, :nbf])
        if nbf < M + L:
            for i in xrange(nbf, M + L):
                for j in xrange(M + L):
                    U[j, i] = random()
            U = gram_schmidt_orthonormalize(U)
        check_ortho(U)
        c_k.append(c)
        U_k.append(U)
    return c_k, U_k

def check_ortho(U):
    nb = U.shape[1]
    diff = num.dot(dagger(U),U) - num.identity(nb, num.Float)
    max_diff = max(num.absolute(diff).flat)
    if  max_diff> 1e-3:
        print 'ERROR: Columns of c are not orthogonal by', max_diff
