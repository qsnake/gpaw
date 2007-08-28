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
        # get the initial Wannier function from the calculator
        self.initialwannier = initialwannier
        raise NotImplementedError

    def InitializeRotationAndCoefficientMatrices(self):
        # Set ZIMatrix and ZIkMatrix to zero.
        self.InitZIMatrices()
        if not hasattr(self, 'initialwannier'):
            self.RandomizeMatrices(seed=self.seed)
        else:
            raise NotImplementedError
            #c , U = self.initialwannier. ...  # XXX
            self.SetListOfRotationMatrices(U)
            self.SetListOfCoefficientMatrices(c)
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
