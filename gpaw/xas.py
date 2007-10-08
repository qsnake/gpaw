import pickle
from math import log, pi, sqrt

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!
from ASE.Units import units, Convert

from gpaw.utilities.cg import CG
import gpaw.mpi as mpi
from gpaw.mpi import MASTER


class XAS:
    def __init__(self, paw, mode="xas"):
        assert paw.world.size == 1 #assert not mpi.parallel
        assert not paw.spinpol # restricted - for now

        nocc = int(paw.nvalence / 2)
        for nucleus in paw.nuclei:
            print "i"
            if nucleus.setup.phicorehole_g is not None:  
                break

        A_ci = nucleus.setup.A_ci

        # xas, xes or all modes
        if mode == "xas":
            n_start = nocc
            n_end = paw.nbands  
            n =  paw.nbands - nocc
        elif mode == "xes":
            n_start = 0
            n_end = nocc  
            n = nocc
        elif mode == "all":
            n_start = 0
            n_end = paw.nbands 
            n = paw.nbands
        else:
            print "wrong keyword for 'mode', use 'xas', 'xes' or 'all'"
            exit

        self.n = n

        nkpts =paw.nkpts
            
        self.eps_n = num.empty(nkpts * n, num.Float)
        self.sigma_cn = num.empty((3, nkpts * n), num.Complex)
        n1 = 0
        for k in range(nkpts):
            n2 = n1 + n
            self.eps_n[n1:n2] = paw.kpt_u[k].eps_n[n_start:n_end] * paw.Ha
            P_ni = nucleus.P_uni[k, n_start:n_end]
            a_cn = inner(A_ci, P_ni)
            #a_cn *= num.conjugate(a_cn)
            print paw.weight_k[k]
            self.sigma_cn[:, n1:n2] = paw.weight_k[k] ** 0.5 * a_cn #.real
            n1 = n2

        self.symmetry = paw.symmetry

    def get_spectra(self, fwhm=0.5, linbroad=None, N=1000, kpoint=None, proj=None,
                    stick=False):
        """ parameters:
            fwhm -     the full width half maximum in eV for gaussian broadening 
            linbroad - a list of three numbers, the first fwhm2, the second the value
                       where the linear increase starts and the third the value where
                       the broadening has reached fwhm2. example [0.5, 540, 550]
            N -        the number of bins in the broadened spectrum
            kpoint -   select a specific k-point to calculate spectrum for
            proj -     a list of vectors to project the transition dipole on. Default
                       is None then only x,y,z components are calculated.  a_stick and
                       a_c will have 3 + len(proj) dimensions and are squares of the
                       transition moments in resp. direction
            stick -    if False return broadened spectrum, if True return stick spectrum

            symmtrization has been moved inside get_spectra because we want to symmtrize
            squares of transition dipoles """
        
        # eps_n = self.eps_n[k_in*self.n: (k_in+1)*self.n -1]

         

        # proj keyword, check normalization of incoming vectors
        if proj is not None:
            proj_2 = num.array(proj,num.Float)
            for i,p in enumerate(proj_2):
                if sum(p ** 2) ** 0.5 != 1.0:
                    print "proj_2 %s not normalized" %i
                    proj_2[i] /=  sum(p ** 2) ** 0.5
        
            # make vector of projections
            sigma1_cn = num.empty( (3 + len(proj_2), len(self.sigma_cn[0])), num.Complex)
            sigma1_cn[0:3,:] = self.sigma_cn
            for i,p in enumerate(proj_2):
                sigma1_cn[3 +i,:] = num.dot(p,self.sigma_cn) 
        
            sigma2_cn = num.empty((len(sigma1_cn),len(sigma1_cn[0])),num.Float)
            sigma2_cn = (sigma1_cn*num.conjugate(sigma1_cn)).real

        else:
           sigma2_cn = (self.sigma_cn * num.conjugate(self.sigma_cn)).real

        #print sigma2_cn

        # now symmetrize
        if kpoint is not None:
            if self.symmetry is not None:
                sigma0_cn = sigma2_cn.copy()
                sigma2_cn = num.zeros((len(sigma0_cn),len(sigma0_cn[0])),num.Float)
                swaps = {}  # Python 2.4: use a set
                for swap, mirror in self.symmetry.symmetries:
                    swaps[swap] = None
                for swap in swaps:
                    sigma2_cn += num.take(sigma0_cn, swap)
                sigma2_cn /= len(swaps)
        
        eps_n = self.eps_n[:]
        if kpoint is not None:
            eps_start = kpoint*self.n
            eps_end = (kpoint+1)*self.n
        else: 
            eps_start = 0
            eps_end = len(self.eps_n)
            
       
        # return stick spectrum if stick=True
        if stick:
            e_stick = eps_n[eps_start:eps_end]
            a_stick = sigma2_cn[:,eps_start:eps_end]

            return e_stick, a_stick

        # else return broadened spectrum
        else:
            emin = min(eps_n) - 2 * fwhm
            emax = max(eps_n) + 2 * fwhm
            e = emin + num.arange(N + 1) * ((emax - emin) / N)
            a_c = num.zeros((len(sigma2_cn), N + 1), num.Float)
            
            if linbroad is None:
                #constant broadening fwhm
                alpha = 4 * log(2) / fwhm**2
            
                for n, eps in enumerate(eps_n[eps_start:eps_end]):
                    x = -alpha * (e - eps)**2
                    x = num.clip(x, -100.0, 100.0)
                    a_c += num.outerproduct(sigma2_cn[:, n + eps_start],
                                        (alpha / pi)**0.5 * num.exp(x))
            else:

                # constant broadening fwhm until linbroad[1] and a
                # constant broadening over linbroad[2] with fwhm2=
                # linbroad[0]
                fwhm2 = linbroad[0]
                lin_e1 = linbroad[1]
                lin_e2 = linbroad[2]
                for n, eps in enumerate(eps_n):
                    if eps < lin_e1:
                        alpha = 4*log(2) / fwhm**2
                    elif eps <=  lin_e2:
                        fwhm_lin = (fwhm + (eps - lin_e1) *
                                    (fwhm2 - fwhm) / (lin_e2 - lin_e1))
                        alpha = 4*log(2) / fwhm_lin**2
                    elif eps >= lin_e2:
                        alpha =  4*log(2) / fwhm2**2
                        
                        x = -alpha * (e - eps)**2
                        x = num.clip(x, -100.0, 100.0)
                        a_c += num.outerproduct(sigma2_cn[:, n],
                                        (alpha / pi)**0.5 * num.exp(x))
                
            return  e, a_c


class RecursionMethod:
    """This class implements the Haydock recursion method. """

    def __init__(self, paw=None, filename=None,
                 tol=1e-10, maxiter=100):

        self.paw = paw
        if paw is not None:
            assert not paw.spinpol # restricted - for now

            self.weight_k = paw.weight_k
            self.tmp1_cG = paw.gd.zeros(3, paw.typecode)
            self.tmp2_cG = paw.gd.zeros(3, paw.typecode)
            self.tmp3_cG = paw.gd.zeros(3, paw.typecode)
            self.z_cG = paw.gd.zeros(3, paw.typecode)
            self.nkpts = paw.nkpts
            self.nmykpts = paw.nmyu
            self.k1 = paw.kpt_comm.rank * self.nmykpts
            self.k2 = self.k1 + self.nmykpts
            self.swaps = {}  # Python 2.4: use a set
            if paw.symmetry is not None:
                for swap, mirror in paw.symmetry.symmetries:
                    self.swaps[swap] = None
        else:
            self.k1 = 0
            self.k2 = None

        self.tol = tol
        self.maxiter = maxiter
        
        if filename is not None:
            self.read(filename)
        else:
            self.initialize_start_vector()

    def read(self, filename):
        data = pickle.load(open(filename))
        self.nkpts = data['nkpts']
        self.swaps = data['swaps']
        self.weight_k = data['weight_k']
        k1, k2 = self.k1, self.k2
        if k2 is None:
            k2 = self.nkpts
        a_kci, b_kci = data['ab']
        self.a_uci = a_kci[k1:k2].copy()
        self.b_uci = b_kci[k1:k2].copy()
        if self.paw is not None and 'arrays' in data:
            print 'reading arrays'
            w_kcG, wold_kcG, y_kcG = data['arrays']
            i = [slice(k1, k2), slice(0, 3)] + self.paw.gd.get_slice()
            self.w_ucG = w_kcG[i].copy()
            self.wold_ucG = wold_kcG[i].copy()
            self.y_ucG = y_kcG[i].copy()

    def write(self, filename, mode=''):
        assert self.paw is not None
        kpt_comm = self.paw.kpt_comm
        gd = self.paw.gd

        if gd.comm.rank == MASTER:
            if kpt_comm.rank == MASTER:
                ni = self.a_uci.shape[2]
                a_kci = num.empty((self.nkpts, 3, ni), self.paw.typecode)
                b_kci = num.empty((self.nkpts, 3, ni), self.paw.typecode)
                kpt_comm.gather(self.a_uci, MASTER, a_kci)
                kpt_comm.gather(self.b_uci, MASTER, b_kci)
                data = {'ab': (a_kci, b_kci),
                        'nkpts': self.nkpts,
                        'swaps': self.swaps,
                        'weight_k':self.weight_k}
            else:
                kpt_comm.gather(self.a_uci, MASTER)
                kpt_comm.gather(self.b_uci, MASTER)
            
        if mode == 'all':
            w0_ucG = gd.collect(self.w_ucG)
            wold0_ucG = gd.collect(self.wold_ucG)
            y0_ucG = gd.collect(self.y_ucG)
            if gd.comm.rank == MASTER:
                if kpt_comm.rank == MASTER:
                    w_kcG = gd.empty((self.nkpts, 3), self.paw.typecode,
                                     global_array=True)
                    wold_kcG = gd.empty((self.nkpts, 3), self.paw.typecode,
                                        global_array=True)
                    y_kcG = gd.empty((self.nkpts, 3), self.paw.typecode,
                                     global_array=True)
                    kpt_comm.gather(w0_ucG, MASTER, w_kcG)
                    kpt_comm.gather(wold0_ucG, MASTER, wold_kcG)
                    kpt_comm.gather(y0_ucG, MASTER, y_kcG)
                    data['arrays'] = (w_kcG, wold_kcG, y_kcG)
                else:
                    kpt_comm.gather(w0_ucG, MASTER)
                    kpt_comm.gather(wold0_ucG, MASTER)
                    kpt_comm.gather(y0_ucG, MASTER)

        if self.paw.master:
            pickle.dump(data, open(filename, 'w'))
        
    def initialize_start_vector(self):
        # Create initial wave function:
        nmykpts = self.nmykpts
        self.w_ucG = self.paw.gd.zeros((nmykpts, 3), self.paw.typecode)
        for nucleus in self.paw.nuclei:
            if nucleus.setup.phicorehole_g is not None:
                break
        A_ci = nucleus.setup.A_ci
        if nucleus.pt_i is not None: # not all CPU's will have a contribution
            for u in range(nmykpts):
                nucleus.pt_i.add(self.w_ucG[u], A_ci, self.k1 + u)

        self.wold_ucG = self.paw.gd.zeros((nmykpts, 3), self.paw.typecode)
        self.y_ucG = self.paw.gd.zeros((nmykpts, 3), self.paw.typecode)
            
        self.a_uci = num.zeros((nmykpts, 3, 0), self.paw.typecode)
        self.b_uci = num.zeros((nmykpts, 3, 0), self.paw.typecode)
        
    def run(self, nsteps):
        ni = self.a_uci.shape[2]
        a_uci = num.empty((self.nmykpts, 3, ni + nsteps), self.paw.typecode)
        b_uci = num.empty((self.nmykpts, 3, ni + nsteps), self.paw.typecode)
        a_uci[:, :, :ni]  = self.a_uci
        b_uci[:, :, :ni]  = self.b_uci
        self.a_uci = a_uci
        self.b_uci = b_uci

        for u in range(self.paw.nmyu):
            for i in range(nsteps):
                self.step(u, ni + i)

    def step(self, u, i):
        print u, i
        integrate = self.paw.gd.integrate
        w_cG = self.w_ucG[u]
        y_cG = self.y_ucG[u]
        wold_cG = self.wold_ucG[u]
        z_cG = self.z_cG
        
        self.solve(w_cG, self.z_cG, u)
        I_c = num.reshape(integrate(num.conjugate(z_cG) * w_cG)**-0.5,
                          (3, 1, 1, 1))
        z_cG *= I_c
        w_cG *= I_c
        
        if i != 0:
            b_c =  1.0 / I_c 
        else:
            b_c = num.reshape(num.zeros(3), (3, 1, 1, 1))
    
        self.paw.kpt_u[u].apply_hamiltonian(self.paw.hamiltonian, 
                                            z_cG, y_cG)
        a_c = num.reshape(integrate(num.conjugate(z_cG) * y_cG), (3, 1, 1, 1))
        wnew_cG = (y_cG - a_c * w_cG - b_c * wold_cG)
        wold_cG[:] = w_cG
        w_cG[:] = wnew_cG
        self.a_uci[u, :, i] = a_c[:, 0, 0, 0]
        self.b_uci[u, :, i] = b_c[:, 0, 0, 0]


    def continued_fraction(self, e, k, c, i, imax):
        a_i = self.a_uci[k, c]
        b_i = self.b_uci[k, c]
        if i == imax - 2:
            return self.terminator(a_i[i], b_i[i], e)
        return 1.0 / (a_i[i] - e -
                      b_i[i + 1]**2 *
                      self.continued_fraction(e, k, c, i + 1, imax))

    def get_spectra(self, eps_s, delta=0.1, imax=None, kpoint=None, fwhm=None, linbroad=None):
        assert not mpi.parallel
        
        n = len(eps_s)
                
        sigma_cn = num.zeros((3, n), num.Float)
        if imax is None:
            imax = self.a_uci.shape[2]
        energyunit = units.GetEnergyUnit()
        Ha = Convert(1, 'Hartree', energyunit)
        eps_n = (eps_s + delta * 1.0j) / Ha
                
        # if a certain k-point is chosen
        if kpoint is not None:
             for c in range(3):
                sigma_cn[c] += self.continued_fraction(eps_n, kpoint, c,
                                                       0, imax).imag
        else:
            for k in range(self.nkpts):
                weight = self.weight_k[k]
                for c in range(3):
                    sigma_cn[c] += weight*self.continued_fraction(eps_n, k, c,
                                                               0, imax).imag

        if len(self.swaps) > 0:
            sigma0_cn = sigma_cn
            sigma_cn = num.zeros((3, n), num.Float)
            for swap in self.swaps:
                sigma_cn += num.take(sigma0_cn, swap)
            sigma_cn /= len(self.swaps)


        # gaussian broadening 
        if fwhm is not None:
            sigma_tmp = num.zeros(sigma_cn.shape, num.Float)

            #constant broadening fwhm
            if linbroad is None:
                alpha = 4 * log(2) / fwhm**2
                for n, eps in enumerate(eps_s):
                    x = -alpha * (eps_s - eps)**2
                    x = num.clip(x, -100.0, 100.0)
                    sigma_tmp += num.outerproduct(sigma_cn[:,n],
                                        (alpha / pi)**0.5 * num.exp(x))

            else:
                # constant broadening fwhm until linbroad[1] and a
                # constant broadening over linbroad[2] with fwhm2=
                # linbroad[0]
                fwhm2 = linbroad[0]
                lin_e1 = linbroad[1]
                lin_e2 = linbroad[2]
                for n, eps in enumerate(eps_s):
                    if eps < lin_e1:
                        alpha = 4*log(2) / fwhm**2
                    elif eps <=  lin_e2:
                        fwhm_lin = (fwhm + (eps - lin_e1) *
                                (fwhm2 - fwhm) / (lin_e2 - lin_e1))
                        alpha = 4*log(2) / fwhm_lin**2
                    elif eps >= lin_e2:
                        alpha =  4*log(2) / fwhm2**2

                    x = -alpha * (eps_s - eps)**2
                    x = num.clip(x, -100.0, 100.0)
                    sigma_tmp += num.outerproduct(sigma_cn[:, n],
                                        (alpha / pi)**0.5 * num.exp(x))
            sigma_cn = sigma_tmp
                    

        return sigma_cn
    
    def solve(self, w_cG, z_cG, u):
        self.paw.kpt_u[u].apply_inverse_overlap(self.paw.pt_nuclei,
                                                w_cG, self.tmp1_cG)
        self.u = u
        CG(self, z_cG, self.tmp1_cG,
           tolerance=self.tol, maxiter=self.maxiter)

    def sum(self, a):
        self.paw.gd.comm.sum(a)
        return a
    
    def __call__(self, in_cG, out_cG):
        """Function that is called by CG. It returns S~-1Sx_in in x_out
        """

        kpt = self.paw.kpt_u[self.u]
        kpt.apply_overlap(self.paw.pt_nuclei, in_cG, self.tmp2_cG)
        kpt.apply_inverse_overlap(self.paw.pt_nuclei, self.tmp2_cG, out_cG)

    def terminator(self, a, b, e):
        """ Analytic formula to terminate the continued fraction from
        [R Haydock, V Heine, and M J Kelly, J Phys. C: Solid State Physics, Vol 8, (1975), 2591-2605]
        """

        return 0.5 * (e - a - ((e - a)**2 - 4 * b**2)**0.5 / b**2)
