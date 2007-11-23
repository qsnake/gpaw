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

        #
        # to allow spin polarized calclulation
        #
        nkpts = paw.nkpts
        if not paw.spinpol:
            nocc = int(paw.nvalence / 2)
            self.list_kpts = range(nkpts)
        else:
            self.list_kpts=[]

            #find kpoints with up spin 
            for i,kpt in  enumerate(paw.kpt_u):
                if kpt.s ==0:
                    self.list_kpts.append(i)
                print self.list_kpts
            assert len(self.list_kpts) == nkpts
                        
            #find number of occupied orbitals, if no fermi smearing
            nocc = 0.
            for i in self.list_kpts:
                nocc += sum(paw.kpt_u[i].f_n)
            nocc = int(nocc + 0.5)
            print "nocc", nocc



                 
        for nucleus in paw.nuclei:
            #print "i"
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
            raise RuntimeError("wrong keyword for 'mode', use 'xas', 'xes' or 'all'")

        self.n = n

        
            
        self.eps_n = num.empty(nkpts * n, num.Float)
        self.sigma_cn = num.empty((3, nkpts * n), num.Complex)
        n1 = 0
        for k in self.list_kpts:
            n2 = n1 + n
            self.eps_n[n1:n2] = paw.kpt_u[k].eps_n[n_start:n_end] * paw.Ha
            P_ni = nucleus.P_uni[k, n_start:n_end]
            a_cn = inner(A_ci, P_ni)
            print "weight", paw.weight_k[k]
            self.sigma_cn[:, n1:n2] = paw.weight_k[k] ** 0.5 * a_cn #.real
            n1 = n2

        self.symmetry = paw.symmetry

    def get_spectra(self, fwhm=0.5, linbroad=None, N=1000, kpoint=None,
                    proj=None,  proj_xyz=True, stick=False):
        """Calculate spectra.

        Parameters:
        
        fwhm:
          the full width half maximum in eV for gaussian broadening
        linbroad:
          a list of three numbers, the first fwhm2, the second the value
          where the linear increase starts and the third the value where
          the broadening has reached fwhm2. example [0.5, 540, 550]
        N:
          the number of bins in the broadened spectrum
        kpoint:
          select a specific k-point to calculate spectrum for
        proj:
          a list of vectors to project the transition dipole on. Default
          is None then only x,y,z components are calculated.  a_stick and
          a_c squares of the transition moments in resp. direction
        proj_xyz:
          if True keep projections in x, y and z. a_stck and a_c will have
          length 3 + len(proj). if False only those projections
          defined by proj keyword, a_stick and a_c will have length len(proj)
        stick:
          if False return broadened spectrum, if True return stick spectrum
          
        Symmtrization has been moved inside get_spectra because we want to
        symmtrice squares of transition dipoles."""
        
        # eps_n = self.eps_n[k_in*self.n: (k_in+1)*self.n -1]
         

        # proj keyword, check normalization of incoming vectors
        if proj is not None:
            proj_2 = num.array(proj,num.Float)
            if len(proj_2.shape) == 1:
                proj_2 = num.array([proj],num.Float)

            for i,p in enumerate(proj_2):
                if sum(p ** 2) ** 0.5 != 1.0:
                    print "proj_2 %s not normalized" %i
                    proj_2[i] /=  sum(p ** 2) ** 0.5
        
            # make vector of projections
            if proj_xyz:
                sigma1_cn = num.empty( (3 + proj_2.shape[0], self.sigma_cn.shape[1]),
                                       num.Complex)
                sigma1_cn[0:3,:] = self.sigma_cn
                for i,p in enumerate(proj_2):
                    sigma1_cn[3 +i,:] = num.dot(p,self.sigma_cn) 
            else:
                sigma1_cn = num.empty((proj_2.shape[0], self.sigma_cn.shape[1]) ,
                                       num.Complex)
                for i,p in enumerate(proj_2):
                    sigma1_cn[i,:] = num.dot(p, self.sigma_cn)
                                                
            sigma2_cn = num.empty(sigma1_cn.shape, num.Float)
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
                 tol=1e-10, maxiter=100, proj=None,
                 proj_xyz=True, inverse_overlap="exact"):

        self.paw = paw
        if paw is not None:
            assert not paw.spinpol # restricted - for now

            self.weight_k = paw.weight_k
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

        if inverse_overlap == "exact":
            self.solver = self.solve
        elif inverse_overlap == "approximate":
            self.solver = self.solve2
        elif inverse_overlap == "noinverse":
            self.solver = self.solve3
        else:
            raise RuntimeError("""Error, inverse_solver must be either 'exact',
            'approximate' or 'noinverse' """)
            
        if filename is not None:
            self.read(filename)
            if paw is not None:
                self.allocate_tmp_arrays()
        else:
            self.initialize_start_vector(proj=proj,proj_xyz=proj_xyz)

    def read(self, filename):
        data = pickle.load(open(filename))
        self.nkpts = data['nkpts']
        self.swaps = data['swaps']
        self.weight_k = data['weight_k']
        self.dim = data['dim']
        k1, k2 = self.k1, self.k2
        if k2 is None:
            k2 = self.nkpts
        a_kci, b_kci = data['ab']
        self.a_uci = a_kci[k1:k2].copy()
        self.b_uci = b_kci[k1:k2].copy()
        if self.paw is not None and 'arrays' in data:
            print 'reading arrays'
            w_kcG, wold_kcG, y_kcG = data['arrays']
            i = [slice(k1, k2), slice(0, self.dim)] + self.paw.gd.get_slice()
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
                a_kci = num.empty((self.nkpts, self.dim, ni), self.paw.typecode)
                b_kci = num.empty((self.nkpts, self.dim, ni), self.paw.typecode)
                kpt_comm.gather(self.a_uci, MASTER, a_kci)
                kpt_comm.gather(self.b_uci, MASTER, b_kci)
                data = {'ab': (a_kci, b_kci),
                        'nkpts': self.nkpts,
                        'swaps': self.swaps,
                        'weight_k':self.weight_k,
                        'dim':self.dim}
            else:
                kpt_comm.gather(self.a_uci, MASTER)
                kpt_comm.gather(self.b_uci, MASTER)
            
        if mode == 'all':
            w0_ucG = gd.collect(self.w_ucG)
            wold0_ucG = gd.collect(self.wold_ucG)
            y0_ucG = gd.collect(self.y_ucG)
            if gd.comm.rank == MASTER:
                if kpt_comm.rank == MASTER:
                    w_kcG = gd.empty((self.nkpts, self.dim), self.paw.typecode,
                                     global_array=True)
                    wold_kcG = gd.empty((self.nkpts, self.dim), self.paw.typecode,
                                        global_array=True)
                    y_kcG = gd.empty((self.nkpts, self.dim), self.paw.typecode,
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

    def allocate_tmp_arrays(self):
        
        self.tmp1_cG = self.paw.gd.zeros(self.dim, self.paw.typecode)
        self.tmp2_cG = self.paw.gd.zeros(self.dim, self.paw.typecode)
        self.z_cG = self.paw.gd.zeros(self.dim, self.paw.typecode)
        
    def initialize_start_vector(self, proj=None, proj_xyz=True):
        # proj is one list of vectors [[e1_x,e1_y,e1_z],[e2_x,e2_y,e2_z]]
        #( or [ex,ey,ez] if only one projection )
        # that the spectrum will be projected on 
        # default is to only calculate the averaged spectrum
        # if proj_xyz is True, keep projection in x,y,z, if False
        # only calculate the projections in proj
        
        # Create initial wave function:
        nmykpts = self.nmykpts
        
        for nucleus in self.paw.nuclei:
            if nucleus.setup.phicorehole_g is not None:
                break
        A_ci = nucleus.setup.A_ci

        #
        # proj keyword
        #

        #check normalization of incoming vectors
        if proj is not None:
            proj_2 = num.array(proj,num.Float)
            if len(proj_2.shape) == 1:
                proj_2 = num.array([proj],num.Float)
            
            for i,p in enumerate(proj_2):
                if sum(p ** 2) ** 0.5 != 1.0:
                    print "proj_2 %s not normalized" %i
                    proj_2[i] /=  sum(p ** 2) ** 0.5

            proj_tmp = []
            for p in proj_2:
               proj_tmp.append(num.dot(p, A_ci))
            proj_tmp = num.array(proj_tmp, num.Float)   

            # if proj_xyz is True, append projections to A_ci
            if proj_xyz:
                A_ci_tmp = num.zeros((3 + proj_2.shape[0], A_ci.shape[1]), num.Float)
                A_ci_tmp[0:3,:] = A_ci 
                A_ci_tmp[3:,:]= proj_tmp

            # otherwise, replace A_ci by projections
            else:
                A_ci_tmp = num.zeros((proj_2.shape[0], A_ci.shape[1]), num.Float)
                A_ci_tmp = proj_tmp
            A_ci = A_ci_tmp

        self.dim = len(A_ci)

        self.allocate_tmp_arrays()

        self.w_ucG = self.paw.gd.zeros((nmykpts, self.dim), self.paw.typecode)
        self.wold_ucG = self.paw.gd.zeros((nmykpts, self.dim), self.paw.typecode)
        self.y_ucG = self.paw.gd.zeros((nmykpts, self.dim), self.paw.typecode)
            
        self.a_uci = num.zeros((nmykpts, self.dim, 0), self.paw.typecode)
        self.b_uci = num.zeros((nmykpts, self.dim, 0), self.paw.typecode)
        
            
        if nucleus.pt_i is not None: # not all CPU's will have a contribution
            for u in range(nmykpts):
                nucleus.pt_i.add(self.w_ucG[u], A_ci, self.k1 + u) 

        print self.w_ucG.shape
        


        
    def run(self, nsteps):
        ni = self.a_uci.shape[2]
        a_uci = num.empty((self.nmykpts, self.dim, ni + nsteps), self.paw.typecode)
        b_uci = num.empty((self.nmykpts, self.dim, ni + nsteps), self.paw.typecode)
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
        
        self.solver(w_cG, self.z_cG, u)
        I_c = num.reshape(integrate(num.conjugate(z_cG) * w_cG)**-0.5,
                          (self.dim, 1, 1, 1))
        z_cG *= I_c
        w_cG *= I_c
        
        if i != 0:
            b_c =  1.0 / I_c 
        else:
            b_c = num.reshape(num.zeros(self.dim), (self.dim, 1, 1, 1))
    
        self.paw.kpt_u[u].apply_hamiltonian(self.paw.hamiltonian, 
                                            z_cG, y_cG)
        a_c = num.reshape(integrate(num.conjugate(z_cG) * y_cG), (self.dim, 1, 1, 1))
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

    def get_spectra(self, eps_s, delta=0.1, imax=None, kpoint=None, fwhm=None,
                    linbroad=None):
        assert not mpi.parallel
        
        n = len(eps_s)
                
        sigma_cn = num.zeros((self.dim, n), num.Float)
        if imax is None:
            imax = self.a_uci.shape[2]
        energyunit = units.GetEnergyUnit()
        Ha = Convert(1, 'Hartree', energyunit)
        eps_n = (eps_s + delta * 1.0j) / Ha
                
        # if a certain k-point is chosen
        if kpoint is not None:
             for c in range(self.dim):
                sigma_cn[c] += self.continued_fraction(eps_n, kpoint, c,
                                                       0, imax).imag
        else:
            for k in range(self.nkpts):
                weight = self.weight_k[k]
                for c in range(self.dim):
                    sigma_cn[c] += weight*self.continued_fraction(eps_n, k, c,
                                                               0, imax).imag

        if len(self.swaps) > 0:
            sigma0_cn = sigma_cn
            sigma_cn = num.zeros((self.dim, n), num.Float)
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
        # exact inverse overlap
        self.paw.kpt_u[u].apply_inverse_overlap(self.paw.pt_nuclei,
                                                w_cG, self.tmp1_cG)
        self.u = u
        CG(self, z_cG, self.tmp1_cG,
           tolerance=self.tol, maxiter=self.maxiter)

    def solve2(self, w_cG, z_cG, u):
        # approximate inverse overlap
        self.paw.kpt_u[u].apply_inverse_overlap(self.paw.pt_nuclei,
                                                w_cG, z_cG)
        self.u = u

    def solve3(self, w_cG, z_cG, u):
        # no inverse overlap
        z_cG[:] =  w_cG
        self.u = u

    

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
