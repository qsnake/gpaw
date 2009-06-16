import pickle
from math import log, pi, sqrt

import numpy as npy
from ase.units import Hartree

from gpaw.utilities.cg import CG
import gpaw.mpi as mpi


class XAS:
    def __init__(self, paw, mode="xas"):
        wfs = paw.wfs
        assert wfs.world.size == 1 #assert not mpi.parallel

        #
        # to allow spin polarized calclulation
        #
        nkpts = len(wfs.ibzk_kc)
        if wfs.nspins == 1:
            nocc = wfs.setups.nvalence // 2
            self.list_kpts = range(nkpts)
        else:
            self.list_kpts=[]

            #find kpoints with up spin 
            for i, kpt in  enumerate(wfs.kpt_u):
                if kpt.s == 0:
                    self.list_kpts.append(i)
                print self.list_kpts
            assert len(self.list_kpts) == nkpts
                        
            #find number of occupied orbitals, if no fermi smearing
            nocc = 0.
            for i in self.list_kpts:
                nocc += sum(wfs.kpt_u[i].f_n)
            nocc = int(nocc + 0.5)
            print "nocc", nocc
                 
        for a, setup in enumerate(wfs.setups):
            #print "i"
            if setup.phicorehole_g is not None:  
                break

        A_ci = setup.A_ci

        # xas, xes or all modes
        if mode == "xas":
            n_start = nocc
            n_end = wfs.nbands  
            n =  wfs.nbands - nocc
        elif mode == "xes":
            n_start = 0
            n_end = nocc  
            n = nocc
        elif mode == "all":
            n_start = 0
            n_end = wfs.nbands 
            n = wfs.nbands
        else:
            raise RuntimeError(
                "wrong keyword for 'mode', use 'xas', 'xes' or 'all'")

        self.n = n
            
        self.eps_n = npy.empty(nkpts * n)
        self.sigma_cn = npy.empty((3, nkpts * n), complex)
        n1 = 0
        for kpt in wfs.kpt_u:
            if kpt.s != 0:
                continue
            
            n2 = n1 + n
            self.eps_n[n1:n2] = kpt.eps_n[n_start:n_end] * Hartree
            P_ni = kpt.P_ani[a][n_start:n_end]
            a_cn = npy.inner(A_ci, P_ni)
            weight = kpt.weight * wfs.nspins / 2
            print "weight", weight
            self.sigma_cn[:, n1:n2] = weight**0.5 * a_cn #.real
            n1 = n2

        self.symmetry = wfs.symmetry

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
            proj_2 = npy.array(proj,float)
            if len(proj_2.shape) == 1:
                proj_2 = npy.array([proj],float)

            for i,p in enumerate(proj_2):
                if sum(p ** 2) ** 0.5 != 1.0:
                    print "proj_2 %s not normalized" %i
                    proj_2[i] /=  sum(p ** 2) ** 0.5
        
            # make vector of projections
            if proj_xyz:
                sigma1_cn = npy.empty( (3 + proj_2.shape[0], self.sigma_cn.shape[1]),
                                       complex)
                sigma1_cn[0:3,:] = self.sigma_cn
                for i,p in enumerate(proj_2):
                    sigma1_cn[3 +i,:] = npy.dot(p,self.sigma_cn) 
            else:
                sigma1_cn = npy.empty((proj_2.shape[0], self.sigma_cn.shape[1]) ,
                                       complex)
                for i,p in enumerate(proj_2):
                    sigma1_cn[i,:] = npy.dot(p, self.sigma_cn)
                                                
            sigma2_cn = npy.empty(sigma1_cn.shape)
            sigma2_cn = (sigma1_cn*npy.conjugate(sigma1_cn)).real

        else:
           sigma2_cn = (self.sigma_cn * npy.conjugate(self.sigma_cn)).real

        #print sigma2_cn

        # now symmetrize
        if kpoint is not None:
            if self.symmetry is not None:
                sigma0_cn = sigma2_cn.copy()
                sigma2_cn = npy.zeros((len(sigma0_cn), len(sigma0_cn[0])))
                swaps = {}  # Python 2.4: use a set
                for swap, mirror in self.symmetry.symmetries:
                    swaps[swap] = None
                for swap in swaps:
                    sigma2_cn += sigma0_cn.take(swap, axis=0)
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
            e = emin + npy.arange(N + 1) * ((emax - emin) / N)
            a_c = npy.zeros((len(sigma2_cn), N + 1))
            
            if linbroad is None:
                #constant broadening fwhm
                alpha = 4 * log(2) / fwhm**2
            
                for n, eps in enumerate(eps_n[eps_start:eps_end]):
                    x = -alpha * (e - eps)**2
                    x = npy.clip(x, -100.0, 100.0)
                    a_c += npy.outer(sigma2_cn[:, n + eps_start],
                                        (alpha / pi)**0.5 * npy.exp(x))
            else:

                # constant broadening fwhm until linbroad[1] and a
                # constant broadening over linbroad[2] with fwhm2=
                # linbroad[0]
                fwhm2 = linbroad[0]
                lin_e1 = linbroad[1]
                lin_e2 = linbroad[2]
                print "fwhm", fwhm, fwhm2, lin_e1, lin_e2
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
                    x = npy.clip(x, -100.0, 100.0)
                    a_c += npy.outer(sigma2_cn[:, n],
                                     (alpha / pi)**0.5 * npy.exp(x))
                
            return  e, a_c


class RecursionMethod:
    """This class implements the Haydock recursion method. """

    def __init__(self, paw=None, filename=None,
                 tol=1e-10, maxiter=100, proj=None,
                 proj_xyz=True):

        if paw is not None:
            wfs = paw.wfs
            assert wfs.nspins == 1 # restricted - for now
            self.wfs = wfs
            self.hamiltonian = paw.hamiltonian
            self.weight_k = wfs.weight_k
            self.nkpts = len(wfs.ibzk_kc)
            self.nmykpts = len(wfs.kpt_u)
            self.k1 = wfs.kpt_comm.rank * self.nmykpts
            self.k2 = self.k1 + self.nmykpts
            self.swaps = {}  # Python 2.4: use a set
            if wfs.symmetry is not None:
                for swap, mirror in wfs.symmetry.symmetries:
                    self.swaps[swap] = None
        else:
            self.k1 = 0
            self.k2 = None

        self.tol = tol
        self.maxiter = maxiter

     
        if filename is not None:
            self.read(filename)
            if wfs is not None:
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
        if self.wfs is not None and 'arrays' in data:
            print 'reading arrays'
            w_kcG, wold_kcG, y_kcG = data['arrays']
            i = [slice(k1, k2), slice(0, self.dim)] + self.wfs.gd.get_slice()
            self.w_ucG = w_kcG[i].copy()
            self.wold_ucG = wold_kcG[i].copy()
            self.y_ucG = y_kcG[i].copy()

    def write(self, filename, mode=''):
        assert self.wfs is not None
        kpt_comm = self.wfs.kpt_comm
        gd = self.wfs.gd

        if gd.comm.rank == 0:
            if kpt_comm.rank == 0:
                nmyu, dim, ni = self.a_uci.shape
                a_kci = npy.empty((kpt_comm.size, nmyu, dim, ni),
                                  self.wfs.dtype)
                b_kci = npy.empty((kpt_comm.size, nmyu, dim, ni),
                                  self.wfs.dtype)
                kpt_comm.gather(self.a_uci, 0, a_kci)
                kpt_comm.gather(self.b_uci, 0, b_kci)
                a_kci.shape = (self.nkpts, dim, ni)
                b_kci.shape = (self.nkpts, dim, ni)
                data = {'ab': (a_kci, b_kci),
                        'nkpts': self.nkpts,
                        'swaps': self.swaps,
                        'weight_k':self.weight_k,
                        'dim':dim}
            else:
                kpt_comm.gather(self.a_uci, 0)
                kpt_comm.gather(self.b_uci, 0)
            
        if mode == 'all':
            w0_ucG = gd.collect(self.w_ucG)
            wold0_ucG = gd.collect(self.wold_ucG)
            y0_ucG = gd.collect(self.y_ucG)
            if gd.comm.rank == 0:
                if kpt_comm.rank == 0:
                    w_kcG = gd.empty((self.nkpts, dim), self.wfs.dtype,
                                     global_array=True)
                    wold_kcG = gd.empty((self.nkpts, dim), self.wfs.dtype,
                                        global_array=True)
                    y_kcG = gd.empty((self.nkpts, dim), self.wfs.dtype,
                                     global_array=True)
                    kpt_comm.gather(w0_ucG, 0, w_kcG)
                    kpt_comm.gather(wold0_ucG, 0, wold_kcG)
                    kpt_comm.gather(y0_ucG, 0, y_kcG)
                    data['arrays'] = (w_kcG, wold_kcG, y_kcG)
                else:
                    kpt_comm.gather(w0_ucG, 0)
                    kpt_comm.gather(wold0_ucG, 0)
                    kpt_comm.gather(y0_ucG, 0)

        if self.wfs.world.rank == 0:
            pickle.dump(data, open(filename, 'w'))

    def allocate_tmp_arrays(self):
        
        self.tmp1_cG = self.wfs.gd.zeros(self.dim, self.wfs.dtype)
        self.tmp2_cG = self.wfs.gd.zeros(self.dim, self.wfs.dtype)
        self.z_cG = self.wfs.gd.zeros(self.dim, self.wfs.dtype)
        
    def initialize_start_vector(self, proj=None, proj_xyz=True):
        # proj is one list of vectors [[e1_x,e1_y,e1_z],[e2_x,e2_y,e2_z]]
        #( or [ex,ey,ez] if only one projection )
        # that the spectrum will be projected on 
        # default is to only calculate the averaged spectrum
        # if proj_xyz is True, keep projection in x,y,z, if False
        # only calculate the projections in proj
        
        # Create initial wave function:
        nmykpts = self.nmykpts
        
        for a, setup in enumerate(self.wfs.setups):
            if setup.phicorehole_g is not None:
                break
        A_ci = setup.A_ci

        #
        # proj keyword
        #

        #check normalization of incoming vectors
        if proj is not None:
            proj_2 = npy.array(proj,float)
            if len(proj_2.shape) == 1:
                proj_2 = npy.array([proj],float)
            
            for i,p in enumerate(proj_2):
                if sum(p ** 2) ** 0.5 != 1.0:
                    print "proj_2 %s not normalized" %i
                    proj_2[i] /=  sum(p ** 2) ** 0.5

            proj_tmp = []
            for p in proj_2:
               proj_tmp.append(npy.dot(p, A_ci))
            proj_tmp = npy.array(proj_tmp, float)   

            # if proj_xyz is True, append projections to A_ci
            if proj_xyz:
                A_ci_tmp = npy.zeros((3 + proj_2.shape[0], A_ci.shape[1]))
                A_ci_tmp[0:3,:] = A_ci 
                A_ci_tmp[3:,:]= proj_tmp

            # otherwise, replace A_ci by projections
            else:
                A_ci_tmp = npy.zeros((proj_2.shape[0], A_ci.shape[1]))
                A_ci_tmp = proj_tmp
            A_ci = A_ci_tmp

        self.dim = len(A_ci)

        self.allocate_tmp_arrays()

        self.w_ucG = self.wfs.gd.zeros((nmykpts, self.dim), self.wfs.dtype)
        self.wold_ucG = self.wfs.gd.zeros((nmykpts, self.dim), self.wfs.dtype)
        self.y_ucG = self.wfs.gd.zeros((nmykpts, self.dim), self.wfs.dtype)
            
        self.a_uci = npy.zeros((nmykpts, self.dim, 0), self.wfs.dtype)
        self.b_uci = npy.zeros((nmykpts, self.dim, 0), self.wfs.dtype)

        A_aci = self.wfs.pt.dict(3, zero=True)
        if a in A_aci:
            A_aci[a] = A_ci
        for u in range(nmykpts):
            self.wfs.pt.add(self.w_ucG[u], A_aci, u)

    def run(self, nsteps, inverse_overlap="exact"):

        if inverse_overlap == "exact":
            self.solver = self.solve
        elif inverse_overlap == "approximate":
            self.solver = self.solve2
        elif inverse_overlap == "noinverse":
            self.solver = self.solve3
        else:
            raise RuntimeError("""Error, inverse_solver must be either 'exact',
            'approximate' or 'noinverse' """)
            

        ni = self.a_uci.shape[2]
        a_uci = npy.empty((self.nmykpts, self.dim, ni + nsteps), self.wfs.dtype)
        b_uci = npy.empty((self.nmykpts, self.dim, ni + nsteps), self.wfs.dtype)
        a_uci[:, :, :ni]  = self.a_uci
        b_uci[:, :, :ni]  = self.b_uci
        self.a_uci = a_uci
        self.b_uci = b_uci

        for u in range(self.nmykpts):
            for i in range(nsteps):
                self.step(u, ni + i)

    def step(self, u, i):
        print u, i
        integrate = self.wfs.gd.integrate
        w_cG = self.w_ucG[u]
        y_cG = self.y_ucG[u]
        wold_cG = self.wold_ucG[u]
        z_cG = self.z_cG
        
        self.solver(w_cG, self.z_cG, u)
        I_c = npy.reshape(integrate(npy.conjugate(z_cG) * w_cG)**-0.5,
                          (self.dim, 1, 1, 1))
        z_cG *= I_c
        w_cG *= I_c
        
        if i != 0:
            b_c =  1.0 / I_c 
        else:
            b_c = npy.reshape(npy.zeros(self.dim), (self.dim, 1, 1, 1))
    
        self.hamiltonian.apply(z_cG, y_cG, self.wfs, self.wfs.kpt_u[u])
        a_c = npy.reshape(integrate(npy.conjugate(z_cG) * y_cG), (self.dim, 1, 1, 1))
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
                
        sigma_cn = npy.zeros((self.dim, n))
        if imax is None:
            imax = self.a_uci.shape[2]
        eps_n = (eps_s + delta * 1.0j) / Hartree
                
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
            sigma_cn = npy.zeros((self.dim, n))
            for swap in self.swaps:
                sigma_cn += sigma0_cn.take(swap, axis=0)
            sigma_cn /= len(self.swaps)

        # gaussian broadening 
        if fwhm is not None:
            sigma_tmp = npy.zeros(sigma_cn.shape)

            #constant broadening fwhm
            if linbroad is None:
                alpha = 4 * log(2) / fwhm**2
                for n, eps in enumerate(eps_s):
                    x = -alpha * (eps_s - eps)**2
                    x = npy.clip(x, -100.0, 100.0)
                    sigma_tmp += npy.outer(sigma_cn[:,n],
                                        (alpha / pi)**0.5 * npy.exp(x))

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
                    x = npy.clip(x, -100.0, 100.0)
                    sigma_tmp += npy.outer(sigma_cn[:, n],
                                        (alpha / pi)**0.5 * npy.exp(x))
            sigma_cn = sigma_tmp
                    

        return sigma_cn
    
    def solve(self, w_cG, z_cG, u):
        # exact inverse overlap
        self.wfs.overlap.apply_inverse(w_cG, self.tmp1_cG, self.wfs,
                                       self.wfs.kpt_u[u])
        self.u = u
        CG(self, z_cG, self.tmp1_cG,
           tolerance=self.tol, maxiter=self.maxiter)

    def solve2(self, w_cG, z_cG, u):
        # approximate inverse overlap
        self.wfs.overlap.apply_inverse(w_cG, z_cG, self.wfs, self.wfs.kpt_u[u])

        self.u = u

    def solve3(self, w_cG, z_cG, u):
        # no inverse overlap
        z_cG[:] =  w_cG
        self.u = u

    

    def sum(self, a):
        self.wfs.gd.comm.sum(a)
        return a
    
    def __call__(self, in_cG, out_cG):
        """Function that is called by CG. It returns S~-1Sx_in in x_out
        """

        kpt = self.wfs.kpt_u[self.u]
        self.wfs.overlap.apply(in_cG, self.tmp2_cG, self.wfs, kpt)
        self.wfs.overlap.apply_inverse(self.tmp2_cG, out_cG, self.wfs, kpt)

    def terminator(self, a, b, e):
        """ Analytic formula to terminate the continued fraction from
        [R Haydock, V Heine, and M J Kelly, J Phys. C: Solid State Physics, Vol 8, (1975), 2591-2605]
        """

        return 0.5 * (e - a - ((e - a)**2 - 4 * b**2)**0.5 / b**2)
