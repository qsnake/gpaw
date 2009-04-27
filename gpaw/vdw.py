# -*- coding: utf-8 -*-

"""Van der Waals density functional.

This module implements the Dion-Rydberg-Schröder-Langreth-Lundqvist
XC-functional.  There are two implementations:

1. A simlpe real-space double sum.

2. A more efficient FFT implementation based on the Román-Péres-Soler paper.

"""

import os
import sys
import pickle
from math import sin, cos, exp, pi, log, sqrt, ceil

import numpy as np
from numpy.fft import fftn, fftfreq, fft, ifftn

from gpaw.xc_functional import XCFunctional
from gpaw.operators import Gradient
import gpaw.mpi as mpi
import _gpaw
 
 
def T(w, x, y, z): 
    return 0.5 * ((1.0 / (w + x) + 1.0 / (y + z)) * 
                  (1.0 / ((w + y) * (x + z)) + 1.0 / ((w + z) * (y + x)))) 

def W(a, b): 
    return 2 * ((3 - a**2) * b * cos(b) * sin(a) + 
                (3 - b**2) * a * cos(a) * sin(b) + 
                (a**2 + b**2 - 3) * sin(a) * sin(b) - 
                3 * a * b * cos(a) * cos(b)) / (a * b)**3 
eta = 8 * pi / 9 
def nu(y, d): 
    return 0.5 * y**2 / (1 - exp(-0.5 * eta * (y / d)**2))

def f(a, b, d, dp): 
    va = nu(a, d) 
    vb = nu(b, d) 
    vpa = nu(a, dp) 
    vpb = nu(b, dp) 
    return 2 * (a * b)**2 * W(a, b) * T(va, vb, vpa, vpb) / pi**2

def phi(d, dp):
    """vdW-DF kernel."""
    from scipy.integrate import quad 
    kwargs = dict(epsabs=1.0e-6, epsrel=1.0e-6, limit=400)
    cut = 35
    return quad(lambda y: quad(f, 0, cut, (y, d, dp), **kwargs)[0],
                0, cut, **kwargs)[0]

C = 12 * (4 * pi / 9)**3
def phi_asymptotic(d, dp):
    """Asymptotic behavior of vdW-DF kernel."""
    d2 = d**2
    dp2 = dp**2
    return -C / (d2 * dp2 * (d2 + dp2)) 

def hRPS(x, xc=1.0):
    """Cutoff function from Román-Péres-Soler paper."""
    x1 = x / xc
    xm = x1 * 1.0
    y = -x1
    z = 1.0 + x1
    for m in range(2, 13):
        xm *= x1
        y -= xm / m
        if m < 12:
            z += xm
    y = np.exp(y)
    return xc * (1.0 - y), z * y

Zab = -0.8491


class VDWFunctional:
    """Base class for vdW-DF."""
    def __init__(self, nspins=1, world=None, q0cut=5.0,
                 phi0=0.5, ds=1.0, Dmax=20.0, nD=201, ndelta=21,
                 soft_correction=False, verbose=False):
        """vdW-DF.

        parameters:

        nspins: int
            Number of spins.
        world: MPI communicator
            Communicator to parallelize over.  Defaults to gpaw.mpi.world.
        q0cut: float
            Maximum value for q0.
        phi0: float
            Smooth value for phi(0,0).
        ds: float
            Cutoff for smooth kernel.
        Dmax: float
            Maximum value for D.
        nD: int
            Number of values for D in kernel-table.
        ndelta: int
            Number of values for delta in kernel-table.
        soft_correction: bool
            Correct for soft kernel.
        verbose: bool
            Print useful information.
        """
        
        if world is None:
            self.world = mpi.world
        else:
            self.world = world

        self.q0cut = q0cut
        self.phi0 = phi0
        self.ds = ds

        self.delta_i = np.linspace(0, 1.0, ndelta)
        self.D_j = np.linspace(0, Dmax, nD)

        self.verbose = verbose
        
        self.revPBEx = XCFunctional('revPBEx', nspins)
        self.LDAc = XCFunctional('None-C_PW', nspins)

        self.read_table()

        self.soft_correction = soft_correction
        if soft_correction:
            dD = self.D_j[1]
            self.C_soft = np.dot(self.D_j**2, self.phi_ij[0]) * 4 * pi * dD
            
        self.gga = True
        self.mgga = not True
        self.hybrid = 0.0
        self.uses_libxc = False
        self.gllb = False
        self.xcname = 'vdw-DF'

        self.gd = None
        self.energy_only = False
        
    def set_grid_descriptor(self, gd):
        if gd.is_non_orthogonal():
            raise NotImplementedError("vdW calculations require an orthogonal cell.")
        self.gd = gd

    def set_non_local_things(self, density, hamiltonian, wfs, atoms,
                             energy_only=False):
        self.set_grid_descriptor(density.finegd)
        self.energy_only = energy_only
        
    def is_gllb(self):
        return False

    def get_name(self):
        return 'vdW-DF'

    def get_setup_name(self):
        return 'revPBE'
    
    def apply_non_local(self, kpt):
        pass

    def get_non_local_kinetic_corrections(self):
        return 0.0

    def adjust_non_local_residual(self, pR_G, dR_G, kpt, n):
        pass

    def get_non_local_force(self, kpt):
        return 0.0
    
    def get_non_local_energy(self, n_g=None, a2_g=None, e_LDAc_g=None,
                             v_LDAc_g=None, v_g=None, deda2_g=None):
        """Calculate non-local correlation energy.

        parameters:

        n_g: ndarray
            Density.
        a2_g: ndarray
            Absolute value of the gradient of the density - squared.
        e_LDAc_g: ndarray
            LDA correlation energy density.
        """
        
        if n_g is None:
            return 0.0
        
        gd = self.gd
        
        if a2_g is None:
            # Calculate square of gradient:
            a2_g = np.zeros_like(n_g)
            dndx_g = np.zeros_like(n_g)
            for c in range(3):
                Gradient(gd, c).apply(n_g, dndx_g)
                a2_g += dndx_g**2

        n_g = n_g.clip(1e-7, np.inf)
        
        if e_LDAc_g is None:
            # Calculate LDA correlation energy density:
            e_LDAc_g = np.empty_like(n_g)
            v_g = np.empty_like(n_g)
            self.LDAc.calculate_spinpaired(e_LDAc_g, n_g, v_g)
        else:
            e_LDAc_g.shape = n_g.shape

        # Calculate q0 and cut it off smoothly at q0cut:
        kF_g = (3 * pi**2 * n_g)**(1.0 / 3.0)
        q0_g, dhdx_g = hRPS(kF_g -
                            4 * pi / 3 * e_LDAc_g / n_g -
                            Zab / 36 / kF_g * a2_g / n_g**2, self.q0cut)

        if self.verbose:
            print ('VDW: q0 (min, mean, max): (%f, %f, %f)' %
                   (q0_g.min(), q0_g.mean(), q0_g.max()))
        
        if self.soft_correction:
            dEcnl = gd.integrate(n_g**2 / q0_g**3) * 0.5 * self.C_soft
        else:
            dEcnl = 0.0
            
        # Distribute density and q0 to all processors:
        n_g = gd.collect(n_g, broadcast=True)
        q0_g = gd.collect(q0_g, broadcast=True)

        if not self.energy_only:
            self.dhdx_g = gd.collect(dhdx_g, broadcast=True)

        Ecnl = self.calculate_6d_integral(n_g, q0_g, a2_g, e_LDAc_g, v_LDAc_g,
                                          v_g, deda2_g)
        return Ecnl + dEcnl
    
    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g, deda2_g):
        """Calculate energy and potential."""
        # LDA correlation:
        e_LDAc_g = np.empty_like(e_g)
        v_LDAc_g = np.zeros_like(v_g)
        self.LDAc.calculate_spinpaired(e_LDAc_g, n_g, v_LDAc_g)
        v_g += v_LDAc_g

        # revPBE exchange:
        self.revPBEx.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
        e_g += e_LDAc_g
        if n_g.ndim == 3:
            # Non-local part:
            e = self.get_non_local_energy(n_g, a2_g, e_LDAc_g, v_LDAc_g,
                                          v_g, deda2_g)
            if self.gd.comm.rank == 0:
                assert e_g.ndim == 1
                e_g[0] += e / self.gd.dv

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                                a2_g, aa2_g, ab2_g,
                                deda2_g, dedaa2_g, dedab2_g):
        raise NotImplementedError("Spin polarized self-consistent vdW calculation not supported yet.")
        """Calculate energy and potential."""
        # LDA correlation:
        e_LDAc_g = np.empty_like(e_g)
        va_LDAc_g = np.zeros_like(va_g)
        vb_LDAc_g = np.zeros_like(vb_g)
        v_LDAc_g = np.zeros_like(va_g)
        self.LDAc.calculate_spinpolarized(e_LDAc_g, na_g, va_LDAc_g, nb_g, vb_LDAc_g)
        va_g += va_LDAc_g
        vb_g += vb_LDAc_g

        # revPBE exchange:
        self.revPBEx.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                             a2_g, aa2_g, ab2_g,
                                             deda2_g, dedaa2_g, dedab2_g)
        e_g += e_LDAc_g 

        if na_g.ndim == 3:
            # Non-local part:
            v_g = np.zeros_like(va_g)
            deda2nl_g = np.zeros_like(deda2_g)
            e = self.get_non_local_energy(na_g + nb_g, a2_g, e_LDAc_g,
                                          (va_LDAc_g + vb_LDAc_g)/2.0,
                                          v_g, deda2nl_g) 
            deda2_g += deda2nl_g
            dedaa2_g += deda2nl_g / 4.0  
            dedab2_g += deda2nl_g / 4.0
            va_g += v_g 
            vb_g += v_g 
            if self.gd.comm.rank == 0:
                assert e_g.ndim == 1
                e_g[0] += e / self.gd.dv 

    def read_table(self):
        name = os.path.join(os.environ.get('GPAW_VDW', '.'),
                            'phi-%.3f-%.3f-%.3f-%d-%d.pckl' %
                            (self.phi0, self.ds, self.D_j[-1],
                             len(self.delta_i), len(self.D_j)))
        try:
            self.phi_ij = pickle.load(open(name))
            if self.verbose:
                print 'VDW: using', name
        except IOError:
            print 'VDW: No such file:', name
            self.make_table(name)
            
    def make_table(self, name):
        print 'VDW: Generating vdW-DF kernel ...'
        print 'VDW:',
        ndelta = len(self.delta_i)
        nD = len(self.D_j)
        self.phi_ij = np.zeros((ndelta, nD))
        for i in range(self.world.rank, ndelta, self.world.size):
            print ndelta - i,
            sys.stdout.flush()
            delta = self.delta_i[i]
            for j in range(nD - 1, -1, -1):
                D = self.D_j[j]
                d = D * (1.0 + delta)
                dp = D * (1.0 - delta)
                if d**2 + dp**2 > self.ds**2:
                    self.phi_ij[i, j] = phi(d, dp)
                else:
                    P = np.polyfit([0, self.D_j[j + 1]**2, self.D_j[j + 2]**2],
                                   [self.phi0,
                                    self.phi_ij[i, j + 1],
                                    self.phi_ij[i, j + 2]],
                                   2)
                    self.phi_ij[i, :j + 3] = np.polyval(P, self.D_j[:j + 3]**2)
                    break

        self.world.sum(self.phi_ij)
        
        print
        print 'VDW: Done!'
        if self.world.rank == 0:
            pickle.dump(self.phi_ij, open(name, 'w'), pickle.HIGHEST_PROTOCOL)

    def make_prl_plot(self, multiply_by_4_pi_D_squared=True):
        import pylab as plt
        x = np.linspace(0, 8.0, 100)
        for delta in [0, 0.5, 0.9]:
            y = [self.phi(D * (1.0 + delta), D * (1.0 - delta))
                 for D in x]
            if multiply_by_4_pi_D_squared:
                y *= 4 * pi * x**2
            plt.plot(x, y, label=r'$\delta=%.1f$' % delta)
        plt.legend(loc='best')
        plt.plot(x, np.zeros(len(x)), 'k-')       
        plt.xlabel('D')
        plt.ylabel(r'$4\pi D^2 \phi(\rm{Hartree})$')
        plt.show()

    def phi(self, d, dp):
        """Kernel function.

        Uses bi-linear interpolation and returns zero for D > Dmax.
        """
        
        P = self.phi_ij
        D = (d + dp) / 2.0
        if D < 1e-14:
            return P[0, 0]
        if D >= self.D_j[-1]:
            return 0.0
        
        delta = abs((d - dp) / (2 * D))
        ddelta = self.delta_i[1]
        x = delta / ddelta
        i = int(x)
        if i == len(self.delta_i) - 1:
            i -= 1
            x = 1.0
        else:
            x -= i

        dD = self.D_j[1]
        y = D / dD
        j = int(y)
        y -= j
        return (x * (y * P[i + 1, j + 1] +
                     (1 - y) * P[i + 1, j]) +
                (1 - x) * (y * P[i, j + 1] +
                           (1 - y) * P[i, j]))


class RealSpaceVDWFunctional(VDWFunctional):
    """Real-space implementation of vdW-DF."""
    def __init__(self, nspins=1, repeat=None, ncut=0.0005, **kwargs):
        """Real-space vdW-DF.

        parameters:

        repeat: 3-tuple
            Repeat the unit cell.
        ncut: float
            Density cutoff.
        """
        
        VDWFunctional.__init__(self, nspins, **kwargs)
        self.repeat = repeat
        self.ncut = ncut
        
    def calculate_6d_integral(self, n_g, q0_g,
                              a2_g=None, e_LDAc_g=None, v_LDAc_g=None,
                              v_g=None, deda2_g=None):
        """Real-space double-sum."""
        gd = self.gd
        n_c = n_g.shape
        R_gc = np.empty(n_c + (3,))
        R_gc[..., 0] = (np.arange(0, n_c[0]) * gd.h_c[0]).reshape((-1, 1, 1))
        R_gc[..., 1] = (np.arange(0, n_c[1]) * gd.h_c[1]).reshape((-1, 1))
        R_gc[..., 2] = np.arange(0, n_c[2]) * gd.h_c[2]

        mask_g = (n_g.ravel() > self.ncut)
        R_ic = R_gc.reshape((-1, 3)).compress(mask_g, axis=0)
        n_i = n_g.ravel().compress(mask_g)
        q0_i = q0_g.ravel().compress(mask_g)

        # Number of grid points:
        ni = len(n_i)

        if self.verbose:
            print 'VDW: number of points:', ni
            
        # Number of pairs per processor:
        world = self.world
        p = ni * (ni - 1) // 2 // world.size
        
        iA = 0
        for r in range(world.size):
            iB = iA + int(0.5 - iA + sqrt((iA - 0.5)**2 + 2 * p))
            if r == world.rank:
                break
            iA = iB

        assert iA <= iB
        
        if world.rank == world.size - 1:
            iB = ni

        if self.repeat is None:
            repeat_c = np.zeros(3, int)
        else:
            repeat_c = np.asarray(self.repeat, int)

        self.rhistogram = np.zeros(200)
        self.Dhistogram = np.zeros(200)
        dr = 0.05
        dD = 0.05
        E_vdwnl = _gpaw.vdw(n_i, q0_i, R_ic, gd.cell_c,
                            gd.pbc_c,
                            repeat_c,
                            self.phi_ij, self.delta_i[1], self.D_j[1],
                            iA, iB,
                            self.rhistogram, dr,
                            self.Dhistogram, dD)
        self.rhistogram *= gd.dv**2 / dr
        self.Dhistogram *= gd.dv**2 / dD
        self.world.sum(self.rhistogram)
        self.world.sum(self.Dhistogram)
        E_vdwnl = self.world.sum(E_vdwnl * gd.dv**2)
        return E_vdwnl


class FFTVDWFunctional(VDWFunctional):
    """FFT implementation of vdW-DF."""
    def __init__(self, nspins=1,
                 Nalpha=20, lambd=1.2, rcut=125.0, Nr=2048, size=None,
                 **kwargs):
        """FFT vdW-DF.

        parameters:

        Nalpha: int
            Number of interpolating cubic splines.
        lambd: float
            Parameter for defining geometric series of interpolation points.
        rcut: float
            Cutoff for kernel function.
        Nr: int
            Number of real-space points for kernel function.
        size: 3-tuple
            Size of FFT-grid.
        """
        
        VDWFunctional.__init__(self, nspins, **kwargs)
        self.Nalpha = Nalpha
        self.lambd = lambd
        self.rcut = rcut
        self.Nr = Nr
        self.size = size
        
        self.C_aip = None
        self.phi_aajp = None
        
    def construct_cubic_splines(self):
        """Construc interpolating splines for q0.

        The recipe is from

          http://en.wikipedia.org/wiki/Spline_(mathematics)
        """
        
        n = self.Nalpha
        lambd = self.lambd
        q1 = self.q0cut * (lambd - 1) / (lambd**(n - 1) - 1)
        q = q1 * (lambd**np.arange(n) - 1) / (lambd - 1)

        if self.verbose:
            print ('VDW: using %d qubic splines: 0.00, %.2f, ..., %.2f, %.2f' %
                   (n, q1, q[-2], q[-1]))
            
        y = np.eye(n)
        a = y
        h = q[1:] - q[:-1]
        alpha = 3 * ((a[2:] - a[1:-1]) / h[1:, np.newaxis] -
                     (a[1:-1] - a[:-2]) / h[:-1, np.newaxis])
        l = np.ones((n, n))
        mu = np.zeros((n, n))
        z = np.zeros((n, n))
        for i in range(1, n - 1):
            l[i] = 2 * (q[i + 1] - q[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]
        b = np.zeros((n, n))
        c = np.zeros((n, n))
        d = np.zeros((n, n))
        for i in range(n - 2, -1, -1):
            c[i] = z[i] - mu[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / 3 / h[i]

        self.C_aip = np.zeros((n, n, 4))
        self.C_aip[:, :-1, 0] = a[:-1].T
        self.C_aip[:, :-1, 1] = b[:-1].T
        self.C_aip[:, :-1, 2] = c[:-1].T
        self.C_aip[:, :-1, 3] = d[:-1].T
        self.C_aip[-1, -1, 0] = 1.0
        self.q_a = q

    def p(self, alpha, q):
        """Interpolating spline."""
        i = int(log(q / self.q_a[1] * (self.lambd - 1) + 1) / log(self.lambd))
        a, b, c, d = self.C_aip[alpha, i]
        dq = q - self.q_a[i]
        return a + dq * (b + dq * (c + dq * d))

    def construct_fourier_transformed_kernels(self):
        self.phi_aajp = phi_aajp = {}
        M = self.Nr
        rcut = self.rcut
        r_g = np.linspace(0, rcut, M, 0)
        k_j = np.arange(M // 2) * (2 * pi / rcut)

        if self.verbose:
            print ("VDW: cutoff for fft'ed kernel: %.3f Hartree" %
                   (0.5 * k_j[-1]**2))
            
        for a in range(self.Nalpha):
            qa = self.q_a[a]
            for b in range(a, self.Nalpha):
                qb = self.q_a[b]
                phi_g = [self.phi(qa * r, qb * r) for r in r_g]
                phi_j = (fft(r_g * phi_g * 1j).real[:M // 2] *
                         (rcut / M * 4 * pi))
                phi_j[0] = np.dot(r_g, r_g * phi_g) * (rcut / M * 4 * pi)
                phi_j[1:] /= k_j[1:]
                phi_aajp[a, b] = phi_aajp[b, a] = spline(k_j, phi_j)

    def set_grid_descriptor(self, gd):
        if (self.gd is not None and
            (self.gd.N_c == gd.N_c).all() and
            (self.gd.pbc_c == gd.pbc_c).all() and
            (self.gd.cell_c == gd.cell_c).all()):
            return

        VDWFunctional.set_grid_descriptor(self, gd)

        if self.size is None:
            self.shape = gd.N_c.copy()
            for c, n in enumerate(self.shape):
                if not gd.pbc_c[c]:
                    self.shape[c] = int(2**ceil(log(n) / log(2)))
        else:
            self.shape = np.array(self.size)
            
        d_c = gd.cell_c / (2 * pi * gd.N_c)
        kx2 = fftfreq(self.shape[0], d_c[0]).reshape((-1,  1,  1))**2
        ky2 = fftfreq(self.shape[1], d_c[1]).reshape(( 1, -1,  1))**2
        kz2 = fftfreq(self.shape[2], d_c[2]).reshape(( 1,  1, -1))**2
        k_k = (kx2 + ky2 + kz2)**0.5
        self.dj_k = k_k / (2 * pi / self.rcut)
        self.j_k = self.dj_k.astype(int)
        self.dj_k -= self.j_k
        self.dj_k *= 2 * pi / self.rcut

        if self.verbose:
            print 'VDW: density array size:', gd.get_size_of_global_array()
            print 'VDW: zero-padded array size:', self.shape
            print ('VDW: maximum kinetic energy: %.3f Hartree' %
                   (0.5 * k_k.max()**2))

    def calculate_6d_integral(self, n_g, q0_g,
                              a2_g=None, e_LDAc_g=None, v_LDAc_g=None,
                              v_g=None, deda2_g=None):
        if self.C_aip is None:
            self.construct_cubic_splines()
            self.construct_fourier_transformed_kernels()

        gd = self.gd
        N = self.Nalpha

        world = self.world

        i_g = (np.log(q0_g / self.q_a[1] * (self.lambd - 1) + 1) /
               log(self.lambd)).astype(int)
            
        dq0_g = q0_g - self.q_a[i_g]
        
        if self.verbose:
            print 'VDW: fft:',
            
        theta_ak = {}
        p_ag = {}
        for a in range(world.rank, N, world.size):
            C_pg = self.C_aip[a, i_g].transpose((3, 0, 1, 2))
            pa_g = (C_pg[0] + dq0_g *
                    (C_pg[1] + dq0_g *
                     (C_pg[2] + dq0_g * C_pg[3])))
            theta_ak[a] = fftn(n_g * pa_g, self.shape).copy()

            if not self.energy_only:
                p_ag[a] = pa_g
                
            if self.verbose:
                print a,
                sys.stdout.flush()

        if self.verbose:
            print
            print 'VDW: convolution:',

        F_ag = {}
        dj_k = self.dj_k
        energy = 0.0
        for a in range(N):
            ranka = a % world.size
            Fa_k = np.zeros(self.shape, complex)
            for b in range(world.rank, N, world.size):
                _gpaw.vdw2(self.phi_aajp[a, b], self.j_k, dj_k,
                           theta_ak[b], Fa_k)

            self.world.sum(Fa_k, ranka)
            if world.rank == ranka:
                energy += np.vdot(theta_ak[a], Fa_k).real

                if not self.energy_only:
                    n1, n2, n3 = gd.get_size_of_global_array()
                    F_ag[a] = ifftn(Fa_k).real[:n1, :n2, :n3].copy()
                
            if self.verbose:
                print a,
                sys.stdout.flush()

        if self.verbose:
            print

        if not self.energy_only:
            self.calculate_potential(n_g, a2_g, i_g, dq0_g, p_ag, F_ag,
                                     e_LDAc_g, v_LDAc_g,
                                     v_g, deda2_g)

        return 0.5 * world.sum(energy) * gd.dv / self.shape.prod()

    def calculate_potential(self, n_g, a2_g, i_g, dq0_g, p_ag, F_ag,
                            e_LDAc_g, v_LDAc_g, v_g, deda2_g):
        N = self.Nalpha
        world = self.world

        a2_g = self.gd.collect(a2_g, broadcast=True)
        e_LDAc_g = self.gd.collect(e_LDAc_g, broadcast=True)
        v_LDAc_g = self.gd.collect(v_LDAc_g, broadcast=True)
        dq0dn_g = ((pi / 3 / n_g)**(2.0 / 3.0) +
                   4 * pi / 3 * (e_LDAc_g / n_g - v_LDAc_g) / n_g +
                   7 * Zab / 108 / (3 * pi**2)**(1.0 / 3.0) * a2_g *
                   n_g**(-10.0 / 3.0))
        dq0da2_g = -Zab / 36 / (3 * pi**2)**(1.0 / 3.0) / n_g**(7.0 / 3.0)
        v0_g = np.zeros_like(n_g)
        deda20_g = np.zeros_like(n_g)

        for a in range(world.rank, N, world.size):
            C_pg = self.C_aip[a, i_g].transpose((3, 0, 1, 2))
            dpadq0_g = C_pg[1] + dq0_g * (2 * C_pg[2] + 3 * dq0_g * C_pg[3])

            dthetaadn_g = p_ag[a] + n_g * dpadq0_g * dq0dn_g * self.dhdx_g
            v0_g += dthetaadn_g * F_ag[a]
            
            dthetaada2_g = n_g * dpadq0_g * dq0da2_g * self.dhdx_g
            deda20_g += dthetaada2_g * F_ag[a]

        world.sum(v0_g)
        world.sum(deda20_g)
        slice = self.gd.get_slice()
        v_g += v0_g[slice]
        deda2_g += deda20_g[slice]
        
def spline(x, y):
    n = len(y)
    result = np.zeros((n, 4))
    a, b, c, d = result.T
    a[:] = y
    h = x[1:] - x[:-1]
    alpha = 3 * ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1])
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / 3 / h[i]
    return result


if __name__ == '__main__':
    vdw = VDWFunctional()
