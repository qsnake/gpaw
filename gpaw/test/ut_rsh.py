#!/usr/bin/env python

import sys
import numpy as np
#import pylab as pl

if 0:
    import os, sys, time
    cmd = 'gdb %s %d' % (sys.executable, os.getpid())
    assert os.system('screen -dm -S dbg %s' % cmd) == 0
    time.sleep(3)

try:
    import fcntl, termios, struct
    _width = struct.unpack("hhhh", fcntl.ioctl(0, termios.TIOCGWINSZ, "\000"*8))[1]
    np.set_printoptions(linewidth=_width)
except IOError:
    pass

from gpaw import debug
from gpaw.mpi import world
from gpaw.utilities.tools import L_to_lm, lm_to_L

# -------------------------------------------------------------------

from gpaw.test.ut_common import ase_svnversion, shapeopt, TestCase, \
    TextTestRunner, CustomTextTestRunner, defaultTestLoader, \
    initialTestLoader, create_random_atoms, create_parsize_maxbands

memstats = False
if memstats:
    # Developer use of this feature requires ASE 3.1.0 svn.rev. 905 or later.
    assert ase_svnversion >= 905 # wasn't bug-free untill 973!
    from ase.utils.memory import MemorySingleton, MemoryStatistics

# -------------------------------------------------------------------

#from quadrature import nL, theta_L, phi_L, weight_L, \
#    condYdYdtheta, condYdYdphi, HitDetector

# Convert the 50 Lebedev quadrature points to angles (correct up to l=11)
from gpaw.sphere.lebedev import R_nv as points_Lv, weight_n as weight_L

assert points_Lv.shape == (50,3)
nL = len(points_Lv)
theta_L = np.arctan2((points_Lv[:,0]**2+points_Lv[:,1]**2)**0.5, points_Lv[:,2])
phi_L = np.arctan2(points_Lv[:,1],points_Lv[:,0])

from gpaw.sphere import lmiter
from gpaw.sphere.rsh import C, Y, dYdtheta, dYdphi, \
    intYY, intYY_ex, intYY_ey, intYY_ez, \
    intYdYdtheta_ex, intYdYdtheta_ey, intYdYdtheta_ez, \
    intYdYdphi_ex, intYdYdphi_ey, intYdYdphi_ez
from gpaw.sphere.legendre import ilegendre, legendre, dlegendre
from gpaw.sphere.csh import C as _C, Y as _Y, \
    dYdtheta as _dYdtheta, dYdphi as _dYdphi

"""
_Y = lambda l,m,theta,phi: _C(l,m)*legendre(l,abs(m),np.cos(theta))*np.exp(1j*m*phi)

# Define theta-derivative of spherical harmoncics
_dYdtheta = lambda l,m,theta,phi: _C(l,m)*dlegendre(l,abs(m),np.cos(theta))*np.exp(1j*m*phi)

# Define phi-derivative of spherical harmoncics
_dYdphi = lambda l,m,theta,phi: 1j*m*_C(l,m)*legendre(l,abs(m),np.cos(theta))*np.exp(1j*m*phi)

def Y(l,m,theta,phi):
    if m == 0:
        return _Y(l,m,theta,phi)
    elif m > 0:
        #return (-1)**m*np.real(_Y(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_Y(l,m,theta,phi)+_Y(l,m,theta,phi).conj())/2**0.5
        #return (-1)**m*(_Y(l,m,theta,phi)+(-1)**m*_Y(l,-m,theta,phi))/2**0.5
        #return (-1)**m*_C(l,m)*legendre(l,abs(m),np.cos(theta))*np.cos(m*phi)*2**0.5
        return C(l,m)*legendre(l,abs(m),np.cos(theta))*np.cos(m*phi)*2**0.5
    else:
        #return (-1)**m*np.imag(_Y(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_Y(l,abs(m),theta,phi)-_Y(l,abs(m),theta,phi).conj())/(2**0.5*1j)
        #return (-1)**m*(_Y(l,abs(m),theta,phi)-(-1)**abs(m)*_Y(l,-abs(m),theta,phi))/(2**0.5*1j)
        #return -_C(l,m)*legendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5
        return -C(l,m)*legendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5

# Define theta-derivative of spherical harmoncics
def dYdtheta(l,m,theta,phi):
    if m == 0:
        return _dYdtheta(l,m,theta,phi)
    elif m > 0:
        #return (-1)**m*np.real(_dYdtheta(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_dYdtheta(l,m,theta,phi)+_dYdtheta(l,m,theta,phi).conj())/2**0.5
        #return (-1)**m*(_dYdtheta(l,m,theta,phi)+(-1)**m*_dYdtheta(l,-m,theta,phi))/2**0.5
        #return (-1)**m*_C(l,m)*dlegendre(l,abs(m),np.cos(theta))*np.cos(m*phi)*2**0.5
        return C(l,m)*dlegendre(l,abs(m),np.cos(theta))*np.cos(m*phi)*2**0.5
    else:
        #return (-1)**m*np.imag(_dYdtheta(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_dYdtheta(l,abs(m),theta,phi)-_dYdtheta(l,abs(m),theta,phi).conj())/(2**0.5*1j)
        #return (-1)**m*(_dYdtheta(l,abs(m),theta,phi)-(-1)**abs(m)*_dYdtheta(l,-abs(m),theta,phi))/(2**0.5*1j)
        #return -_C(l,m)*dlegendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5
        return -C(l,m)*dlegendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5

# Define phi-derivative of spherical harmoncics
def dYdphi(l,m,theta,phi):
    if m == 0:
        return _dYdphi(l,m,theta,phi)
    elif m > 0:
        #return (-1)**m*np.real(_dYdphi(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_dYdphi(l,m,theta,phi)+_dYdphi(l,m,theta,phi).conj())/2**0.5
        #return (-1)**m*(_dYdphi(l,m,theta,phi)+(-1)**m*_dYdphi(l,-m,theta,phi))/2**0.5
        #return -(-1)**m*m*_C(l,m)*legendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5
        return -m*C(l,m)*legendre(l,abs(m),np.cos(theta))*np.sin(m*phi)*2**0.5
    else:
        #return (-1)**m*np.imag(_dYdphi(l,abs(m),theta,phi))*2**0.5
        #return (-1)**m*(_dYdphi(l,abs(m),theta,phi)-_dYdphi(l,abs(m),theta,phi).conj())/(2**0.5*1j)
        #return (-1)**m*(_dYdphi(l,abs(m),theta,phi)-(-1)**abs(m)*_dYdphi(l,-abs(m),theta,phi))/(2**0.5*1j)
        #return (-1)**m*m*_C(l,m)*legendre(l,abs(m),np.cos(theta))*np.cos(abs(m)*phi)*2**0.5
        return -m*C(l,m)*legendre(l,abs(m),np.cos(theta))*np.cos(m*phi)*2**0.5

"""

def condYdYdtheta(l1, m1, l2, m2, lquad=11, out=False):
    """Predict whether this combination requires a higher-order quadrature."""
    # Address minor issues when l1+l2 < lquad
    out |= (l1-l2)%2 == 1 and abs(m1-m2)%4 == 3 and m1*m2 < 0 #~77%, fpos~0%
    out |= (l1+l2)%2 == 1 and m1==0 and m2%2 == 1 #~74%, fpos~0.3%
    out |= (l1-l2)%2 == 1 and m2 != 0 and abs(m1-m2)%2 == 1 and abs(m1+m2)%4 == 3 #~80%, fpos~1.4%

    # Works well if lmax>=6 as lquad=11 and 11//2=5, otherwise irrelevant
    out |= l1+l2 >= lquad and (l1-l2)%2 == 1 and abs(m1-m2)%2 == 1 #~89%, fpos~1.6%
    out |= l1+l2 >= lquad and (l1-l2)%2 == 1 and (m1+0.5)*(m2+0.5) >= 0 and abs(m1-m2)%4 == 0 and abs(m1+m2)%2 == 0 # ~77%, fpos~0%
    out |= l1+l2 >= lquad and (l1-l2)%2 == 1 and (m1+0.5)*(m2+0.5) >= 0 and abs(m1-m2)%2 == 0 and abs(m1+m2)%4 == 0 # ~77%, fpos~0%
    return out

def condYdYdphi(l1, m1, l2, m2, lquad=11, out=False):
    """Predict whether this combination requires a higher-order quadrature."""
    # Address minor issues when l1+l2 < lquad
    out |= (l1-l2)%2 == 1 and abs(m1-m2)%4 == 3 and m1*m2 < 0 #~87%, fpos~0%
    out |= (l1+l2)%2 == 1 and m1==0 and m2%2 == 1 #~84%, fpos~0.4%
    out |= (l1-l2)%2 == 1 and m1*m2 > 0 and abs(m1+m2)%4 == 3 #~87%, fpos~0.1%

    # Works well if lmax>=6 as lquad=11 and 11//2=5, otherwise irrelevant
    out |= l1+l2 > lquad and (l1-l2)%2 == 1 and abs(m1-m2)%2 == 1 and m2 != 0 #~94%, fpos~1.1%
    return out

# -------------------------------------------------------------------

class UTSphereParallelSetup(TestCase):
    """Test case for various properties of the real spherical harmonics"""

    def test_convention_sign(self):
        # Test sign convention of the spherical harmonics (cf. GPAW's)
        from gpaw.spherical_harmonics import Y as Y0
        r_L = np.ones_like(theta_L)
        x_L = np.cos(phi_L)*np.sin(theta_L)
        y_L = np.sin(phi_L)*np.sin(theta_L)
        z_L = np.cos(theta_L)

        for l,m in lmiter(6, comm=world):
            s_L = Y(l,m,theta_L,phi_L)
            s0_L = Y0(lm_to_L(l,m), x_L, y_L, z_L)
            e = np.abs(s_L-s0_L).max()
            self.assertAlmostEqual(e, 0, 12, '%.17g max. (l=%2d, m=%2d)' % (e,l,m))

    #def test_convention_conj(self): #XXX TODO for real spherical harmonics
    #    # Test conjugation of the spherical harmonics (cf. QM1 p. 285)
    #    for l,m in lmiter(9, full=False, comm=world):
    #        s_L = (-1)**m*Y(l, -m, theta_L, phi_L)
    #        s0_L = Y(l, m, theta_L, phi_L).conj()
    #        e = np.abs(s_L - s0_L).max()
    #        self.assertAlmostEqual(e, 0, 12, '%.17g max. (l=%2d, m=%2d)' % (e,l,m))

    def test_addition_theorem(self):
        lmax = 9

        # Test that the real spherical harmonic addition theorem holds
        thetam_L = np.random.uniform(0, np.pi, size=theta_L.shape)
        world.broadcast(thetam_L, 0)
        phim_L = np.random.uniform(0, 2*np.pi, size=phi_L.shape)
        world.broadcast(phim_L, 0)
        cosv_L = np.cos(theta_L)*np.cos(thetam_L) \
            + np.sin(theta_L)*np.sin(thetam_L)*np.cos(phi_L-phim_L)
        P0_lL = np.array([legendre(l, 0, cosv_L) for l in range(lmax+1)])
        P_lL = np.zeros_like(P0_lL)
        for l,m in lmiter(lmax, comm=world):
            P_lL[l] += 4 * np.pi / (2*l + 1.) * Y(l, m, theta_L, phi_L) \
                * Y(l, m, thetam_L, phim_L) # real so no conjugation needed
        world.sum(P_lL)
        self.assertAlmostEqual(np.abs(P_lL-P0_lL).max(), 0, 6)

    def test_multipole_expansion(self):
        lmax = 9

        # Test multipole expansion in real spherical harmonics
        R = 1.0
        npts = 1000
        tol = 1e-9
        # ((R-dR)/(R+dR))**(lmax+1) = tol
        # (lmax+1)*np.log((R-dR)/(R+dR)) = np.log(tol)
        # (R-dR)/(R+dR) = np.exp(np.log(tol)/(lmax+1))
        # R-dR = (R+dR) * tol**(1/(lmax+1))
        # R * (1-tol**(1/(lmax+1))) = dR * (1+tol**(1/(lmax+1)))
        dR = R * (1-tol**(1./(lmax+1))) / (1+tol**(1./(lmax+1)))
        assert abs(((R-dR)/(R+dR))**(lmax+1) - tol) < 1e-12

        r_g = np.random.uniform(R+dR, 10*R, size=npts)
        world.broadcast(r_g, 0)
        theta_g = np.random.uniform(0, np.pi, size=npts)
        world.broadcast(theta_g, 0)
        phi_g = np.random.uniform(0, np.pi, size=npts)
        world.broadcast(phi_g, 0)

        r_vg = np.empty((3, npts), dtype=float)
        r_vg[0] = r_g*np.cos(phi_g)*np.sin(theta_g)
        r_vg[1] = r_g*np.sin(phi_g)*np.sin(theta_g)
        r_vg[2] = r_g*np.cos(theta_g)

        rm_g = np.random.uniform(0, R-dR, size=npts)
        world.broadcast(rm_g, 0)
        thetam_g = np.random.uniform(0, np.pi, size=npts)
        world.broadcast(thetam_g, 0)
        phim_g = np.random.uniform(0, np.pi, size=npts)
        world.broadcast(phim_g, 0)

        rm_vg = np.empty((3, npts), dtype=float)
        rm_vg[0] = rm_g*np.cos(phim_g)*np.sin(thetam_g)
        rm_vg[1] = rm_g*np.sin(phim_g)*np.sin(thetam_g)
        rm_vg[2] = rm_g*np.cos(thetam_g)

        f0_g = np.sum((r_vg-rm_vg)**2, axis=0)**(-0.5)
        f_g = np.zeros_like(f0_g)

        for l,m in lmiter(lmax, comm=world):
            f_g += 4 * np.pi / (2*l + 1.) * r_g**(-1) * (rm_g/r_g)**l \
                * Y(l, m, theta_g, phi_g) * Y(l, m, thetam_g, phim_g).conj()
        world.sum(f_g)

        e = np.abs(f_g-f0_g).max()
        self.assertAlmostEqual(e, 0, 9)

    def test_relation_complex(self):
        # Test relation for real spherical harmonics in terms of the complex
        for l,m in lmiter(9, comm=world):
            if m == 0:
                s_L = _Y(l,m,theta_L,phi_L)
            elif m > 0:
                s_L = (-1)**m * 2**0.5*np.real(_Y(l,m,theta_L,phi_L))
            else:
                s_L = (-1)**abs(m) * 2**0.5*np.imag(_Y(l,abs(m),theta_L,phi_L))
            s0_L = Y(l,m,theta_L,phi_L)
            e = np.abs(s_L - s0_L).max()
            self.assertAlmostEqual(e, 0, 12, '%.17g max. (l=%2d, m=%2d)' % (e,l,m))

    def test_integral_orthogonality(self):
        lmax = 5 #XXX use meshgrid above l=5

        # Test orthogonality of the spherical harmonics
        for l1,m1 in lmiter(lmax, comm=world):
            s1_L = Y(l1, m1, theta_L, phi_L)
            for l2,m2 in lmiter(lmax):
                s2_L = Y(l2, m2, theta_L, phi_L)

                # Note that weights times surface area make up for sin(theta)
                v = 4 * np.pi * np.vdot(s1_L, s2_L * weight_L)
                v0 = intYY(l1, m1, l2, m2)
                self.assertAlmostEqual(v, v0, 12, '%s != %s (l1=%2d, m1=%2d' \
                    ', l2=%2d, m2=%2d)' % (v,v0,l1,m1,l2,m2))

    def test_integral_moment_first(self):
        lmax = 5 #XXX use meshgrid above l=5

        # Test first-order moments of the spherical harmonics
        for l1,m1 in lmiter(lmax, comm=world):
            s1_L = Y(l1, m1, theta_L, phi_L)
            for l2,m2 in lmiter(lmax):
                s2_L = Y(l2, m2, theta_L, phi_L)

                # Note that weights times surface area make up for sin(theta)
                v_ex = 4 * np.pi * np.vdot(s1_L, np.cos(phi_L) \
                    * np.sin(theta_L) * s2_L * weight_L)
                v_ey = 4 * np.pi * np.vdot(s1_L, np.sin(phi_L) \
                    * np.sin(theta_L) * s2_L * weight_L)
                v_ez = 4 * np.pi * np.vdot(s1_L, np.cos(theta_L) \
                    * s2_L * weight_L)

                v0_ex = intYY_ex(l1, m1, l2, m2)
                v0_ey = intYY_ey(l1, m1, l2, m2)
                v0_ez = intYY_ez(l1, m1, l2, m2)

                self.assertAlmostEqual(v_ex, v0_ex, 12, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ex,v0_ex,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ey, v0_ey, 12, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ey,v0_ey,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ez, v0_ez, 12, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ez,v0_ez,l1,m1,l2,m2))

    def test_integral_derivative_theta(self):
        lmax = 7
        jmax = (lmax + 1)**2

        ntheta = 200
        nphi = 20
        dtheta = np.pi / ntheta
        dphi = 2 * np.pi / nphi
        theta_g, phi_g = np.meshgrid( \
            np.linspace(dtheta / 2, np.pi - dtheta / 2, ntheta), \
            np.linspace(dphi / 2, 2 * np.pi - dphi / 2, nphi))

        # Test theta-derivative of the spherical harmonics
        #if world.rank == 0:
        #    print '\n%s\nSpherical harmonic theta-derivatives\n%s' % ('-'*40,'-'*40)

        #hd = HitDetector(lmax, comm=world)
        #me = 0.0
        cf = lambda lmax,l1,m1: nL + ntheta * nphi \
            * sum(condYdYdtheta(l1, m1, l2, m2) for l2,m2 in lmiter(lmax))

        for l1,m1 in lmiter(lmax, comm=world, cost=cf):
            if cf(lmax, l1, m1) > nL:
                s1_g = Y(l1, m1, theta_g, phi_g)
            s1_L = Y(l1, m1, theta_L, phi_L)

            for l2,m2 in lmiter(lmax):
                v0_ex = intYdYdtheta_ex(l1, m1, l2, m2)
                v0_ey = intYdYdtheta_ey(l1, m1, l2, m2)
                v0_ez = intYdYdtheta_ez(l1, m1, l2, m2)

                # Note that weights times surface area make up for sin(theta)
                if condYdYdtheta(l1, m1, l2, m2):
                    ds2dtheta_g = dYdtheta(l2, m2, theta_g, phi_g)
                    v_ex = dphi * dtheta * np.vdot(s1_g, np.cos(phi_g) \
                        * np.cos(theta_g) * ds2dtheta_g * np.sin(theta_g))
                    v_ey = dphi * dtheta * np.vdot(s1_g, np.sin(phi_g) \
                        * np.cos(theta_g) * ds2dtheta_g * np.sin(theta_g))
                    v_ez = dphi * dtheta * np.vdot(s1_g, -np.sin(theta_g) \
                        * ds2dtheta_g * np.sin(theta_g))
                    del ds2dtheta_g
                else:
                    ds2dtheta_L = dYdtheta(l2, m2, theta_L, phi_L)
                    v_ex = 4 * np.pi * np.vdot(s1_L, np.cos(phi_L) \
                        * np.cos(theta_L) * ds2dtheta_L * weight_L)
                    v_ey = 4 * np.pi * np.vdot(s1_L, np.sin(phi_L) \
                        * np.cos(theta_L) * ds2dtheta_L * weight_L)
                    v_ez = 4 * np.pi * np.vdot(s1_L, -np.sin(theta_L) \
                        * ds2dtheta_L * weight_L)
                    del ds2dtheta_L

                ## Predict: Will this combination require an expensive re-calculation?
                #indicator = condYdYdtheta(l1, m1, l2, m2)
                #hd.update(indicator, e, tol, lm_to_L(l1,m1), lm_to_L(l2,m2))

                #if abs(l1-l2) % 2 == 1 and abs(m1-m2) <= 1:
                #    print 'l1=%2d, m1=%2d, l2=%2d, m2=%2d, v_e=[(%8.5f,%8.5fj),(%8.5f,%8.5fj),(%8.5f,%8.5fj)], err=[%12.9f,%12.9f,%12.9f]' % (l1,m1,l2,m2,v_ex.real,v_ex.imag,v_ey.real,v_ey.imag,v_ez.real,v_ez.imag,abs(v_ex-v0_ex),abs(v_ey-v0_ey),abs(v_ez-v0_ez))

                self.assertAlmostEqual(v_ex, v0_ex, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ex,v0_ex,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ey, v0_ey, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ey,v0_ey,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ez, v0_ez, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ez,v0_ez,l1,m1,l2,m2))
                #me = max([me, abs(v_ex-v0_ex), abs(v_ey-v0_ey), abs(v_ez-v0_ez)])

        #catch(debug)
        #hd.finalize()
        #if debug and world.rank == 0:
        #    hd.plot(fig=1)
        #sys.stdout.flush()
        #world.barrier()
        #print 'MAX ERROR:', me

        #del lmax, jmax, tol, ntheta, nphi, dtheta, dphi, theta_g, phi_g, hd, cf, s1_L, \
        #    ds2dtheta_L, v_ex, v0_ex, v_ey, v0_ey, v_ez, v0_ez

    def test_integral_derivative_phi(self):
        lmax = 7
        jmax = (lmax + 1)**2

        eps = 1e-12 # to avoid division-by-zero errors
        ntheta = 200
        nphi = 20
        dtheta = np.pi/ntheta
        dphi = 2*np.pi/nphi
        theta_g, phi_g = np.meshgrid(np.linspace(dtheta/2,np.pi-dtheta/2,ntheta), \
                                     np.linspace(dphi/2,2*np.pi-dphi/2,nphi))

        ## Test phi-derivative of the spherical harmonics
        #if world.rank == 0:
        #    print '\n%s\nSpherical harmonic phi-derivatives\n%s' % ('-'*40,'-'*40)

        #hd = HitDetector(lmax, comm=world)
        #me = 0.0
        cf = lambda lmax,l1,m1: nL + ntheta * nphi \
            * sum(condYdYdphi(l1, m1, l2, m2) for l2,m2 in lmiter(lmax))

        for l1,m1 in lmiter(lmax, comm=world, cost=cf):
            if cf(lmax, l1, m1) > nL:
                s1_g = Y(l1, m1, theta_g, phi_g)
            s1_L = Y(l1, m1, theta_L, phi_L)

            for l2,m2 in lmiter(lmax):
                v0_ex = intYdYdphi_ex(l1,m1,l2,m2)
                v0_ey = intYdYdphi_ey(l1,m1,l2,m2)
                v0_ez = intYdYdphi_ez(l1,m1,l2,m2)

                # Note that weights times surface area make up for sin(theta)
                if condYdYdphi(l1, m1, l2, m2):
                    ds2dphi_g = dYdphi(l2,m2,theta_g,phi_g)
                    v_ex = dphi * dtheta * np.vdot(s1_g, \
                        -np.sin(phi_g) * ds2dphi_g)
                    v_ey = dphi * dtheta * np.vdot(s1_g, \
                        np.cos(phi_g) * ds2dphi_g)
                    v_ez = dphi * dtheta * np.vdot(s1_g, 0 * ds2dphi_g) #XXX
                    del ds2dphi_g
                else:
                    ds2dphi_L = dYdphi(l2, m2, theta_L, phi_L)
                    v_ex = 4 * np.pi * np.vdot(s1_L, -np.sin(phi_L) \
                        / (np.sin(theta_L) + eps) * ds2dphi_L * weight_L)
                    v_ey = 4 * np.pi * np.vdot(s1_L, np.cos(phi_L) \
                        / (np.sin(theta_L) + eps) * ds2dphi_L * weight_L)
                    v_ez = 4 * np.pi * np.vdot(s1_L, 0 * ds2dphi_L * weight_L) #XXX
                    del ds2dphi_L

                ## Predict: Will this combination require an expensive re-calculation?
                #indicator = condYdYdphi(l1, m1, l2, m2)
                #hd.update(indicator, e, tol, lm_to_L(l1,m1), lm_to_L(l2,m2))

                #if abs(l1-l2)%2 == 1 and abs(m1-m2) <= 1:
                #    print 'l1=%2d, m1=%2d, l2=%2d, m2=%2d, v_e=[(%8.5f,%8.5fj),(%8.5f,%8.5fj),(%8.5f,%8.5fj)], err=[%12.9f,%12.9f,%12.9f]' % (l1,m1,l2,m2,v_ex.real,v_ex.imag,v_ey.real,v_ey.imag,v_ez.real,v_ez.imag,abs(v_ex-v0_ex),abs(v_ey-v0_ey),abs(v_ez-v0_ez))

                self.assertAlmostEqual(v_ex, v0_ex, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ex,v0_ex,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ey, v0_ey, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ey,v0_ey,l1,m1,l2,m2))
                self.assertAlmostEqual(v_ez, v0_ez, 3, '%s != %s (l1=%2d, ' \
                    'm1=%2d, l2=%2d, m2=%2d)' % (v_ez,v0_ez,l1,m1,l2,m2))
                #me = max([me, abs(v_ex-v0_ex), abs(v_ey-v0_ey), abs(v_ez-v0_ez)])

        #catch(debug)
        #hd.finalize()
        #if debug and world.rank == 0:
        #    hd.plot(fig=2)
        #    pl.show()
        #sys.stdout.flush()
        #world.barrier()
        #print 'MAX ERROR:', me

        #del lmax, jmax, tol, ntheta, nphi, dtheta, dphi, theta_g, phi_g, hd, cf, s1_L, \
        #    ds2dphi_L, v_ex, v0_ex, v_ey, v0_ey, v_ez, v0_ez


# -------------------------------------------------------------------

"""
class HitDetector:
    def __init__(self, lmax, comm=world, verbose=False):
        self.lmax = lmax
        self.jmax = (self.lmax+1.0)**2
        self.comm = comm
        self.verbose = verbose
        self.mynfails = 0
        self.myhits, self.myfneg, self.myfpos = [], [], []
        self.nfails, self.hits, self.fneg, self.fpos = None, None, None, None

    def update(self, indicator, error, tolerance, *args):
        if error > tolerance:
            self.mynfails += 1
        if indicator == (error>tolerance):
            self.myhits.append(args)
        elif indicator:
            # false positive if indicator says 'yes'
            self.myfpos.append(args)
        else:
            # false negative if indicator says 'no'
            self.myfneg.append(args)
        if self.verbose:
            (l1,m1),(l2,m2) = map(L_to_lm, args)
            print 'l1=%2d, m1=%2d, l2=%2d, m2=%2d, l1+l2=%2d, |l1-l2|=%2d, ' \
                '|m1-m2|=%2d, I=%d, I0=%d, dI=%2d' % (l1,m1,l2,m2,l1+l2, \
                abs(l1-l2),abs(m1-m2), indicator, error>tolerance, \
                int(indicator)-int(error>tolerance))

    def finalize(self):
        self.nfails = self.comm.sum(self.mynfails)
        mycounts_c = np.array([len(self.myhits), len(self.myfneg),
                               len(self.myfpos)], dtype=int)
        reqs = []
        if self.comm.rank == 0:
            counts_rc = np.empty((self.comm.size,3), dtype=int)
            self.comm.gather(mycounts_c, 0, counts_rc)
            np.cumsum(counts_rc, axis=0, out=counts_rc)
            self.nhits, self.nfneg, self.nfpos = counts_rc[-1]

            if self.verbose:
                N = self.jmax**2
                print 'INDICATOR: total=%6d, fails=%8.4f %%, hits=%8.4f %%, ' \
                    'fneg=%8.4f %%, fpos=%8.4f %%' % (N, 1e2*self.nfails/N, \
                    1e2*self.nhits/N, 1e2*self.nfneg/self.nfails, \
                    1e2*self.nfpos/(N-self.nfails))

            # Allocate buffers for collected data
            self.hits = np.empty((self.nhits,2), dtype=int)
            self.fneg = np.empty((self.nfneg,2), dtype=int)
            self.fpos = np.empty((self.nfpos,2), dtype=int)
            bufs_c = [self.hits, self.fneg, self.fpos]

            # Start receiving data from slaves
            for rank in range(1,self.comm.size):
                for c,counts_r in enumerate(counts_rc.transpose()):
                    buf = bufs_c[c][counts_r[rank-1]:counts_r[rank]]
                    if self.verbose:
                        print 'receive: rank=%d, c=%d, buf=%s contiguous:%s, ' \
                            '%d:%d' % (rank,c,buf.shape,buf.flags.contiguous, \
                            counts_r[rank-1],counts_r[rank])
                    reqs.append(self.comm.receive(buf, rank, tag=c, block=False))
            del counts_rc

            # Fill in own data
            self.hits[0:mycounts_c[0]] = np.array(self.myhits)
            self.fneg[0:mycounts_c[1]] = np.array(self.myfneg)
            self.fpos[0:mycounts_c[2]] = np.array(self.myfpos)
        else:
            self.comm.gather(mycounts_c, 0)

            # XXX does this need to be non-blocking as well?
            bufs_c = [self.myhits, self.myfneg, self.myfpos]
            for c,buf in enumerate(bufs_c):
                if self.verbose:
                    print 'send   : rank=%d, c=%d, buf=%s contiguous:%s, ' \
                        '%d:%d' % (self.comm.rank,c,np.array(buf).shape, \
                        np.array(buf).flags.contiguous,0,mycounts_c[c])
                reqs.append(self.comm.send(np.array(buf), 0, tag=c, block=False))

        # Wait for all non-blocking MPI requests to complete
        if reqs:
            if self.verbose: print 'rank=%d, waiting...' % self.comm.rank
            self.comm.waitall(reqs)

    def plot(self, fig=1):
        assert self.comm.rank == 0, 'Must be run on master only.'
        fig = pl.figure(fig,(12,12))
        ax = pl.axes()

        # Draw seperation lines for different l1,l2 blocks
        for l1 in range(self.lmax+1):
            ax.plot([-0.5, self.jmax-0.5], [lm_to_L(l1,-l1)-0.5,lm_to_L(l1,-l1)-0.5], '--k')
            ax.plot([lm_to_L(l1,-l1)-0.5,lm_to_L(l1,-l1)-0.5], [-0.5, self.jmax-0.5], '--k')
            for l2 in range(self.lmax+1):
                m = min(l1,l2)
                ax.plot([lm_to_L(l1,-m)-0.5,lm_to_L(l1,m)+0.5], [lm_to_L(l2,-m)-0.5,lm_to_L(l2,m)+0.5], ':k')

        if self.nhits: ax.plot(self.hits[:,0], self.hits[:,1], ',r')
        if self.nfneg: ax.plot(self.fneg[:,0], self.fneg[:,1], 'dg')
        if self.nfpos: ax.plot(self.fpos[:,0], self.fpos[:,1], 'sb')

        ax.set_xlim([-0.5, self.jmax+0.5])
        ax.set_ylim([-0.5, self.jmax+0.5])
        return fig, ax
"""

# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_rsh.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []

##     for test in [UTSphereParallelSetup]:
##         info = ['', test.__name__, test.__doc__.strip('\n'), '']
##         testsuite = initialTestLoader.loadTestsFromTestCase(test)
##         map(testrunner.stream.writeln, info)
##         testresult = testrunner.run(testsuite)
##         assert testresult.wasSuccessful(), 'Initial verification failed!'
##         parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
##                         tci in testsuite._tests if hasattr(tci, '_parinfo')])
##     parinfo = np.unique(np.sort(parinfo)).tolist()


    testcases = [UTSphereParallelSetup]
    #for dtype in [float, complex]:
    #    for parstride_bands in [False, True]:
    #        for blocking in ['fast', 'best']: # 'light'
    #            for async in [False, True]:
    #                testcases.append(UTConstantWavefunctionFactory(dtype, \
    #                    parstride_bands, blocking, async))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        #if __name__ == '__builtin__' and not testresult.wasSuccessful(): #XXX ENABLE!!!
        if not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_rsh.log for details.')
