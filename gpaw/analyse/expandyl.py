from math import pi

import numpy as np
from ase.units import Bohr, Hartree

import gpaw.mpi as mpi
from gpaw.spherical_harmonics import Y
from gpaw.utilities.vector import Vector3d
from gpaw.utilities.timing import StepTimer
from gpaw.utilities.tools import coordinates

class AngularIntegral:
    """Perform an angular integral on the grid.

    center:
      the center (Ang)
    gd:
      grid_descriptor of the grids to expand
    Rmax:
      maximal radius of the expansion (Ang)
    dR:
      grid spacing in the radius (Ang)
    """
    def __init__(self, center, gd, Rmax=None, dR=None):
        assert gd.orthogonal
        center = Vector3d(center) / Bohr

        self.center = center
        self.gd = gd

        # set Rmax to the maximal radius possible
        # i.e. the corner distance
        if not Rmax:
            Rmax = 0
            extreme = gd.h_cv.diagonal() * gd.N_c
            for corner in ([0,0,0],[1,0,0],[0,1,0],[1,1,0],
                           [0,0,1],[1,0,1],[0,1,1],[1,1,1]):
                Rmax = max(Rmax,
                           self.center.distance(np.array(corner) * extreme))
        else:
            Rmax /= Bohr
        self.Rmax = Rmax

        if not dR:
            dR = min(gd.h_cv.diagonal())
        else:
            dR /= Bohr
        self.dR = dR

        self.initialize()

    def initialize(self):
        """Initialize grids."""
        
        Rmax = self.Rmax
        dR = self.dR
        gd = self.gd

        # initialize the ylm and Radial grids

        # self.V_R will contain the volume of the R shell
        # self.R_R will contain the mean radius of the R shell
        # self.R_g will contain the radial indicees corresponding to
        #     each grid point
        # self.ball_g will contain the mask of the ball of radius Rmax
        nR = int(Rmax / dR + 1)
        V_R = np.zeros((nR,))
        R_R = np.zeros((nR,))
        R_g = gd.zeros(dtype=int) - 1
        ball_g = gd.zeros(dtype=int)
        for i in range(gd.beg_c[0], gd.end_c[0]):
            ii = i - gd.beg_c[0]
            for j in range(gd.beg_c[1], gd.end_c[1]):
                jj = j - gd.beg_c[1]
                for k in range(gd.beg_c[2], gd.end_c[2]):
                    kk = k - gd.beg_c[2]
                    vr = self.center - Vector3d([i * gd.h_cv[0, 0],
                                                 j * gd.h_cv[1, 1],
                                                 k * gd.h_cv[2, 2]])
                    r = vr.length()
                    if r>0 and r<Rmax:
                        rhat = vr / r
                        iR = int(r / dR)
                        R_g[ii,jj,kk] = iR
                        ball_g[ii,jj,kk] = 1
                        V_R[iR] += 1
                        R_R[iR] += r            
        gd.comm.sum(V_R)
        gd.comm.sum(R_R)

        self.R_g = R_g
        self.ball_g = ball_g
        self.V_R = V_R * gd.dv
        self.nominalR_R = self.dR * (np.arange(len(self.V_R)) + .5)
        self.R_R = np.where(V_R > 0, R_R / V_R, self.nominalR_R)

    def integrate(self, f_g):
        """Integrate a function on the grid over the angles."""
        int_R = []
        for i, dV in enumerate(self.V_R):
            # get the R shell
            R_g = np.where(self.R_g == i, 1, 0)
            int_R.append(self.gd.integrate(f_g * R_g) / self.dR)
        return np.array(int_R)

    def average(self, f_g):
        """Give the angular average of a function on the grid."""
        return self.integrate(f_g) / self.radii()**2 / (4 * pi)

    def radii(self, model='nominal'):
        """Return the radii of the radial shells"""
        if model == 'nominal':
            return self.nominalR_R
        elif model == 'mean':
            return self.R_R
        else:
            raise NonImplementedError

class ExpandYl(AngularIntegral):
    """Expand the smooth wave functions in spherical harmonics
    relative to a given center.

    center:
      the center for the expansion (Ang)
    gd:
      grid_descriptor of the grids to expand
    lmax:
      maximal angular momentum in the expansion (lmax<7)
    Rmax:
      maximal radius of the expansion (Ang)
    dR:
      grid spacing in the radius (Ang)
    """
    def __init__(self, center, gd, lmax=6, Rmax=None, dR=None):

        self.lmax = lmax
        self.L_l = []
        for l in range(lmax+1):
            for m in range(2*l+1):
                self.L_l.append(l)

        AngularIntegral.__init__(self, center, gd, Rmax, dR)

    def initialize(self):
        """Initialize grids."""
        
        center = self.center
        Rmax = self.Rmax
        dR = self.dR
        gd = self.gd
        nL = len(self.L_l)

        # initialize the ylm and Radial grids

        # self.V_R will contain the volume of the R shell
        # self.R_g will contain the radial indicees corresponding to
        #     each grid point
        # self.ball_g will contain the mask of the ball of radius Rmax
        # self.y_Lg will contain the YL values corresponding to
        #     each grid point
        V_R = np.zeros((int(Rmax / dR + 1),))
        npY = np.vectorize(Y, (float,), 'spherical harmonic')

        r_cg, r2_g = coordinates(gd, self.center, tiny=1.e-78)
        r_g = np.sqrt(r2_g)
        rhat_cg = r_cg / r_g

        ball_g = np.where(r_g < Rmax, 1, 0)
        R_g = np.where(r_g < Rmax, r_g / dR, -1).astype(int)
        y_Lg = []
        for L in range(nL):
            y_Lg.append(npY(L, rhat_cg[0], rhat_cg[1], rhat_cg[2]))

        for i, v in enumerate(V_R):
            V_R[i] = np.where(R_g == i, 1, 0).sum()
        gd.comm.sum(V_R)

        self.R_g = R_g
        self.ball_g = ball_g
        self.V_R = V_R * gd.dv
        self.y_Lg = y_Lg


    def expand(self,psit_g):
        """Expand a wave function"""
      
        gamma_l = np.zeros((self.lmax+1))
        nL = len(self.L_l)
        L_l = self.L_l
        dR = self.dR
        
        for i,dV in enumerate(self.V_R):
            # get the R shell and it's Volume
            R_g = np.where(self.R_g == i, 1, 0)
            if dV > 0:
                for L in range(nL):
                    psit_LR = self.gd.integrate(psit_g * R_g * self.y_Lg[L])
                    gamma_l[L_l[L]] += 4*pi / dV * psit_LR**2
        # weight of the wave function inside the ball
        weight =  self.gd.integrate(psit_g**2 * self.ball_g)
        
        return gamma_l, weight

    def to_file(self,calculator,
                filename='expandyl.dat',
                spins=None,
                kpoints=None,
                bands=None
                ):
        """Expand a range of wave functions and write the result
        to a file"""
        if mpi.rank == 0:
            f = open(filename, 'w')
        else:
            f = open('/dev/null', 'w')

        if not spins:
            srange = range(calculator.wfs.nspins)
        else:
            srange = spins
        if not kpoints:
            krange = range(len(calculator.wfs.ibzk_kc))
        else:
            krange = kpoints
        if not bands:
            nrange = range(calculator.wfs.nbands)
        else:
            nrange = bands

        print >> f, '# Yl expansion','of smooth wave functions'
        lu = 'Angstrom'
        print >> f, '# center =', self.center * Bohr, lu
        print >> f, '# Rmax =', self.Rmax * Bohr, lu
        print >> f, '# dR =', self.dR * Bohr, lu
        print >> f, '# lmax =', self.lmax 
        print >> f, '# s    k     n',
        print >> f, '     e[eV]      occ',
        print >> f, '    norm      sum   weight',
        spdfghi = 's p d f g h i'.split()
        for l in range(self.lmax+1):
            print >> f, '      %'+spdfghi[l],
        print >> f

        for s in srange:
            for k in krange:
                u = k*calculator.wfs.nspins + s
                for n in nrange:
                    kpt = calculator.wfs.kpt_u[u]
                    psit_G = kpt.psit_nG[n]
                    norm = self.gd.integrate(psit_G**2)

                    gl, weight = self.expand(psit_G)
                    gsum = np.sum(gl)
                    gl = 100 * gl / gsum

                    print >> f, '%2d %5d %5d' % (s, k, n),
                    print >> f, '%10.4f %8.4f' % (kpt.eps_n[n] * Hartree,
                                                  kpt.f_n[n]),
                    print >> f, "%8.4f %8.4f %8.4f" % (norm, gsum, weight),
                
                    for g in gl:
                        print >> f, "%8.2f" %g,
                    print >> f
                    f.flush()
        f.close()
