from math import pi

import Numeric as num
from ASE.Units import units, Convert

import gpaw.mpi as mpi
from gpaw.spherical_harmonics import Y
from gpaw.utilities.vector import Vector3d
from gpaw.utilities.timing import StepTimer

class ExpandYl:
    """Expand the smooth wave functions in spherical harmonics
    relative to a given center.

    center:
      the center for the expansion (units.GetLengthUnit())
    gd:
      grid_descriptor of the grids to expand
    lmax:
      maximal angular momentum in the expansion (lmax<7)
    Rmax:
      maximal radius of the expansion (units.GetLengthUnit())
    dR:
      grid spacing in the radius (units.GetLengthUnit())
    """
    def __init__(self,center,gd,lmax=6,Rmax=None,dR=None):

        a0 = Convert(1, 'Bohr', units.GetLengthUnit())
        center = Vector3d(center) / a0

        self.a0 = a0
        self.center = center
        self.lmax=lmax
        self.gd = gd

        self.L_l= []
        for l in range(lmax+1):
            for m in range(2*l+1):
                self.L_l.append(l)
        nL = len(self.L_l)

        # set Rmax to the maximal radius possible
        # i.e. the corner distance
        if not Rmax:
            Rmax = 0
            extreme = gd.h_c*gd.N_c
            for corner in ([0,0,0],[1,0,0],[0,1,0],[1,1,0],
                           [0,0,1],[1,0,1],[0,1,1],[1,1,1]):
                Rmax = max(Rmax,
                           self.center.distance(num.array(corner) * extreme) )
        else:
            Rmax /= a0
        self.Rmax = Rmax

        if not dR:
            dR = min(gd.h_c)
        else:
            dR /= a0 
        self.dR = dR
        
        # initialize the ylm and Radial grids

        # self.R_V will contain the volume of the R shell
        # self.R_g will contain the radial indicees corresponding to
        #     each grid point
        # self.y_Lg will contain the YL values corresponding to
        #     each grid point
        R_V = num.zeros((int(Rmax/dR+1),),num.Float)
        y_Lg = gd.zeros((nL,),num.Float)
        R_g = num.zeros(y_Lg[0].shape,num.Int)-1
        for i in range(gd.beg_c[0],gd.end_c[0]):
            ii = i - gd.beg_c[0]
            for j in range(gd.beg_c[1],gd.end_c[1]):
                jj = j - gd.beg_c[1]
                for k in range(gd.beg_c[2],gd.end_c[2]):
                    kk = k - gd.beg_c[2]
                    vr = center - Vector3d([i*gd.h_c[0],
                                            j*gd.h_c[1],
                                            k*gd.h_c[2]])
                    r = vr.length()
                    if r>0 and r<Rmax:
                        rhat = vr/r
                        for L in range(nL):
                            y_Lg[L,ii,jj,kk] = Y(L, rhat[0], rhat[1], rhat[2])
                        iR = int(r/dR)
                        R_g[ii,jj,kk] = iR
                        R_V[iR] += 1
        gd.comm.sum(R_V)

        self.R_g = R_g
        self.R_V = R_V * gd.dv
        self.y_Lg = y_Lg

    def expand(self,psit_g):
        """Expand a wave function"""
      
        gamma_l = num.zeros((self.lmax+1),num.Float)
        nL = len(self.L_l)
        L_l = self.L_l
        dR = self.dR
        
        for i,dV in enumerate(self.R_V):
            # get the R shell and it's Volume
            R_g = num.where(self.R_g == i, 1, 0)
            if dV > 0:
                for L in range(nL):
                    psit_LR = self.gd.integrate(psit_g * R_g * self.y_Lg[L])
                    gamma_l[L_l[L]] += 4*pi / dV * psit_LR**2
                
        return gamma_l

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
            srange = range(calculator.nspins)
        else:
            srange = spins
        if not kpoints:
            krange = range(calculator.nkpts)
        else:
            krange = kpoints
        if not bands:
            nrange = range(calculator.nbands)
        else:
            nrange = bands

        print >> f, '# Yl expansion','of smooth wave functions'
        lu = units.GetLengthUnit()
        print >> f, '# center =',self.center * self.a0, lu
        print >> f, '# Rmax =', self.Rmax * self.a0, lu
        print >> f, '# dR =', self.dR * self.a0, lu
        print >> f, '# lmax =', self.lmax 
        print >> f, '# s    k     n',
        print >> f, '    norm      sum',
        spdfghi = 's p d h f g h i'.split()
        for l in range(self.lmax+1):
            print >> f, '      %'+spdfghi[l],
        print >> f

        for s in srange:
            for k in krange:
                u = k*calculator.nspins + s
                for n in nrange:
                    psit_G = calculator.kpt_u[u].psit_nG[n]
                    norm = self.gd.integrate(psit_G**2)

                    gl = self.expand(psit_G)
                    gsum = num.sum(gl)
                    gl = 100 * gl / gsum

                    print >> f, '%2d %5d %5d' % (s,k,n),
                    print >> f, "%8.4f %8.4f" % (norm,gsum),
                
                    for g in gl:
                        print >> f, "%8.2f" %g,
                    print >> f
                    f.flush()
        f.close()
