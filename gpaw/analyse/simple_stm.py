import sys
from math import sqrt

import numpy as npy

from ase.units import Bohr
from ase.dft.stm import STM
from ase.io.cube import write_cube
from ase.io.plt import write_plt

from gpaw.mpi import MASTER

class SimpleStm(STM):
    """Simple STM object to simulate STM pictures.

    The simulation uses either a single pseudo-wavefunction (PWF)
    or the PWFs inside the given bias range (XXX TODO)."""
    def __init__(self, atoms):
        STM.__init__(self, atoms)

        self.calc.initialize_wave_functions()

        self.gd = self.calc.gd
        self.offset_c = [int(not a) for a in self.gd.domain.pbc_c]
        

    def calculate_ldos(self, bias):
        """bias is the n, k, s list/tuple."""
        self.bias = bias
        self.wf = True

        n, k, s = bias
        psi = self.calc.get_pseudo_wave_function(n, k, s)
        self.ldos = (psi * npy.conj(psi)).real

    def write_3D(self, bias, file, filetype=None):
        """Write the density as a 3D file.

        Units: [e/A^3]"""
        self.calculate_ldos(bias)

        if filetype is None:
            # estimate file type from name ending
            filetype = file.split('.')[-1]
        filetype.lower()

        if filetype == 'cube':
            write_cube(file, self.calc.get_atoms(), self.ldos)
        elif filetype == 'plt':
            write_plt(file, self.calc.get_atoms(), self.ldos)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')

    def current_to_density(self, current):
        """The connection between density n and current I

        n [e/Angstrom^3] = 0.0002 sqrt(I [nA])

        as given in Hofer et al., RevModPhys 75 (2003) 1287
        """
        return 0.0002 * sqrt(current)

    def density_to_current(self, density):
        return 5000. * density**2

    def scan_const_current(self, current, bias):
        """Get the height image for constant current I [nA].
        """
        return self.scan_const_density(self.current_to_density(current),
                                       bias)

    def scan_const_density(self, density, bias, interpolate=False):
        """Get the height image for constant density [e/Angstrom^3].
        """
 
        self.calculate_ldos(bias)

        self.density = density

        gd = self.calc.gd
        h_c = gd.h_c
        pbc_c = gd.domain.pbc_c
        nx, ny = (gd.N_c - self.offset_c)[:2]

        # each cpu will have the full array, but works on its
        # own part only
        heights = npy.zeros((nx, ny)) - 1
        for i in range(gd.beg_c[0], gd.end_c[0]):
            ii = i - gd.beg_c[0]
            for j in range(gd.beg_c[1], gd.end_c[1]):
                jj = j - gd.beg_c[1]
                
                zline = self.ldos[ii, jj]
                
                # check from above until you find the required density 
                for k in range(gd.end_c[2]-1, gd.beg_c[2]-1, -1):
                    kk = k - gd.beg_c[2]
                    if zline[kk] > density:
                        heights[i - self.offset_c[0], 
                                j - self.offset_c[1]] = k
                        break

        # collect the results
        gd.comm.max(heights)

        if interpolate:
            # collect the full grid to enable interpolation
            fullgrid = gd.collect(self.ldos)

            kmax = self.ldos.shape[2] - 1
            for i in range(gd.beg_c[0], gd.end_c[0]):
                ii = i - gd.beg_c[0]
                i -= self.offset_c[0]
                for j in range(gd.beg_c[1], gd.end_c[1]):
                    jj = j - gd.beg_c[1]
                    j -= self.offset_c[1]
                    if heights[i, j] > 0:
                        if heights[i, j] < kmax:
                            c1 = fullgrid[i, j, int(heights[i, j])]
                            c2 = fullgrid[i, j, int(heights[i, j])+1]
                            k = heights[i, j] + (density - c1) / (c2 -  c1)
                        else:
                            k = kmax

        self.heights = npy.where(heights > 0,
                                 (heights + self.offset_c[2]) * h_c[2], -1)

        return heights
        
    def write(self, file=None):
        """Write STM data to a file in gnuplot readable tyle."""

        gd = self.calc.gd
        bias = self.bias

        if gd.rank != MASTER:
            return
        
        heights = self.heights

        # the lowest point is not stored for non-periodic BCs
        nx, ny = heights.shape[:2]
        h_c = gd.h_c * Bohr
        xvals = [(i + self.offset_c[0]) * h_c[0] for i in range(nx)]
        yvals = [(i + self.offset_c[1]) * h_c[1] for i in range(ny)]

        if file is None:
            n, k, s = bias
            fname = 'stm_n%dk%ds%d.dat' % (n, k, s)
        else:
            fname = file
        f = open(fname, 'w')

        try:
            import datetime
            print >> f, '#', datetime.datetime.now().ctime()
        except:
            pass
        print >> f, '# Simulated STM picture'
        print >> f, '# density=', self.density,'[e/Angstrom^3]',
        print >> f, '(current=', self.density_to_current(self.density), '[nA])'
        if self.wf:
            print >> f, '# pseudo-wf n=%d k=%d s=%d' % tuple(self.bias)
        else:
            print >> f, '# bias=', self.bias, '[eV]'
        print >> f, '# x[Angs.]   y[Angs.]     h[Angs.] (-1 is not found)'
        for i in range(nx):
            for j in range(ny):
                if heights[i, j] == -1:
                    height = -1
                else:
                    height = heights[i, j] * Bohr
                print >> f, '%10g %10g %12g' % (yvals[j], xvals[i], height)
            print >> f
        f.close()

