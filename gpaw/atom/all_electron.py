# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Atomic Density Functional Theory
"""

from math import pi, sqrt, log
import pickle
import sys

import Numeric as num
import LinearAlgebra as linalg
from ASE.ChemicalElements.name import names

from gpaw.atom.configurations import configurations
from gpaw.grid_descriptor import RadialGridDescriptor
from gpaw.xc_functional import XCOperator, XCFunctional
from gpaw.utilities import hartree

# fine-structure constant
alpha = 1 / 137.036

class AllElectron:
    """Object for doing an atomic DFT calculation."""

    def __init__(self, symbol, xcname='LDA', scalarrel=False):
        """Do an atomic DFT calculation.
        
        Example:

        a = AllElectron('Fe')
        a.Run()
        """

        self.symbol = symbol
        self.xcname = xcname
        self.scalarrel = scalarrel

        # Get reference state:
        self.Z, nlfe_j = configurations[symbol]

        # Collect principal quantum numbers, angular momentum quantum
        # numbers, occupation numbers and eigenvalues (j is a combined
        # index for n and l):        
        self.n_j = [n for n, l, f, e in nlfe_j]
        self.l_j = [l for n, l, f, e in nlfe_j]
        self.f_j = [f for n, l, f, e in nlfe_j]
        self.e_j = [e for n, l, f, e in nlfe_j]

        print
        if scalarrel:
            print 'Scalar-relativistic atomic',
        else:
            print 'Atomic',
        print '%s calculation for %s (%s, Z=%d)' % (
            xcname, symbol, names[self.Z], self.Z)

    def intialize_wave_functions(self):
        r = self.r
        dr = self.dr
        # Initialize with Slater function:
        for l, e, u in zip(self.l_j, self.e_j, self.u_j):
            a = sqrt(-2.0 * e)

            # This one: "u[:] = r**(1 + l) * num.exp(-a * r)" gives
            # OverflowError: math range error XXX
            u[:] = r**(1 + l)
            rmax = 350.0 / a     # numpy!
            gmax = int(rmax * self.N / (self.beta + rmax))
            u[:gmax] *= num.exp(-a * r[:gmax])
            u[gmax:] = 0.0

            norm = num.dot(u**2, dr)
            u *= 1.0 / sqrt(norm)
        
    def run(self):
        #     beta g
        # r = ------, g = 0, 1, ..., N - 1
        #     N - g
        #
        #        rN
        # g = --------
        #     beta + r

        maxnodes = max([n - l - 1 for n, l in zip(self.n_j, self.l_j)])
        N = (maxnodes + 1) * 150
        print N, 'radial gridpoints.'
        beta = 0.4
        g = num.arange(N, typecode=num.Float)
        self.r = beta * g / (N - g)
        self.dr = beta * N / (N - g)**2
        self.rgd = RadialGridDescriptor(self.r, self.dr)
        self.d2gdr2 = -2 * N * beta / (beta + self.r)**3
        self.N = N
        self.beta = beta

        # Number of orbitals:
        nj = len(self.n_j)
        
        # Radial wave functions multiplied by radius:
        self.u_j = num.zeros((nj, self.N), num.Float)

        # Effective potential multiplied by radius:
        self.vr = num.zeros(N, num.Float)

        # Electron density:
        self.n = num.zeros(N, num.Float)

        self.xc = XCOperator(XCFunctional(self.xcname, self.scalarrel),
                             self.rgd)

        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j

        Z = self.Z    # nuclear charge
        r = self.r    # radial coordinate
        dr = self.dr  # dr/dg
        n = self.n    # electron density
        vr = self.vr  # effective potential multiplied by r

        vHr = num.zeros(self.N, num.Float)
        vXC = num.zeros(self.N, num.Float)

        try:
            f = open(self.symbol + '.restart', 'r')
        except IOError:
            self.intialize_wave_functions()
            n[:] = self.calculate_density()
        else:
            print 'Using old density for initial guess.'
            n[:] = pickle.load(f)
            n *= Z / (num.dot(n * r**2, dr) * 4 * pi)

        bar = '|------------------------------------------------|'
        print bar
        niter = 0
        qOK = log(1e-10)
        while True:
            # calculate hartree potential
            hartree(0, n * r * dr, self.beta, self.N, vHr)

            # add potential from nuclear point charge (v = -Z / r)
            vHr -= Z

            # calculated exchange correlation potential and energy
            vXC[:] = 0.0
            Exc = self.xc.get_energy_and_potential(n, vXC)

            # calculate new total Kohn-Sham effective potential and
            # admix with old version
            vr[:] = vHr + vXC * r
            if niter > 0:
                vr[:] = 0.4 * vr + 0.6 * vrold
            vrold = vr.copy()

            # solve Kohn-Sham equation and determine the density change
            self.solve()
            dn = self.calculate_density() - n
            n += dn

            # estimate error from the square of the density change integrated
            q = log(num.sum((r * dn)**2))

            # print progress bar
            if niter == 0:
                q0 = q
                b0 = 0
            else:
                b = int((q0 - q) / (q0 - qOK) * 50)
                if b > b0:
                    sys.stdout.write(bar[b0:min(b, 50)])
                    sys.stdout.flush()
                    b0 = b

            # check if converged and break loop if so
            if q < qOK:
                sys.stdout.write(bar[b0:])
                sys.stdout.flush()
                break
            
            niter += 1
            if niter > 117:
                raise RuntimeError, 'Did not converge!'
            
        print
        print 'Converged in %d iteration%s.' % (niter, 's'[:niter != 1])
        
        pickle.dump(n, open(self.symbol + '.restart', 'w'))

        Epot = 2 * pi * num.dot(n * r * (vHr - Z), dr)
        Ekin = -4 * pi * num.dot(n * vr * r, dr)
        for f, e in zip(f_j, e_j):
            Ekin += f * e

        print
        print 'Energy contributions:'
        print '-------------------------'
        print 'Kinetic:   %+13.6f' % Ekin
        print 'XC:        %+13.6f' % Exc
        print 'Potential: %+13.6f' % Epot
        print '-------------------------'
        print 'Total:     %+13.6f' % (Ekin + Exc + Epot)
        print

        print 'state    eigenvalue         ekin         rmax'
        print '---------------------------------------------'
        for m, l, f, e, u in zip(n_j, l_j, f_j, e_j, self.u_j):
            # Find kinetic energy:
            k = e - num.sum((num.where(abs(u) < 1e-160, 0, u)**2 * #XXXNumeric!
                             vr * dr)[1:] / r[1:])
            
            # Find outermost maximum:
            g = self.N - 4
            while u[g - 1] > u[g]:
                g -= 1
            x = r[g - 1:g + 2]
            y = u[g - 1:g + 2]
            A = num.transpose(num.array([x**i for i in range(3)]))
            c, b, a = linalg.solve_linear_equations(A, y)
            assert a < 0.0
            rmax = -0.5 * b / a
            
            t = 'spdf'[l]
            print '%d%s^%-2d: %12.6f %12.6f %12.3f' % (m, t, f, e, k, rmax)
        print '---------------------------------------------'
        print '(units: Bohr and Hartree)'
        
        for m, l, u in zip(n_j, l_j, self.u_j):
            self.write(u, 'ae', n=m, l=l)
            
        self.write(n, 'n')
        self.write(vr, 'vr')
        self.write(vHr, 'vHr')
        self.write(vXC, 'vXC')
        
        self.Ekin = Ekin
        self.Epot = Epot
        self.Exc = Exc

    def write(self, array, name=None, n=None, l=None):
        if name:
            name = self.symbol + '.' + name
        else:
            name = self.symbol
            
        if l is not None:
            assert n is not None
            if n > 0:
                # Bound state:
                name += '.%d%s' % (n, 'spdf'[l])
            else:
                name += '.x%d%s' % (-n, 'spdf'[l])
                
        f = open(name, 'w')
        for r, a in zip(self.r, array):
            print >> f, r, a
    
    def calculate_density(self):
        """Return the electron charge density divided by 4 pi"""
        n = num.dot(self.f_j,
                    num.where(abs(self.u_j) < 1e-160, 0,
                              self.u_j)**2) / (4 * pi)
        n[1:] /= self.r[1:]**2
        n[0] = n[1]
        return n
    
    def solve(self):
        """Solve the Schrodinger equation

        ::
        
             2 
            d u     1  dv  du   u     l(l + 1)
          - --- - ---- -- (-- - -) + [-------- + 2M(v - e)] u = 0,
              2      2 dr  dr   r         2
            dr    2Mc                    r

        
        where the relativistic mass::

                   1
          M = 1 - --- (v - e)
                    2
                  2c
        
        and the fine-structure constant alpha = 1/c = 1/137.036
        is set to zero for non-scalar-relativistic calculations.

        On the logaritmic radial grids defined by::

              beta g
          r = ------,  g = 0, 1, ..., N - 1
              N - g

                 rN
          g = --------, r = [0; oo[
              beta + r
  
        the Schrodinger equation becomes::
        
           2 
          d u      du  
          --- c  + -- c  + u c  = 0
            2  2   dg  1      0
          dg

        with the vectors c0, c1, and c2  defined by::

                 2 dg 2
          c  = -r (--)
           2       dr
        
                  2         2
                 d g  2    r   dg dv
          c  = - --- r  - ---- -- --
           1       2         2 dr dr
                 dr       2Mc
          
                                    2    r   dv
          c  = l(l + 1) + 2M(v - e)r  + ---- --
           0                               2 dr
                                        2Mc
        """
        r = self.r
        dr = self.dr
        vr = self.vr
        
        c2 = -(r / dr)**2
        c10 = -self.d2gdr2 * r**2 # first part of c1 vector
        
        if self.scalarrel:
            self.r2dvdr = num.zeros(self.N, num.Float)
            self.rgd.derivative(vr, self.r2dvdr)
            self.r2dvdr *= r
            self.r2dvdr -= vr
        else:
            self.r2dvdr = None
            
        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j,
                                             self.e_j, self.u_j)):
            nodes = n - l - 1 # analytically expected number of nodes
            delta = -0.2 * e
            nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel)
            # adjust eigenenergy until u has the correct number of nodes
            while nn != nodes:
                diff = cmp(nn, nodes)
                while diff == cmp(nn, nodes):
                    e -= diff * delta
                    nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                                  self.scalarrel)
                delta /= 2

            # adjust eigenenergy until u is smooth at the turning point
            de = 1.0
            while abs(de) > 1e-9:
                norm = num.dot(num.where(abs(u) < 1e-160, 0, u)**2, dr)
                u *= 1.0 / sqrt(norm)
                de = 0.5 * A / norm
                x = abs(de / e)
                if x > 0.1:
                    de *= 0.1 / x
                e -= de
                assert e < 0.0
                nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                              self.scalarrel)
            self.e_j[j] = e
            u *= 1.0 / sqrt(num.dot(num.where(abs(u) < 1e-160, 0, u)**2, dr))

    def kin(self, l, u, e=None): # XXX move to Generator
        r = self.r[1:]
        dr = self.dr[1:]
        
        c0 = 0.5 * l * (l + 1) / r**2
        c1 = -0.5 * self.d2gdr2[1:]
        c2 = -0.5 * dr**-2
        
        if e is not None and self.scalarrel:
            x = 0.5 * alpha**2
            Mr = r * (1.0 + x * e) - x * self.vr[1:]
            c0 += ((Mr - r) * (self.vr[1:] - e * r) +
                   0.5 * x * self.r2dvdr[1:] / Mr) / r**2
            c1 -= 0.5 * x * self.r2dvdr[1:] / (Mr * dr * r)

        fp = c2 + 0.5 * c1
        fm = c2 - 0.5 * c1
        f0 = c0 - 2 * c2
        kr = num.zeros(self.N, num.Float)
        kr[1:] = f0 * u[1:] + fm * u[:-1]
        kr[1:-1] += fp[:-1] * u[2:]
        kr[0] = 0.0
        return kr    

def shoot(u, l, vr, e, r2dvdr, r, dr, c10, c2, scalarrel=False, gmax=None):
    """n, A = shoot(u, l, vr, e, ...)
    
    For guessed trial eigenenergy e, integrate the radial Schrodinger
    equation::

          2 
         d u      du  
         --- c  + -- c  + u c  = 0
           2  2   dg  1      0
         dg
        
               2 dg 2
        c  = -r (--)
         2       dr
        
                2         2
               d g  2    r   dg dv
        c  = - --- r  - ---- -- --
         1       2         2 dr dr
               dr       2Mc
        
                                  2    r   dv
        c  = l(l + 1) + 2M(v - e)r  + ---- --
         0                               2 dr
                                      2Mc
                                      
    The resulting wavefunction is returned in input vector u.
    The number of nodes of u is returned in attribute n.
    Returned attribute A, is a measure of the size of the derivative
    discontinuity at the classical turning point.
    The trial energy e is correct if A is zero and n is the correct number
    of nodes."""
    
    if scalarrel:
        x = 0.5 * alpha**2 # x = 1 / (2c^2)
        Mr = r * (1.0 + x * e) - x * vr
    else:
        Mr = r
    c0 = l * (l + 1) + 2 * Mr * (vr - e * r)
    if gmax is None and num.alltrue(c0 > 0):
        print """
Problem with initial electron density guess!  Try to run the program
with the '-n' option (non-scalar-relativistic calculation) and then
try again without the '-n' option (this will generate a good initial
guess for the density).
"""
        raise SystemExit
    c1 = c10
    if scalarrel:
        c0 += x * r2dvdr / Mr
        c1 = c10 - x * r * r2dvdr / (Mr * dr)

    # vectors needed for numeric integration of diff. equation
    fm = 0.5 * c1 - c2
    fp = 0.5 * c1 + c2
    f0 = c0 - 2 * c2
    
    if gmax is None:
        # set boundary conditions at r -> oo (u(oo) = 0 is implicit)
        u[-1] = 1.0
        
        # perform backwards integration from infinity to the turning point
        g = len(u) - 2
        u[-2] = u[-1] * f0[-1] / fm[-1]
        while c0[g] > 0.0: # this defines the classical turning point
            u[g - 1] = (f0[g] * u[g] + fp[g] * u[g + 1]) / fm[g]
            if u[g - 1] < 0.0:
                # There should't be a node here!  Use a more negative
                # eigenvalue:
                print '!!!!!!',
                return 100, None
            if u[g - 1] > 1e100:
                u *= 1e-100
            g -= 1

        # stored values of the wavefunction and the first derivative
        # at the turning point
        gtp = g + 1 
        utp = u[gtp]
        dudrplus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    else:
        gtp = gmax

    # set boundary conditions at r -> 0
    u[0] = 0.0
    u[1] = 1.0
    
    # perform forward integration from zero to the turning point
    g = 1
    nodes = 0
    while g <= gtp: # integrate one step further than gtp
                    # (such that dudr is defined in gtp)
        u[g + 1] = (fm[g] * u[g - 1] - f0[g] * u[g]) / fp[g]
        if u[g + 1] * u[g] < 0:
            nodes += 1
        g += 1
    if gmax is not None:
        return

    # scale first part of wavefunction, such that it is continuous at gtp
    u[:gtp + 2] *= utp / u[gtp]

    # determine size of the derivative discontinuity at gtp
    dudrminus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    A = (dudrplus - dudrminus) * utp
    
    return nodes, A
            
if __name__ == '__main__':
    a = AllElectron('Cu', scalarrel=True)
    a.run()
