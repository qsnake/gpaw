"""This module is used to generate atomic orbital basis sets."""

import os
import sys
from StringIO import StringIO

from math import pi, cos, sin
import numpy as npy
from numpy.linalg import solve
from ase.units import Hartree

from gpaw.spline import Spline
from gpaw.atom.generator import Generator, parameters
from gpaw.atom.polarization import PolarizationOrbitalGenerator, Reference
from gpaw.utilities import devnull, divrl
from gpaw.basis_data import Basis, BasisFunction
from gpaw.version import version

AMPLITUDE = 100. # default confinement potential modifier

class BasisMaker:
    """Class for creating atomic basis functions."""
    def __init__(self, generator, name=None, run=True):
        if isinstance(generator, str): # treat 'generator' as symbol
            generator = Generator(generator, scalarrel=True)
        self.generator = generator
        self.name = name
        if run:
            generator.N = 2000
            generator.run(write_xml=False, **parameters[generator.symbol])

    def get_unsmoothed_projector_coefficients(self, psi_jg, l):
        """Calculates scalar products of psi with non-smoothed projectors.

        Returns a matrix with (i,j)'th element equal to::

            < p  | psi  > ,
               i      j

        where the argument psi is the coefficient matrix for a system of
        vectors, and where p_i are determined by::

                            -----
               ~             \      ~
             < p  | psi  > =  )   < p  | phi  > < p  | psi  > ,
                i      j     /       i      k      k      j
                            -----
                              k

        where p_i-tilde are the projectors and phi_k the AE partial waves.
        """
        if npy.rank(psi_jg) == 1:
            # vector/matrix polymorphism hack
            return self.get_unsmoothed_projector_coefficients([psi_jg], l)[0]

        g = self.generator
        u = g.u_ln[l]
        q = g.q_ln[l]

        m = len(q)
        n = len(u)

        A = npy.zeros((m, n))
        b = npy.zeros((m, len(psi_jg)))

        # This can probably be done in the constructor once and for all
        # Not to mention that it can be replaced by a matrix multiplication
        for i in range(m):
            for j in range(n):
                A[i, j] = npy.dot(g.dr, q[i] * u[j])

        for i in range(m):
            for j in range(len(psi_jg)):
                b[i, j] = npy.dot(g.dr, q[i] * psi_jg[j])

        p = solve(A, b)
        return p

    def unsmoothify(self, psit_jg, l):
        """Given smooth functions psit, return non-smooth ones.
        
        Converts each column of psit, interpreted as a pseudo wave
        function, to original wave functions using the formula::

                                  -----
                          ~        \    /             ~    \    ~     ~
            | psi  > = | psi  > +   )  ( | phi > - | phi  > ) < p  | psi  >
                 i          i      /    \     j         j  /     j      i
                                  -----
                                    j
        """
        if npy.rank(psit_jg) == 1:
            # vector/matrix polymorphism hack
            return self.unsmoothify([psit_jg], l)[0]
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psi_jg = [psit_jg[j] + sum([(u[i]-s[i])*q[i,j]
                                    for i in range(len(s))])
               for j in range(len(psit_jg))]
        return psi_jg

    def smoothify(self, psi_jg, l):
        """Given non-smooth functions psi, return smooth ones.
        
        Converts each column of psi, interpreted as an all-electron
        wave function, to pseudo wave functions using the formula::

                                  -----
               ~                   \    /   ~              \           
            | psi  > = | psi  > +   )  ( | phi > - | phi  > ) < p  | psi  >
                 i          i      /    \     j         j  /     j      i
                                  -----
                                    j

        """
        if npy.rank(psi_jg) == 1:
            # vector/matrix polymorphism hack
            return self.smoothify([psi_jg], l)[0]
        
        p = self.get_unsmoothed_projector_coefficients(psi_jg, l)
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psit_jg = [psi_jg[j] + sum([(s[i]-u[i])*p[i,j]
                              for i in range(len(s))])
                for j in range(len(psi_jg))]
        return psit_jg

    def make_orbital_vector(self, j, rcut, vconf=None):
        """Returns a smooth basis vector given an all-electron one."""
        l = self.generator.l_j[j]
        psi_g, e = self.generator.solve_confined(j, rcut, vconf)
        psit_g = self.smoothify(psi_g, l)
        return psit_g

    def make_split_valence_vector(self, psi_g, l, rcut):
        """Get polynomial which joins psi smoothly at rcut.

        Returns an array of function values f(r) * r, where::
        
                  l           2
          f(r) = r  * (a - b r ),  r < rcut
          f(r) = psi(r),           r >= rcut

        where a and b are determined such that f(r) is continuous and
        differentiable at rcut.  The parameter psi should be an atomic
        orbital.
        """
        g = self.generator
        icut = g.r2g(rcut)
        r1 = g.r[icut] # ensure that rcut is moved to a grid point
        r2 = g.r[icut + 1]
        y1 = psi_g[icut] / g.r[icut]
        y2 = psi_g[icut + 1] / g.r[icut + 1]
        b = - (y2 / r2**l - y1 / r1**l)/(r2**2 - r1**2)
        a = (y1 / r1**l + b * r1**2)
        psi_g2 = g.r**(l + 1) * (a - b * g.r**2)
        psi_g2[icut:] = psi_g[icut:]
        return psi_g2

    def make_polarization_function(self, rcut, l, referencefile=None, 
                                   index=None, ngaussians=None, txt=devnull):
        """Generate polarization function using the polarization module."""
        symbol = self.generator.symbol
        ref = Reference(symbol, referencefile, index)
        gd, kpt_u, center = ref.get_reference_data()
        symbols = ref.atoms.get_chemical_symbols()
        symbols[ref.index] = '[%s]' % symbols[ref.index] # mark relevant atom

        print >> txt, 'Reference system [ %s ]:' % ref.filename,
        print >> txt, ' '.join(['%s' % sym for sym in symbols])
        cell = ' x '.join(['%.02f' % a for a in ref.cell])
        print >> txt, 'Cell = %s :: gpts = %s' % (cell, ref.gpts)
        generator = PolarizationOrbitalGenerator(rcut, gaussians=ngaussians)
        y = generator.generate(l, gd, kpt_u, center)
        print >> txt, 'Quasi Gaussians: %d' % len(generator.alphas)
        r_alphas = generator.r_alphas
        print >> txt, 'Gaussian characteristic lengths evenly distributed'
        print >> txt, 'Rchars from %.03f to %.03f' % (min(r_alphas),
                                                      max(r_alphas))
        print >> txt, 'k-points: %d' % len(kpt_u)
        print >> txt, 'Reference states: %d' % len(kpt_u[0].psit_nG)
        print >> txt, 'Quality: %.03f' % generator.quality

        rowstrings = [' '.join(['%4.2f' % f for f in row])
                      for row in generator.qualities]

        # fancy formatting
        rowcount, columncount = generator.qualities.shape
        rowheader = list('-' * columncount)
        columnheader = list('|' * rowcount)
        columnheader[rowcount // 2] = 'k'
        print >> txt, ' ', ' m '.center(len(rowstrings[0]), '-')
        for char, string in zip(columnheader, rowstrings):
            print >>  txt, char, string

        r = self.generator.r
        psi = r**l * y(r)
        return psi * r # Recall that wave functions are represented as psi*r

    def make_mock_vector(self, rcut, l):
        """Return orbital-like polynomial."""
        r = self.generator.r
        x = r / rcut
        y = (1 - 3 * x**2 + 2 * x**3) * x**l
        icut = self.generator.r2g(rcut)
        y[icut:] *= 0
        return y * r # Recall that wave functions are represented as psi*r

    def rcut_by_energy(self, j, esplit=.1, tolerance=.1, rguess=6.):
        """Find confinement cutoff corresponding to given orbital energy shift.

        Creates a confinement potential for the orbital given by j,
        such that the confined-orbital energy is (emin to emax) eV larger
        than the free-orbital energy."""
        g = self.generator
        e_base = g.e_j[j]
        rc = rguess
        ri = rc * .6
        vconf = g.get_confinement_potential(AMPLITUDE, ri, rc)

        psi_g, e = g.solve_confined(j, rc, vconf)
        de_min, de_max = esplit/Hartree, (esplit+tolerance)/Hartree

        rmin = 0.
        rmax = g.r[-1]
        i_left = g.r2g(rmin)
        i_right = g.r2g(rmax)

        de = e - e_base
        #print '--------'
        #print 'Start bisection'
        #print'e_base =',e_base
        #print 'e =',e
        #print '--------'
        while de < de_min or de > de_max:
            if de < de_min: # Move rc left -> smaller cutoff, higher energy
                rmax = rc
                rc = (rc + rmin) / 2.
            else: # Move rc right
                rmin = rc
                rc = (rc + rmax) / 2.
            ri = rc * .6
            vconf = g.get_confinement_potential(AMPLITUDE, ri, rc)
            psi_g, e = g.solve_confined(j, rc, vconf)
            de = e - e_base
            #print 'rc = %.03f :: e = %.03f :: de = %.03f' % (rc, e*Hartree,
            #                                                 de*Hartree)
        #print 'Done!'
        return psi_g, e, de, vconf, ri, rc

    def rsplit_by_norm(self, l, u, tailnorm, txt):
        g = self.generator
        norm = npy.dot(g.dr, u*u)
        partial_norm = 0.
        i = len(u) - 1
        while partial_norm / norm < tailnorm:
            # Integrate backwards.  This is important since the pseudo
            # wave functions have strange behaviour near the core.
            partial_norm += g.dr[i] * u[i]**2
            i -= 1
        rsplit = g.r[i+1]
        msg = ('Tail norm %.03f :: rsplit=%.02f Bohr' %
               (partial_norm, rsplit))
        print >> txt, msg
        splitwave = self.make_split_valence_vector(u, l, rsplit)
        return rsplit, partial_norm, splitwave

    def generate(self, zetacount=2, polarizationcount=1, 
                 tailnorm=(.15, .25, .35), energysplit=.2, tolerance=1.0e-3, 
                 referencefile=None, referenceindex=None, rcutpol_rel=1., 
                 rcutmax=20., ngaussians=None, txt='-'):
        """Generate an entire basis set."""
        if txt == '-':
            txt = sys.stdout
        elif txt is None:
            txt = devnull

        if isinstance(tailnorm, float):
            tailnorm = (tailnorm,)
        assert (1 + len(tailnorm) >= max(polarizationcount, zetacount),
                'Needs %d tail norm values, but only %d are specified' %
                (max(polarizationcount, zetacount) - 1, len(tailnorm)))

        buffer = StringIO()
        class TeeStream: # Quick hack to both write and save output
            def __init__(self, out1, out2):
                self.out1 = out1
                self.out2 = out2
            def write(self, string):
                self.out1.write(string)
                self.out2.write(string)
        txt = TeeStream(txt, buffer)

        # Find out all relevant orbitals
        # We'll probably need: s, p and d.
        # The orbitals we want are stored in u_j.
        # Thus we must find the j corresponding to the highest energy of
        # each orbital-type.
        #
        # ASSUMPTION: The last index of a given value in l_j corresponds
        # exactly to the orbital we want.
        g = self.generator
        print >> txt, 'Basis functions for %s' % g.symbol
        print >> txt, '====================' + '='*len(g.symbol)
        print >> txt
        lmax = max(g.l_j)
        lvalues = range(lmax + 1)
        
        j_l = [] # index j by l rather than the other way around
        reversed_l_j = list(g.l_j)
        reversed_l_j.reverse() # the values we want are stored last
        for l in lvalues:
            j = len(reversed_l_j) - reversed_l_j.index(l) - 1
            j_l.append(j)

        singlezetas = []
        multizetas = [[] for i in range(zetacount - 1)]
        polarization_functions = []

        splitvalencedescr = 'split-valence wave, fixed tail norm'

        for l in lvalues:
            # Get one unmodified pseudo-orbital basis vector for each l
            j = j_l[l]
            n = g.n_j[j]
            orbitaltype = str(n) + 'spdf'[l]
            msg = 'Basis functions for l=%d, n=%d' % (l, n)
            print >> txt, msg + '\n', '-' * len(msg)
            print >> txt
            print >> txt, 'Zeta 1: softly confined pseudo wave,',
            u, e, de, vconf, ri, rc = self.rcut_by_energy(j, energysplit,
                                                          tolerance)
            if rc > rcutmax:
                ri = ri * rc / rcutmax # scale things down
                rc = rcutmax
                vconf = g.get_confinement_potential(AMPLITUDE, ri, rc)
                u, e = g.solve_confined(j, rc, vconf)
                print >> txt, 'using maximum cutoff'
                print >> txt, 'rc=%.02f Bohr' % rc
            else:
                print >> txt, 'fixed energy shift'    
                print >> txt, 'DE=%.03f eV :: rc=%.02f Bohr' % (de * Hartree,
                                                                rc)
            phit_g = self.smoothify(u, l)
            bf = BasisFunction(l, rc, phit_g,
                               '%s-sz confined orbital' % orbitaltype)
            singlezetas.append(bf)

            for i in range(zetacount - 1):
                zeta = i + 2
                print >> txt, '\nZeta %d: %s' % (zeta, splitvalencedescr)
                rsplit, norm, splitwave = self.rsplit_by_norm(l, phit_g,
                                                              tailnorm[i],
                                                              txt)
                descr = '%s-%sz split-valence wave' % (orbitaltype,
                                                       'dtq56789'[i])
                bf = BasisFunction(l, rsplit, phit_g - splitwave, descr)
                multizetas[i].append(bf)
                #doublezetas.append(bf)
            
            """if zetacount > 1:
                # add one split-valence vector using fixed-tail-norm scheme
                print >> txt, '\nZeta 2: split-valence wave, fixed tail norm'

                rsplit = find_rsplit_by_norm(u, tailnorm[0])

                msg = 'Tail norm %.03f :: rsplit=%.02f Bohr' % (partial_norm,
                                                                rsplit)
                print >> txt, msg
                splitwave = self.make_split_valence_vector(phit_g, l, rsplit)
                bf_dz = BasisFunction(l, rsplit, phit_g - splitwave, 
                                      '%s-dz split-valence wave' % orbitaltype)

                doublezetas.append(bf_dz)

                # If there are even more zetas, make new, smaller split radii
                # We'll just distribute them evenly between 0 and rsplit
                extra_split_radii = npy.linspace(rsplit, 0., zetacount)[1:-1]
                for i, rsplit in enumerate(extra_split_radii):
                    print >> txt, '\nZeta %d: extra split-valence wave' % (3+i)
                    print >> txt, 'rsplit=%.02f Bohr' % rsplit
                    splitwave = self.make_split_valence_vector(phit_g, l, 
                                                               rsplit)
                    bf_multizeta = BasisFunction(l, rsplit, phit_g - splitwave,
                                                 '%s-%sz split-valence wave' 
                                                 % (orbitaltype, 'tq5678'[i]))
                    other_multizetas[i].append(bf_multizeta)
               """
            
        if polarizationcount > 0:
            # Now make up some properties for the polarization orbital
            # We just use the cutoffs from the previous one times a factor
            rcut = max([bf.rc for bf in singlezetas]) * rcutpol_rel
            rcut = min(rcut, rcutmax)
            l_pol = lmax + 1
            msg = 'Polarization function: l=%d, rc=%.02f' % (l_pol, rcut)
            print >> txt, '\n' + msg
            print >> txt, '-' * len(msg)
            psi_pol = self.make_polarization_function(rcut, l_pol,
                                                      referencefile,
                                                      referenceindex,
                                                      ngaussians,
                                                      txt)
            
            # We'll just make a hack here to make it go more smoothly to zero
            #gc1 = g.r2g(.1*rcut)
            #gc2 = g.r2g(.4*rcut) + 1
            #ri = g.r[gc1]
            #rc = g.r[gc2 - 1]

            #R = (g.r[gc1:gc2]-ri) / (rc-ri)
            #F = 1 - 3 * R**2 + 2 * R**3
            #psi_pol[gc1:gc2] *= F
            #psi_pol[gc2:] = 0
            #print >> txt, 'Forced cutoff over %.03f to %.03f !!' % (ri, rc)
            
            bf_pol = BasisFunction(l_pol, rcut, psi_pol, 
                                   '%s-type polarization' % 'spdfg'[l_pol])
            polarization_functions.append(bf_pol)
            for i in range(polarizationcount - 1):
                npol = i + 2
                msg = '\n%s: %s' % (['Secondary', 'Tertiary'][i],
                                    splitvalencedescr)
                print >> txt, msg
                rsplit, norm, splitwave = self.rsplit_by_norm(l_pol, psi_pol,
                                                              tailnorm[i],
                                                              txt)
                descr = ('%s-type split-valence polarization %d'
                         % ('spdfg'[l_pol], npol))
                bf_pol = BasisFunction(l_pol, rsplit, psi_pol - splitwave,
                                       descr)
                polarization_functions.append(bf_pol)
                    
        bf_j = []
        bf_j.extend(singlezetas)
        for multizeta_list in multizetas:
            bf_j.extend(multizeta_list)
        bf_j.extend(polarization_functions)

        rcmax = max([bf.rc for bf in bf_j])

        equidistant_grid = npy.linspace(0., rcmax, 2**10)
        for bf in bf_j:
            norm = npy.dot(self.generator.dr, bf.phit_g * bf.phit_g)**.5
            bf.phit_g /= norm
            # We have been storing phit_g * r, but we just want phit_g
            bf.phit_g = divrl(bf.phit_g, 1, g.r)

            # Quick hack to change to equidistant coordinates
            spline = Spline(bf.l, g.r[g.r2g(bf.rc)],
                            bf.phit_g,# * g.r**bf.l, 
                            g.r, beta=g.beta, points=100)
            bf.phit_g = npy.array([spline(r) * r**bf.l
                                   for r in equidistant_grid])
        
        basis = Basis(g.symbol, self.name, False)
        basis.ng = len(equidistant_grid)
        basis.d = equidistant_grid[1]
        basis.bf_j = bf_j
        basis.generatordata = buffer.getvalue().strip()
        basis.generatorattrs = {'version' : version}
        buffer.close()

        return basis

    def grplot(self, bf_j):
        """Plot basis functions on generator's radial grid."""
        import pylab
        rc = max([bf.rc for bf in bf_j])
        g = self.generator
        r = g.r
        for bf in bf_j:
            label = bf.type
            # XXX times g.r or not times g.r ?
            pylab.plot(r, bf.phit_g / r, label=label[:12])
        axis = pylab.axis()
        newaxis = [0., rc, axis[2], axis[3]]
        pylab.axis(newaxis)
        pylab.legend()
        pylab.show()

    def plot(self, basis, figure=None, title=None, filename=None):
        """Plot basis functions using pylab."""
        # XXX method should no longer belong to a basis maker
        import pylab
        rc = max([bf.rc for bf in basis.bf_j])
        r = npy.linspace(0., basis.d * (basis.ng - 1), basis.ng)
        g = self.generator
        if figure is not None:
            pylab.figure(figure)
        else:
            pylab.figure() # not very elegant
        if title is None:
            title = g.symbol
        pylab.title(title)
        for bf in basis.bf_j:
            label = bf.type
            # XXX times g.r or not times g.r ?
            pylab.plot(r, bf.phit_g * r, label=label[:12])
        axis = pylab.axis()
        newaxis = [0., rc, axis[2], axis[3]]
        pylab.axis(newaxis)
        pylab.legend()
        if filename is not None:
            pylab.savefig(filename)
