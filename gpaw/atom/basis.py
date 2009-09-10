"""This module is used to generate atomic orbital basis sets."""

import os
import sys
from StringIO import StringIO

import numpy as npy
from numpy.linalg import solve
from ase.units import Hartree

from gpaw.spline import Spline
from gpaw.atom.all_electron import AllElectron, ConvergenceError
from gpaw.atom.generator import Generator, parameters
from gpaw.atom.polarization import PolarizationOrbitalGenerator, Reference,\
     QuasiGaussian, default_rchar_rel, rchar_rels
from gpaw.utilities import devnull, divrl
from gpaw.basis_data import Basis, BasisFunction
from gpaw.version import version


class BasisMaker:
    """Class for creating atomic basis functions."""
    def __init__(self, generator, name=None, run=True, gtxt='-',
                 non_relativistic_guess=False, xc='PBE'):
        if isinstance(generator, str): # treat 'generator' as symbol
            generator = Generator(generator, scalarrel=True,
                                  xcname=xc, txt=gtxt,
                                  nofiles=True)
            generator.N *= 4
        self.generator = generator
        self.name = name
        if run:
            if non_relativistic_guess:
                ae0 = AllElectron(generator.symbol, scalarrel=False,
                                  nofiles=False, txt=gtxt, xcname=xc)
                ae0.N = generator.N
                ae0.beta = generator.beta
                ae0.run()
                # Now files will be stored such that they can
                # automagically be used by the next run()
            generator.run(write_xml=False, **parameters[generator.symbol])

    def smoothify(self, psi_mg, l):
        """Generate pseudo wave functions from all-electron ones.

        The pseudo wave function is::
        
                                   ___
               ~                   \    /   ~             \    ~     ~
            | psi  > = | psi  > +   )  | | phi > - | phi > ) < p  | psi > ,
                 m          m      /__  \     i         i /     i      m
                                    i

        where the scalar products are found by solving::
        
                            ___
              ~             \     ~             ~     ~
            < p | psi  > =   )  < p  | phi  > < p  | psi  > .
               i     m      /__    i      j      j      m
                             j

        In order to ensure smoothness close to the core, the
        all-electron wave function and partial wave are then
        multiplied by a radial function which approaches 0 near the
        core, such that the pseudo wave function approaches::

                        ___
               ~        \      ~       ~     ~
            | psi  > =   )  | phi >  < p  | psi >    (for r << rcut),
                 m      /__      i      i      m
                         i

        which is exact if the projectors/pseudo partial waves are complete.
        """  
        if npy.rank(psi_mg) == 1:
            return self.smoothify(psi_mg[None], l)[0]

        g = self.generator
        u_ng = g.u_ln[l]
        q_ng = g.q_ln[l]
        s_ng = g.s_ln[l]

        Pi_nn = npy.dot(g.r * q_ng, u_ng.T)
        Q_nm = npy.dot(g.r * q_ng, psi_mg.T)
        Qt_nm = npy.linalg.solve(Pi_nn, Q_nm)

        # Weight-function for truncating all-electron parts smoothly near core
        gmerge = g.r2g(g.rcut_l[l])
        w_g = npy.ones(g.r.shape)
        w_g[0:gmerge] = (g.r[0:gmerge] / g.r[gmerge])**2.
        w_g = w_g[None]
        
        psit_mg = psi_mg * w_g + npy.dot(Qt_nm.T, s_ng - u_ng * w_g)
        return psit_mg

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
        raise DeprecationWarning
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
        raise DeprecationWarning
        if npy.rank(psit_jg) == 1:
            # vector/matrix polymorphism hack
            return self.unsmoothify([psit_jg], l)[0]
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psi_jg = [psit_jg[j] + sum([(u[i] - s[i]) * q[i, j]
                                    for i in range(len(s))])
               for j in range(len(psit_jg))]
        return psi_jg

    def old_smoothify(self, psi_jg, l):
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
        raise DeprecationWarning
        if npy.rank(psi_jg) == 1:
            # vector/matrix polymorphism hack
            return self.old_smoothify([psi_jg], l)[0]
        
        p = self.get_unsmoothed_projector_coefficients(psi_jg, l)
        g = self.generator
        (q, u, s) = (g.q_ln[l], g.u_ln[l], g.s_ln[l])
        
        psit_jg = [psi_jg[j] + sum([(s[i] - u[i]) * p[i, j]
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

        print >> txt, 'Coefficients:', ' '.join(['%5.2f' % f for f in y.coefs])

        rowstrings = [' '.join(['%4.2f' % f for f in row])
                      for row in generator.qualities]

        # fancy formatting
        rowcount, columncount = generator.qualities.shape
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

    def rcut_by_energy(self, j, esplit=.1, tolerance=.1, rguess=6.,
                       vconf_args=None):
        """Find confinement cutoff corresponding to given orbital energy shift.

        Creates a confinement potential for the orbital given by j,
        such that the confined-orbital energy is (emin to emax) eV larger
        than the free-orbital energy."""
        g = self.generator
        e_base = g.e_j[j]
        rc = rguess

        if vconf_args is None:
            vconf = None
        else:
            amplitude, ri_rel = vconf_args
            vconf = g.get_confinement_potential(amplitude, ri_rel * rc, rc)

        psi_g, e = g.solve_confined(j, rc, vconf)
        de_min, de_max = esplit / Hartree, (esplit + tolerance) / Hartree

        rmin = 0.
        rmax = g.r[-1]

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
            if vconf is not None:
                vconf = g.get_confinement_potential(amplitude, ri_rel * rc, rc)
            psi_g, e = g.solve_confined(j, rc, vconf)
            de = e - e_base
            #print 'rc = %.03f :: e = %.03f :: de = %.03f' % (rc, e*Hartree,
            #                                                 de*Hartree)
            #if rmin - rmax < 1e-
            if g.r2g(rmax) - g.r2g(rmin) <= 1: # adjacent points
                break # Cannot meet tolerance due to grid resolution
        #print 'Done!'
        return psi_g, e, de, vconf, rc

    def rsplit_by_norm(self, l, u, tailnorm_squared, txt):
        """Find radius outside which remaining tail has a particular norm."""
        g = self.generator
        norm_squared = npy.dot(g.dr, u*u)
        partial_norm_squared = 0.
        i = len(u) - 1
        absolute_tailnorm_squared = tailnorm_squared * norm_squared
        while partial_norm_squared < absolute_tailnorm_squared:
            # Integrate backwards.  This is important since the pseudo
            # wave functions have strange behaviour near the core.
            partial_norm_squared += g.dr[i] * u[i]**2
            i -= 1
        rsplit = g.r[i+1]
        msg = ('Tail norm %.03f :: rsplit=%.02f Bohr' %
               ((partial_norm_squared / norm_squared)**.5, rsplit))
        print >> txt, msg
        splitwave = self.make_split_valence_vector(u, l, rsplit)
        return rsplit, partial_norm_squared, splitwave

    def generate(self, zetacount=2, polarizationcount=1,
                 tailnorm=(0.16, 0.3, 0.6), energysplit=0.1, tolerance=1.0e-3,
                 referencefile=None, referenceindex=None, rcutpol_rel=1.0, 
                 rcutmax=20.0, #ngaussians=None,
                 rcharpol_rel=None,
                 vconf_args=(12.0, 0.6), txt='-',
                 include_energy_derivatives=False,
                 lvalues=None):
        """Generate an entire basis set.

        This is a high-level method which will return a basis set
        consisting of several different basis vector types.

        Parameters:

        ===================== =================================================
        ``zetacount``         Number of basis functions per occupied orbital
        ``polarizationcount`` Number of polarization functions
        ``tailnorm``          List of tail norms for split-valence scheme
        ``energysplit``       Energy increase defining confinement radius (eV)
        ``tolerance``         Tolerance of energy split (eV)
        ``referencefile``     gpw-file used to generate polarization function
        ``referenceindex``    Index in reference system of relevant atom
        ``rcutpol_rel``       Polarization rcut relative to largest other rcut
        ``rcutmax``           No cutoff will be greater than this value
        ``vconf_args``        Parameters (alpha, ri/rc) for conf. potential
        ``txt``               Log filename or '-' for stdout
        ===================== =================================================

        Returns a fully initialized Basis object.
        """
        if txt == '-':
            txt = sys.stdout
        elif txt is None:
            txt = devnull

        if isinstance(tailnorm, float):
            tailnorm = (tailnorm,)
        assert 1 + len(tailnorm) >= max(polarizationcount, zetacount), \
               'Needs %d tail norm values, but only %d are specified' % \
               (max(polarizationcount, zetacount) - 1, len(tailnorm))

        textbuffer = StringIO()
        class TeeStream: # Quick hack to both write and save output
            def __init__(self, out1, out2):
                self.out1 = out1
                self.out2 = out2
            def write(self, string):
                self.out1.write(string)
                self.out2.write(string)
        txt = TeeStream(txt, textbuffer)

        if vconf_args is not None:
            amplitude, ri_rel = vconf_args

        g = self.generator

        # Find out all relevant orbitals
        # We'll probably need: s, p and d.
        # The orbitals we want are stored in u_j.
        # Thus we must find the j corresponding to the highest energy of
        # each orbital-type.
        #
        # However not all orbitals in l_j are actually occupied, so we
        # will check the occupations in the generator object's lists
        #
        # ASSUMPTION: The last index of a given value in l_j corresponds
        # exactly to the orbital we want, except those which are not occupied
        #
        # Get (only) one occupied valence state for each l
        # Not including polarization in this list
        if lvalues is None:
            lvalues = npy.unique([l for l, f in zip(g.l_j[g.njcore:], 
                                                    g.f_j[g.njcore:])
                                  if f > 0])
            if lvalues[0] != 0: # Always include s-orbital !
                lvalues = npy.array([0] + list(lvalues))
            
        title = '%s Basis functions for %s' % (g.xcname, g.symbol)
        print >> txt, title
        print >> txt, '=' * len(title)
        
        j_l = {} # index j by l rather than the other way around
        reversed_l_j = list(g.l_j)
        reversed_l_j.reverse() # the values we want are stored last
        for l in lvalues:
            j = len(reversed_l_j) - reversed_l_j.index(l) - 1
            j_l[l] = j

        singlezetas = []
        energy_derivative_functions = []
        multizetas = [[] for i in range(zetacount - 1)]
        polarization_functions = []

        splitvalencedescr = 'split-valence wave, fixed tail norm'
        derivativedescr = 'derivative of sz wrt. (ri/rc) of potential'

        for l in lvalues:
            # Get one unmodified pseudo-orbital basis vector for each l
            j = j_l[l]
            n = g.n_j[j]
            orbitaltype = str(n) + 'spdf'[l]
            msg = 'Basis functions for l=%d, n=%d' % (l, n)
            print >> txt
            print >> txt, msg + '\n', '-' * len(msg)
            print >> txt
            if vconf_args is None:
                adverb = 'sharply'
            else:
                adverb = 'softly'
            print >> txt, 'Zeta 1: %s confined pseudo wave,' % adverb,

            u, e, de, vconf, rc = self.rcut_by_energy(j, energysplit,
                                                      tolerance,
                                                      vconf_args=vconf_args)
            if rc > rcutmax:
                rc = rcutmax # scale things down
                if vconf is not None:
                    vconf = g.get_confinement_potential(amplitude, ri_rel * rc,
                                                        rc)
                u, e = g.solve_confined(j, rc, vconf)
                print >> txt, 'using maximum cutoff'
                print >> txt, 'rc=%.02f Bohr' % rc
            else:
                print >> txt, 'fixed energy shift'    
                print >> txt, 'DE=%.03f eV :: rc=%.02f Bohr' % (de * Hartree,
                                                                rc)
            if vconf is not None:
                print >> txt, ('Potential amp=%.02f :: ri/rc=%.02f' %
                               (amplitude, ri_rel))
            phit_g = self.smoothify(u, l)
            bf = BasisFunction(l, rc, phit_g,
                               '%s-sz confined orbital' % orbitaltype)
            norm = npy.dot(g.dr, phit_g * phit_g)**.5
            print >> txt, 'Norm=%.03f' % norm
            singlezetas.append(bf)

            zetacounter = iter(xrange(2, zetacount + 1))

            if include_energy_derivatives:
                assert zetacount > 1
                zeta = zetacounter.next()
                print >> txt, '\nZeta %d: %s' % (zeta, derivativedescr)
                vconf2 = g.get_confinement_potential(amplitude,
                                                     ri_rel * rc * .99, rc)
                u2, e2 = g.solve_confined(j, rc, vconf2)
                
                phit2_g = self.smoothify(u2, l)
                dphit_g = phit2_g - phit_g
                dphit_norm = npy.dot(g.dr, dphit_g * dphit_g) ** .5
                dphit_g /= dphit_norm
                descr = '%s-dz E-derivative of sz' % orbitaltype
                bf = BasisFunction(l, rc, dphit_g, descr)
                energy_derivative_functions.append(bf)

            for i, zeta in enumerate(zetacounter): # range(zetacount - 1):
                print >> txt, '\nZeta %d: %s' % (zeta, splitvalencedescr)
                # Unresolved issue:  how does the lack of normalization
                # of the first function impact the tail norm scheme?
                # Presumably not much, since most interesting stuff happens
                # close to the core.
                rsplit, norm, splitwave = self.rsplit_by_norm(l, phit_g,
                                                              tailnorm[i]**2.0,
                                                              txt)
                descr = '%s-%sz split-valence wave' % (orbitaltype,
                                                       '0sdtq56789'[zeta])
                bf = BasisFunction(l, rsplit, phit_g - splitwave, descr)
                multizetas[i].append(bf)
            
        if polarizationcount > 0:
            # Now make up some properties for the polarization orbital
            # We just use the cutoffs from the previous one times a factor
            rcut = max([bf.rc for bf in singlezetas]) * rcutpol_rel
            rcut = min(rcut, rcutmax)
            # Find 'missing' values in lvalues
            for i, l in enumerate(lvalues):
                if i != l:
                    l_pol = i
                    break
            else:
                l_pol = lvalues[-1] + 1
            msg = 'Polarization function: l=%d, rc=%.02f' % (l_pol, rcut)
            print >> txt, '\n' + msg
            print >> txt, '-' * len(msg)
            # Make a single Gaussian for polarization function.
            #
            # It is known that for given l, the sz cutoff defined
            # by some fixed energy is strongly correlated to the
            # value of the characteristic radius which best reproduces
            # the wave function found by interpolation.
            #
            # We know that for e.g. d orbitals:
            #   rchar ~= .37 rcut[sz](.3eV)
            # Since we don't want to spend a lot of time finding
            # these value for other energies, we just find the energy
            # shift at .3 eV now

            j = max(j_l.values())
            u, e, de, vconf, rc_fixed = self.rcut_by_energy(j, .3, 1e-2,
                                                            6., (12., .6))

            if rcharpol_rel is None:
                rcharpol_rel = rchar_rels.get(l_pol, default_rchar_rel)
            rchar = rcharpol_rel * rc_fixed
            gaussian = QuasiGaussian(1./rchar**2, rcut)
            psi_pol = gaussian(g.r) * g.r**(l_pol + 1)
            norm = npy.dot(g.dr, psi_pol * psi_pol) ** .5
            psi_pol /= norm
            print >> txt, 'Single quasi Gaussian'
            msg = 'Rchar = %.03f*rcut = %.03f Bohr' % (rcharpol_rel, rchar)
            adjective = 'Gaussian'
            print >> txt, msg
            #else:
            #    psi_pol = self.make_polarization_function(rcut, l_pol,
            #                                              referencefile,
            #                                              referenceindex,
            #                                              ngaussians, txt)
            #    adjective = 'interpolated'

            type = '%s-type %s polarization' % ('spdfg'[l_pol], adjective)
            bf_pol = BasisFunction(l_pol, rcut, psi_pol, type)
                                   
            polarization_functions.append(bf_pol)
            for i in range(polarizationcount - 1):
                npol = i + 2
                msg = '\n%s: %s' % (['Secondary', 'Tertiary', 'Quaternary', \
                                     'Quintary', 'Sextary', 'Septenary'][i],
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
        bf_j.extend(energy_derivative_functions)
        for multizeta_list in multizetas:
            bf_j.extend(multizeta_list)
        bf_j.extend(polarization_functions)
        
        rcmax = max([bf.rc for bf in bf_j])

        # The non-equidistant grids are really only suited for AE WFs
        d = 1./64.
        equidistant_grid = npy.arange(0., rcmax + d, d)
        ng = len(equidistant_grid)

        for bf in bf_j:
            # We have been storing phit_g * r, but we just want phit_g
            bf.phit_g = divrl(bf.phit_g, 1, g.r)
            
            gcut = min(int(1 + bf.rc / d), ng - 1)
            
            assert equidistant_grid[gcut] >= bf.rc
            assert equidistant_grid[gcut - 1] <= bf.rc
            
            bf.rc = equidistant_grid[gcut]
            # Note: bf.rc *must* correspond to a grid point (spline issues)
            bf.ng = gcut + 1
            # XXX all this should be done while building the basis vectors,
            # not here
            
            # Quick hack to change to equidistant coordinates
            spline = Spline(bf.l, g.r[g.r2g(bf.rc)],
                            bf.phit_g,
                            g.r, beta=g.beta, points=100)
            bf.phit_g = npy.array([spline(r) * r**bf.l
                                   for r in equidistant_grid[:bf.ng]])
            bf.phit_g[-1] = 0.

        basis = Basis(g.symbol, self.name, False)
        basis.ng = ng
        basis.d = d
        basis.bf_j = bf_j
        basis.generatordata = textbuffer.getvalue().strip()
        basis.generatorattrs = {'version' : version}
        textbuffer.close()

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
            phit_g = npy.zeros_like(r)
            phit_g[:len(bf.phit_g)] = bf.phit_g
            pylab.plot(r, phit_g * r, label=label[:12])
        axis = pylab.axis()
        newaxis = [0., rc, axis[2], axis[3]]
        pylab.axis(newaxis)
        pylab.legend()
        if filename is not None:
            pylab.savefig(filename)

