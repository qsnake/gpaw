import os
import sys
import time
from math import log

import Numeric as num
from ASE.ChemicalElements.symbol import symbols

from gpaw.utilities import devnull
from gpaw.mpi import MASTER
from gpaw.version import version
import gpaw


class Output:
    """Class for handling all text output."""

    def __init__(self):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        ``None``:
          Throw output away.
        ``'-'``:
          Use standard-output (``sys.stdout``).
        A filename:
          open a new file.
        """

        p = self.input_parameters
        txt = p['txt']
        if txt is None or (not self.master):
            txt = devnull
        elif txt == '-':
            txt = sys.stdout
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.txt = txt
        self.verbose = p['verbose']

        self.print_logo()

    def text(self, *args, **kwargs):
        self.txt.write(kwargs.get('sep', ' ').join([str(arg)
                                                    for arg in args]) +
                       kwargs.get('end', '\n'))
        
    def print_logo(self):
        self.text()
        self.text('  ___ ___ ___ _ _ _  ')
        self.text(' |   |   |_  | | | | ')
        self.text(' | | | | | . | | | | ')
        self.text(' |__ |  _|___|_____| ', version)
        self.text(' |___|_|             ')
        self.text()

        uname = os.uname()
        self.text('User:', os.getenv('USER') + '@' + uname[1])
        self.text('Date:', time.asctime())
        self.text('Arch:', uname[4])
        self.text('Pid: ', os.getpid())
        self.text('Dir: ', os.path.dirname(gpaw.__file__))
                  
    def print_init(self, pos_ac):
        t = self.text
        p = self.input_parameters

        self.print_parameters()
        
        t()
        t('unitcell:')
        t('         periodic  length  points   spacing')
        t('  -----------------------------------------')
        for c in range(3):
            t('  %s-axis   %s   %8.4f   %3d    %8.4f' % 
              ('xyz'[c],
               ['no ', 'yes'][self.domain.periodic_c[c]],
               self.a0 * self.domain.cell_c[c],
               self.gd.N_c[c],
               self.a0 * self.gd.h_c[c]))
        t()

        t('Positions:')
        for a, pos_c in enumerate(pos_ac):
            symbol = self.nuclei[a].setup.symbol
            t('%3d %2s %8.4f%8.4f%8.4f' % 
              ((a, symbol) + tuple(self.a0 * pos_c)))

    def print_parameters(self):
        t = self.text
        p = self.input_parameters
        
        if self.spinpol:
            t('Spin-polarized calculation.')
            t('Magnetic moment: %.6f' % sum(self.density.magmom_a), end='')
            if self.fixmom:
                t('(fixed)')
            else:
                t()
        else:
            t('Spin-paired calculation')

        t('Total charge: %.6f' % p['charge'])
        t('Fermi temperature: %.6f' % (self.kT * self.Ha))
        t('Eigensolver: %s (%s)' % (p['eigensolver'], fd(p['stencils'][0])))
        t('Poisson solver: %s (%s)' % (p['poissonsolver'],
                                       fd(p['stencils'][1])))
        t('Interpolation: %d. order' % (2 * p['stencils'][2]))
          
        if self.gamma:
            t('Gamma-point calculation')
        t('Reference energy:', self.Eref * self.Ha)

        if self.kpt_comm.size > 1:
            if self.nspins == 2 and self.nkpts == 1:
                t('Parallelization over spin')
            elif self.nspins == 2:
                t('Parallelization over k-points and spin on %d processors' %
                  self.kpt_comm.size)
            else:
                t('Parallelization over k-points on %d processors' %
                  self.kpt_comm.size)

        domain = self.domain
        if domain.comm.size > 1:
            t('Using domain decomposition: %d x %d x %d' %
              tuple(domain.parsize_c))

        if self.symmetry is not None:
            self.symmetry.print_symmetries(t)
        
        t(('%d k-point%s in the irreducible part of the ' +
           'Brillouin zone (total: %d)') %
          (self.nkpts, ' s'[1:self.nkpts], len(self.bzk_kc)))

        if self.fixdensity > self.maxiter:
            t('Fixing the initial density')
        else:
            mixer = self.density.mixer
            t('Linear mixing parameter: %.6f' % mixer.beta)
            t('Pulay mixing with %d old densities' % mixer.nmaxold)
            t('Damping of long wave oscillations: %.6f' % mixer.x)

        cc = p['convergence']
        t('Convergence criteria:')
        t('* Energy: %f / atom' % (cc['energy'] * self.Ha))
        t('* Density: %f electrons' % cc['density'])
        t('* Eigenstates: %f' % cc['eigenstates'])
        if cc['bands'] == 'occupied':
            t('Converge occupied states only.')
        else:
            t('Converge %d bands.' % cc['bands'])

    def print_converged(self):
        t = self.text
        t('------------------------------------')
        t('Converged after %d iterations.' % self.niter)

        t()
        self.print_all_information()

    def print_all_information(self):
        t = self.text    
        if len(self.nuclei) == 1:
            t('energy contributions relative to reference atom:', end='')
        else:
            t('energy contributions relative to reference atoms:', end='')
        t('(reference = %.5f)' % (self.Eref * self.Ha))

        t('-------------------------')

        energies = [('kinetic:', self.Ekin),
                    ('potential:', self.Epot),
                    ('external:', self.Eext),
                    ('XC:', self.Exc),
                    ('entropy (-ST):', -self.S),
                    ('local:', self.Ebar)]

        for name, e in energies:
            t('%-14s %+10.5f' % (name, self.Ha * e))

        t('-------------------------')
        t('free energy:   %+10.5f' % (self.Ha * self.Etot))
        t('zero Kelvin:   %+10.5f' % (self.Ha * (self.Etot + 0.5 * self.S)))
        t()
        epsF = self.occupation.get_fermi_level()
        if epsF is not None:
            t('Fermi level:', self.Ha * epsF)

        self.print_eigenvalues()

        t()
        charge = self.finegd.integrate(self.density.rhot_g)
        t('total charge: %f electrons' % charge)

        dipole = self.finegd.calculate_dipole_moment(self.density.rhot_g)
        if self.density.charge == 0:
            t('dipole moment: %s' % (dipole * self.a0))
        else:
            t('center of charge: %s' % (dipole * self.a0))

        if self.nspins == 2:
            self.density.calculate_local_magnetic_moments()

            t()
            t('total magnetic moment: %f' % self.occupation.magmom)
            t('local magnetic moments:')
            for nucleus in self.nuclei:
                t(nucleus.a, nucleus.mom)
            t()


    def print_iteration(self):
        # Output from each iteration:
        t = self.text    

        if self.verbose != 0:
            T = time.localtime()
            t()
            t('------------------------------------')
            t('iter: %d %d:%02d:%02d' % (self.niter, T[3], T[4], T[5]))
            t()
            t('Poisson solver converged in %d iterations' %
                      self.hamiltonian.npoisson)
            t('Fermi level found  in %d iterations' % self.occupation.niter)
            t('Error in wave functions: %.13f' % self.error['eigenstates'])
            t()
            self.print_all_information()

        else:        
            if self.niter == 0:
                header = """\
                     log10-error:    total        iterations:
           time      wfs    density  energy       fermi  poisson"""
                if self.spinpol:
                    header += '  magmom'
                t(header)

            T = time.localtime()

            if self.error['eigenstates'] == 0.0:
                eigerror = ''
            else:
                eigerror = '%-+5.1f' % (log(self.error['eigenstates']) /
                                        log(10))
                
            dNt = self.density.mixer.get_charge_sloshing()
            if dNt is None or self.nvalence == 0:
                dNt = ''
            else:
                dNt = '%+.1f' % (log(dNt / self.nvalence) / log(10))

            niterocc = self.occupation.niter
            if niterocc == -1:
                niterocc = ''
            else:
                niterocc = '%d' % niterocc

            niterpoisson = '%d' % self.hamiltonian.npoisson
            
            t("""\
iter: %3d  %02d:%02d:%02d  %-5s  %-5s    %-12.5f %-5s  %-7s""" %
              (self.niter,
               T[3], T[4], T[5],
               eigerror,
               dNt,
               self.Ha * (self.Etot + 0.5 * self.S),
               niterocc,
               niterpoisson), end='')
            
            if self.spinpol:
                t('  %+.4f' % self.occupation.magmom)
            else:
                t()

        self.txt.flush()

    def print_forces(self):
        c = self.Ha / self.a0
        for a, nucleus in enumerate(self.nuclei):
            self.text('forces ', a, nucleus.setup.symbol, self.F_ac[a] * c)

    def print_eigenvalues(self):
        """Print eigenvalues and occupation numbers."""
        print >> self.txt, eigenvalue_string(self)

    def plot_atoms(self):
        atoms = self.atoms
        cell_c = num.diagonal(atoms.GetUnitCell()) / self.a0
        pos_ac = atoms.GetCartesianPositions() / self.a0
        Z_a = atoms.GetAtomicNumbers()
        pbc_c = atoms.GetBoundaryConditions()
        self.text(plot(pos_ac, Z_a, cell_c))

def eigenvalue_string(paw,comment=None):
    """
    Write eigenvalues and occupation numbers into a string.
    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """

    if not comment: comment=''

    Ha = paw.Ha

    if paw.nkpts > 1 or paw.kpt_comm.size > 1:
        # not implemented yet:
        return ''

    s = ''
    if paw.nspins == 1:
        s += comment + ' band     eps        occ\n'
        kpt = paw.kpt_u[0]
        for n in range(paw.nbands):
            s += ('%4d %10.5f %10.5f\n' %
                  (n, Ha * kpt.eps_n[n], kpt.f_n[n]))
    else:
        s += comment + '                up                   down\n'
        s += comment + ' band     eps        occ        eps        occ\n'
        epsa_n = paw.kpt_u[0].eps_n
        epsb_n = paw.kpt_u[1].eps_n
        fa_n = paw.kpt_u[0].f_n
        fb_n = paw.kpt_u[1].f_n
        for n in range(paw.nbands):
            s += ('%4d %10.5f %10.5f %10.5f %10.5f\n' %
                  (n,
                   Ha * epsa_n[n], fa_n[n],
                   Ha * epsb_n[n], fb_n[n]))
    return s

def plot(positions, numbers, cell):
    """Ascii-art plot of the atoms.

    Example::

      from ASE import ListOfAtoms, Atom
      a = 4.0
      n = 20
      d = 1.0
      x = d / 3**0.5
      atoms = ListOfAtoms([Atom('C', (0.0, 0.0, 0.0)),
                           Atom('H', (x, x, x)),
                           Atom('H', (-x, -x, x)),
                           Atom('H', (x, -x, -x)),
                           Atom('H', (-x, x, -x))],
                          cell=(a, a, a), periodic=True)
      for line in plot(2*atoms.GetCartesianPositions() + (a,a,a),
                       atoms.GetAtomicNumbers(),
                       2*num.array(atoms.GetUnitCell().flat[::4])):
          print line

          .-----------.
         /|           |
        / |           |
       *  |      H    |
       |  | H  C      |
       |  |  H        |
       |  .-----H-----.
       | /           /
       |/           /
       *-----------*
    """

    s = 1.3
    nx, ny, nz = n = (s * cell * (1.0, 0.25, 0.5) + 0.5).astype(num.Int)
    sx, sy, sz = n / cell
    grid = Grid(nx + ny + 4, nz + ny + 1)
    positions = (positions % cell + cell) % cell
    ij = num.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = num.around(ij).astype(num.Int)
    for a, Z in enumerate(numbers):
        symbol = symbols[Z]
        i, j = ij[a]
        depth = positions[a, 1]
        for n, c in enumerate(symbol):
            grid.put(c, i + n + 1, j, depth)
    k = 0
    for i, j in [(1, 0), (1 + nx, 0)]:
        grid.put('*', i, j) 
        grid.put('.', i + ny, j + ny)
        if k == 0:
            grid.put('*', i, j + nz)
        grid.put('.', i + ny, j + nz + ny)
        for y in range(1, ny):
            grid.put('/', i + y, j + y, y / sy) 
            if k == 0:
                grid.put('/', i + y, j + y + nz, y / sy) 
        for z in range(1, nz):
            if k == 0:
                grid.put('|', i, j + z)
            grid.put('|', i + ny, j + z + ny)
        k = 1
    for i, j in [(1, 0), (1, nz)]:
        for x in range(1, nx):
            if k == 1:
                grid.put('-', i + x, j)
            grid.put('-', i + x + ny, j + ny)
        k = 0
    return '\n'.join([''.join([chr(x) for x in line])
                      for line in num.transpose(grid.grid)[::-1]])

class Grid:
    def __init__(self, i, j):
        self.grid = num.zeros((i, j), num.Int8)
        self.grid[:] = ord(' ')
        self.depth = num.zeros((i, j), num.Float)
        self.depth[:] = 1e10

    def put(self, c, i, j, depth=1e9):
        if depth < self.depth[i, j]:
            self.grid[i, j] = ord(c)
            self.depth[i, j] = depth


def fd(n):
    if n == 'M':
        return 'Mehrstellen finite-difference stencil'
    if n == 1:
        return 'Nearest neighbor central finite-difference stencil'
    return '%d nearest neighbors central finite-difference stencil' % n
