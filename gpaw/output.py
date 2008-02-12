import os
import sys
import time
from math import log

import numpy as npy
import ase
from ase.data import chemical_symbols

from gpaw.utilities import devnull
from gpaw.mpi import MASTER
from gpaw.version import version
import gpaw


class Output:
    """Class for handling all text output."""

    def __init__(self):
        self.txt = None

    def set_text(self, txt, verbose=True):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        * None:  Throw output away.
        * '-':  Use standard-output (``sys.stdout``).
        * A filename:  Open a new file.
        """

        self.verbose = verbose

        firsttime = (self.txt is None)
        
        if txt is None or (not self.master):
            self.txt = devnull
        elif txt == '-':
            self.txt = sys.stdout
        elif isinstance(txt, str):
            if isinstance(self.txt, file) and self.txt.name == txt:
                pass
            else:
                if not firsttime:
                    # We want every file to start with the logo, so
                    # that the ase.io.read() function will recognize
                    # it as a GPAW text file.
                    firsttime = True 
                self.txt = open(txt, 'w')
        if firsttime:
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
        self.text('User:', os.getenv('USER', '???') + '@' + uname[1])
        self.text('Date:', time.asctime())
        self.text('Arch:', uname[4])
        self.text('Pid: ', os.getpid())
        self.text('Dir: ', os.path.dirname(gpaw.__file__))
        self.text('ase:  ', os.path.dirname(ase.__file__))
        self.text('numpy:', os.path.dirname(npy.__file__))
        self.text('units: Angstrom and eV')

                  
    def print_init(self, pos_ac):
        t = self.text
        p = self.input_parameters

##        self.print_parameters()
        
        t()
        t('Unit Cell:')
        t('         Periodic  Length  Points   Spacing')
        t('  -----------------------------------------')
        for c in range(3):
            t('  %s-axis   %s   %8.4f   %3d    %8.4f' % 
              ('xyz'[c],
               ['no ', 'yes'][int(self.domain.pbc_c[c])],
               self.a0 * self.domain.cell_c[c],
               self.gd.N_c[c],
               self.a0 * self.gd.h_c[c]))
        t()

    def print_positions(self, pos_ac):
        t = self.text
        t()
        t('Positions:')
        for a, pos_c in enumerate(pos_ac):
            symbol = self.nuclei[a].setup.symbol
            t('%3d %-2s %8.4f%8.4f%8.4f' % 
              ((a, symbol) + tuple(self.a0 * pos_c)))
        t()

    def print_parameters(self):
        t = self.text
        p = self.input_parameters
        
        t('Using the %s Exchange-Correlation Functional.' % self.xcfunc.xcname)
        if self.spinpol:
            t('Spin-Polarized Calculation.')
            t('Magnetic Moment:   %.6f' % sum(self.density.magmom_a), end='')
            if self.fixmom:
                t('(fixed)')
            else:
                t()
        else:
            t('Spin-Paired Calculation')
        
        t('Total Charge:      %.6f' % p['charge'])
        t('Fermi Temperature: %.6f' % (self.kT * self.Ha))
        t('Eigen Solver:      %s \n                   (%s)' %
          (p['eigensolver'], fd(p['stencils'][0])))
        t('Poisson Solver:    %s \n                   (%s)' %
          ([0, 'GaussSeidel', 'Jacobi'][self.hamiltonian.poisson.relax_method],
           fd(self.hamiltonian.poisson.nn)))
        order = str((2 * p['stencils'][1]))
        if order[-1] == '1':
            order = order+'st'
        elif order[-1] == '2':
            order = order+'nd'
        elif order[-1] == '3':
            order = order+'rd'
        else:
            order = order+'th'
        
        t('Interpolation:     '+order+' Order')
        t('Reference Energy:  %.6f' % (self.Eref * self.Ha))
        t()          
        if self.gamma:
            t('Gamma Point Calculation')

        if self.kpt_comm.size > 1:
            if self.nspins == 2 and self.nkpts == 1:
                t('Parallelization Over Spin')
            elif self.nspins == 2:
                t('Parallelization Over k-points and Spin on %d Processors' %
                  self.kpt_comm.size)
            else:
                t('Parallelization Over k-points on %d Processors' %
                  self.kpt_comm.size)

        domain = self.domain
        if domain.comm.size > 1:
            t('Using Domain Decomposition: %d x %d x %d' %
              tuple(domain.parsize_c))

        if self.symmetry is not None:
            self.symmetry.print_symmetries(t)
        
        t(('%d k-point%s in the Irreducible Part of the ' +
           'Brillouin Zone (total: %d)') %
          (self.nkpts, ' s'[1:self.nkpts], len(self.bzk_kc)))

        if self.fixdensity > self.maxiter:
            t('Fixing the initial density')
        else:
            mixer = self.density.mixer
            t('Linear Mixing Parameter:           %g' % mixer.beta)
            t('Pulay Mixing with %d Old Densities' % mixer.nmaxold)
            t('Damping of Long Wave Oscillations: %g' % mixer.weight)

        cc = p['convergence']
        t()
        t('Convergence Criteria:')
        t('Total Energy Change per Atom:           %g eV / atom' %
          (cc['energy']))
        t('Integral of Absolute Density Change:    %g electrons' %
          cc['density'])
        t('Integral of Absolute Eigenstate Change: %g' % cc['eigenstates'])
        t('Number of Bands in Calculation:         %i' % self.nbands)
        t('Bands to Converge:                      ', end='')
        if cc['bands'] == 'occupied':
            t('Occupied States Only')
        else:
            t('%d Lowest Bands' % cc['bands'])
        t('Number of Valence Electrons:            %i' % self.nvalence)

    def print_converged(self):
        t = self.text
        t('------------------------------------')
        t('Converged After %d Iterations.' % self.niter)

        t()
        self.print_all_information()

    def print_all_information(self):
        t = self.text    
        if len(self.nuclei) == 1:
            t('Energy Contributions Relative to Reference Atom:', end='')
        else:
            t('Energy Contributions Relative to Reference Atoms:', end='')
        t('(reference = %.5f)' % (self.Eref * self.Ha))

        t('-------------------------')

        energies = [('Kinetic:      ', self.Ekin),
                    ('Potential:    ', self.Epot),
                    ('External:     ', self.Eext),
                    ('XC:           ', self.Exc),
                    ('Entropy (-ST):', -self.S),
                    ('Local:        ', self.Ebar)]

        for name, e in energies:
            t('%-14s %+10.5f' % (name, self.Ha * e))

        t('-------------------------')
        t('Free Energy:   %+10.5f' % (self.Ha * self.Etot))
        t('Zero Kelvin:   %+10.5f' % (self.Ha * (self.Etot + 0.5 * self.S)))
        t()
        epsF = self.occupation.get_fermi_level()
        if epsF is not None:
            t('Fermi Level:', self.Ha * epsF)

        self.print_eigenvalues()

        t()
        charge = self.finegd.integrate(self.density.rhot_g)
        t('Total Charge:  %f electrons' % charge)

        dipole = self.finegd.calculate_dipole_moment(self.density.rhot_g)
        if self.density.charge == 0:
            t('Dipole Moment: %s' % (dipole * self.a0))
        else:
            t('Center of Charge: %s' % (dipole * self.a0))

        if self.nspins == 2:
            self.density.calculate_local_magnetic_moments()

            t()
            t('Total Magnetic Moment: %f' % self.occupation.magmom)
            t('Local Magnetic Moments:')
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
            t('Poisson Solver Converged in %d Iterations' %
                      self.hamiltonian.npoisson)
            t('Fermi Level Found  in %d Iterations' % self.occupation.niter)
            t('Error in Wave Functions: %.13f' % self.error['eigenstates'])
            t()
            self.print_all_information()

        else:        
            if self.niter == 0:
                header = """\
                     log10-error:    Total        Iterations:
           Time      WFS    Density  Energy       Fermi  Poisson"""
                if self.spinpol:
                    header += '  MagMom'
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
iter: %3d  %02d:%02d:%02d  %-5s  %-5s    %- 12.5f %-5s  %-7s""" %
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
        t = self.text
        t()
        t('Forces in eV/Ang:')
        c = self.Ha / self.a0        
        for a, nucleus in enumerate(self.nuclei):
            symbol = nucleus.setup.symbol
            t('%3d %-2s %8.4f%8.4f%8.4f' % 
              ((a, symbol) + tuple(self.F_ac[a] * c)))

    def print_eigenvalues(self):
        """Print eigenvalues and occupation numbers."""
        print >> self.txt, eigenvalue_string(self)

    def plot_atoms(self, atoms):
        cell_c = npy.diagonal(atoms.get_cell()) / self.a0
        pos_ac = atoms.get_positions() / self.a0
        Z_a = atoms.get_atomic_numbers()
        pbc_c = atoms.get_pbc()
        self.text(plot(pos_ac, Z_a, cell_c))

def eigenvalue_string(paw, comment=None):
    """
    Write eigenvalues and occupation numbers into a string.
    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """

    if not comment: comment=' '

    Ha = paw.Ha

    if paw.nkpts > 1:
        # not implemented yet:
        return ''

    s = ''
    if paw.nspins == 1:
        s += comment + 'Band   Eigenvalues  Occupancy\n'        
        eps_n = paw.collect_eigenvalues(k=0, s=0)
        f_n   = paw.collect_occupations(k=0, s=0)
        for n in range(paw.nbands):
            s += ('%4d   %10.5f  %10.5f\n' %
                  (n, Ha * eps_n[n], f_n[n]))
    else:
        s += comment + '                 Up                     Down\n'
        s += comment + 'Band  Eigenvalues  Occupancy  Eigenvalues  Occupancy\n'
        epsa_n = paw.collect_eigenvalues(k=0, s=0)
        epsb_n = paw.collect_eigenvalues(k=0, s=1)
        fa_n   = paw.collect_occupations(k=0, s=0)
        fb_n   = paw.collect_occupations(k=0, s=1)
        if paw.master:
            for n in range(paw.nbands):
                s += (' %4d  %11.5f  %9.5f  %11.5f  %9.5f\n' %
                      (n, Ha * epsa_n[n], fa_n[n], Ha * epsb_n[n], fb_n[n]))
    return s

def plot(positions, numbers, cell):
    """Ascii-art plot of the atoms.

    Example::

      from ase import *
      a = 4.0
      n = 20
      d = 1.0
      x = d / 3**0.5
      atoms = ListOfAtoms(symbols='CH4',
                          positions=[(0, 0, 0),
                                     (x, x, x),
                                     (-x, -x, x),
                                     (x, -x, -x),
                                     (-x, x, -x)],
                          cell=(a, a, a), pbc=True)
      for line in plot(2*atoms.GetCartesianPositions() + (a,a,a),
                       atoms.GetAtomicNumbers(),
                       2*npy.array(atoms.GetUnitCell().ravel()[::4])):
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
##   y
##   |
##   .-- x
##  /
## z

    s = 1.3
    nx, ny, nz = n = (s * cell * (1.0, 0.25, 0.5) + 0.5).astype(int)
    sx, sy, sz = n / cell
    grid = Grid(nx + ny + 4, nz + ny + 1)
    positions = (positions % cell + cell) % cell
    ij = npy.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = npy.around(ij).astype(int)
    for a, Z in enumerate(numbers):
        symbol = chemical_symbols[Z]
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
                      for line in npy.transpose(grid.grid)[::-1]])

class Grid:
    def __init__(self, i, j):
        self.grid = npy.zeros((i, j), npy.int8)
        self.grid[:] = ord(' ')
        self.depth = npy.zeros((i, j))
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
