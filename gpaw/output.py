import os
import sys
import time
from math import log
from math import sqrt

import numpy as npy
import ase
from ase.version import version as ase_version
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

from gpaw.utilities import devnull
from gpaw.mpi import size, parallel
from gpaw.version import version
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize, sl_inverse_cholesky, dry_run, extra_parameters
from gpaw.utilities.memory import maxrss
import gpaw

def initialize_text_stream(txt, rank, old_txt=None):
    """Set the stream for text output.
    
    If `txt` is not a stream-object, then it must be one of:
    
    * None:  Throw output away.
    * '-':  Use standard-output (``sys.stdout``).
    * A filename:  Open a new file.
    """
    firsttime = (old_txt is None)

    if txt is None or rank != 0:
        return devnull, firsttime
    elif txt == '-':
        return sys.stdout, firsttime
    elif isinstance(txt, str):
        if isinstance(old_txt, file) and old_txt.name == txt:
            return old_txt, firsttime
        else:
            if not firsttime:
                # We want every file to start with the logo, so
                # that the ase.io.read() function will recognize
                # it as a GPAW text file.
                firsttime = True
            return open(txt, 'w'), firsttime
    else:
        assert hasattr(txt, 'write'), 'Not a stream object!'
        return txt, firsttime

    return old_txt, firsttime

class PAWTextOutput:
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

        self.txt, firsttime = initialize_text_stream(txt, self.wfs.world.rank,
                                                     self.txt)
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
        self.text('ase:  ', os.path.dirname(ase.__file__),
                  ' version: ', ase_version)
        self.text('numpy:', os.path.dirname(npy.__file__))
        self.text('units: Angstrom and eV')

        if extra_parameters:
            self.text('Extra parameters:', extra_parameters)

    def print_cell_and_parameters(self):
        self.plot_atoms(self.atoms)
        self.print_unit_cell(self.atoms.get_positions() / Bohr)
        self.print_parameters()

    def print_unit_cell(self, pos_ac):
        t = self.text
        p = self.input_parameters

        t()
        t('Unit Cell:')
        t('         Periodic  Length  Points   Spacing')
        t('  -----------------------------------------')
        for c in range(3):
            t('  %s-axis   %s   %8.4f   %3d    %8.4f' %
              ('xyz'[c],
               ['no ', 'yes'][int(self.gd.pbc_c[c])],
               Bohr * self.gd.cell_c[c],
               self.gd.N_c[c],
               Bohr * self.gd.h_c[c]))
        t()

    def print_positions(self):
        t = self.text
        t()
        t('Positions:')
        symbols = self.atoms.get_chemical_symbols()
        for a, pos_c in enumerate(self.atoms.get_positions()):
            symbol = symbols[a]
            t('%3d %-2s %9.4f %9.4f %9.4f' % ((a, symbol) + tuple(pos_c)))
        t()

    def print_parameters(self):
        t = self.text
        p = self.input_parameters

        t('Using the %s Exchange-Correlation Functional.'
          % self.hamiltonian.xcfunc.xcname)
        if self.wfs.nspins == 2:
            t('Spin-Polarized Calculation.')
            t('Magnetic Moment:   %.6f' % self.density.magmom_a.sum(), end='')
            if p.fixmom:
                t('(fixed)')
            else:
                t()
        else:
            t('Spin-Paired Calculation')

        t('Total Charge:      %.6f' % p['charge'])
        t('Fermi Temperature: %.6f' % (self.occupations.kT * Hartree))
        t('Mode:              %s' % p['mode'])
        eigensolver = p['eigensolver']
        if eigensolver is None:
            eigensolver = {'lcao':'lcao (direct)', 'fd':'rmm-diis'}[p['mode']]
        t('Eigen Solver:      %s' % eigensolver)
        if p['mode'] != 'lcao':
            t('                   (%s)' % fd(p['stencils'][0]))
        diag_string = 'Lapack'
        if sl_diagonalize: assert parallel
        if sl_diagonalize: assert scalapack()
        if scalapack() and sl_diagonalize:
            assert len(sl_diagonalize) == 4
            # set ScaLapack defaults
            # cpus_per_node used only to make the topology of the grid
            if sl_diagonalize[3] is 'd':
                cpus_per_node = 4
            else:
                cpus_per_node = int(sl_diagonalize[3])
            assert cpus_per_node > 0
            npcol = max(1, cpus_per_node *
                        (int(sqrt(size) / cpus_per_node + 0.5)))
            nprow = min(max(1, int(size / npcol)), npcol)
            npcol = max(max(1, int(size / npcol)), npcol)
            sl_defaults = [nprow, npcol, 32, cpus_per_node]
            for sl_args_index in range(len(sl_diagonalize)):
                if sl_diagonalize[sl_args_index] is 'd':
                    sl_diagonalize[sl_args_index] = sl_defaults[sl_args_index]
            assert sl_diagonalize[0] * sl_diagonalize[1] <= size
            diag_string = ('ScaLapack - grid: [nprow, npcol, nb] = %s'
                           % sl_diagonalize[:3])
        t('Diagonalizer:      %s' % diag_string)
        inverse_cholesky_string = 'Lapack'
        if sl_inverse_cholesky: assert parallel
        if sl_inverse_cholesky: assert scalapack()
        if scalapack() and sl_inverse_cholesky:
            assert len(sl_inverse_cholesky) == 4
            # set ScaLapack defaults
            # cpus_per_node used only to make the topology of the grid
            if sl_inverse_cholesky[3] is 'd':
                cpus_per_node = 4
            else:
                cpus_per_node = int(sl_inverse_cholesky[3])
            assert cpus_per_node > 0
            npcol = max(1, cpus_per_node *
                        (int(sqrt(size) / cpus_per_node + 0.5)))
            nprow = min(max(1, int(size / npcol)), npcol)
            npcol = max(max(1, int(size / npcol)), npcol)
            sl_defaults = [nprow, npcol, 32, cpus_per_node]
            for sl_args_index in range(len(sl_inverse_cholesky)):
                if sl_inverse_cholesky[sl_args_index] is 'd':
                    sl_inverse_cholesky[sl_args_index] = sl_defaults[sl_args_index]
            assert sl_inverse_cholesky[0]*sl_inverse_cholesky[1] <= size
            inverse_cholesky_string = ('ScaLapack' +
                                       ' - grid: [nprow, npcol, nb] = %s'
                                       % sl_inverse_cholesky[:3])
        t('Inverse Cholesky:  %s' % inverse_cholesky_string)
        t('Poisson Solver:    %s \n                   (%s)' %
          ([0, 'GaussSeidel', 'Jacobi'][self.hamiltonian.poisson.relax_method],
           fd(self.hamiltonian.poisson.nn)))
        order = str((2 * p['stencils'][1]))
        if order[-1] == '1':
            order = order + 'st'
        elif order[-1] == '2':
            order = order + 'nd'
        elif order[-1] == '3':
            order = order + 'rd'
        else:
            order = order + 'th'

        t('Interpolation:     ' + order + ' Order')
        t('Reference Energy:  %.6f' % (self.wfs.setups.Eref * Hartree))
        t()
        if self.wfs.gamma:
            t('Gamma Point Calculation')

        nibzkpts = self.wfs.nibzkpts

        if self.wfs.kpt_comm.size > 1:
            if self.wfs.nspins == 2 and nibzkpts == 1:
                t('Parallelization Over Spin')
            elif self.wfs.nspins == 2:
                t('Parallelization Over k-points and Spin on %d Processors' %
                  self.wfs.kpt_comm.size)
            else:
                t('Parallelization Over k-points on %d Processors' %
                  self.wfs.kpt_comm.size)

        gd = self.gd
        if gd.comm.size > 1:
            t('Using Domain Decomposition: %d x %d x %d' %
              tuple(gd.parsize_c))

        if self.wfs.symmetry is not None:
            self.wfs.symmetry.print_symmetries(t)

        t(('%d k-point%s in the Irreducible Part of the ' +
           'Brillouin Zone (total: %d)') %
          (nibzkpts, ' s'[1:nibzkpts], len(self.wfs.bzk_kc)))

        if self.scf.fixdensity > self.scf.maxiter:
            t('Fixing the initial density')
        else:
            mixer = self.density.mixer
            t('Linear Mixing Parameter:           %g' % mixer.beta)
            t('Pulay Mixing with %d Old Densities' % mixer.nmaxold)
            if mixer.metric_type is None:
                t('No Damping of Long Wave Oscillations')
            else:
                t('Damping of Long Wave Oscillations: %g' % mixer.weight)

        cc = p['convergence']
        t()
        t('Convergence Criteria:')
        t('Total Energy Change per Atom:           %g eV / atom' %
          (cc['energy']))
        t('Integral of Absolute Density Change:    %g electrons' %
          cc['density'])
        t('Integral of Absolute Eigenstate Change: %g' % cc['eigenstates'])
        t('Number of Bands in Calculation:         %i' % self.wfs.nbands)
        t('Bands to Converge:                      ', end='')
        if cc['bands'] == 'occupied':
            t('Occupied States Only')
        elif cc['bands'] == 'all':
            t('All')
        else:
            t('%d Lowest Bands' % cc['bands'])
        t('Number of Valence Electrons:            %i'
          % self.wfs.setups.nvalence)

    def print_converged(self, iter):
        t = self.text
        t('------------------------------------')
        t('Converged After %d Iterations.' % iter)

        t()
        self.print_all_information()

    def print_all_information(self):
        t = self.text
        if len(self.atoms) == 1:
            t('Energy Contributions Relative to Reference Atom:', end='')
        else:
            t('Energy Contributions Relative to Reference Atoms:', end='')
        t('(reference = %.5f)' % (self.wfs.setups.Eref * Hartree))

        t('-------------------------')

        energies = [('Kinetic:      ',  self.hamiltonian.Ekin),
                    ('Potential:    ',  self.hamiltonian.Epot),
                    ('External:     ',  self.hamiltonian.Eext),
                    ('XC:           ',  self.hamiltonian.Exc),
                    ('Entropy (-ST):', -self.hamiltonian.S),
                    ('Local:        ',  self.hamiltonian.Ebar)]

        for name, e in energies:
            t('%-14s %+10.5f' % (name, Hartree * e))

        t('-------------------------')
        t('Free Energy:   %+10.5f' % (Hartree * self.hamiltonian.Etot))
        t('Zero Kelvin:   %+10.5f' % (Hartree * (self.hamiltonian.Etot +
                                                 0.5 * self.hamiltonian.S)))
        t()
        try:
            epsF = self.occupations.get_fermi_level()
        except NotImplementedError:
            pass
        else:
            t('Fermi Level:', Hartree * epsF)

        self.print_eigenvalues()
        self.print_forces()

        if self.density.rhot_g is None:
            return
        
        t()
        charge = self.finegd.integrate(self.density.rhot_g)
        t('Total Charge:  %f electrons' % charge)

        dipole = self.get_dipole_moment()
        if self.density.charge == 0:
            t('Dipole Moment: %s' % dipole)
        else:
            t('Center of Charge: %s' % (dipole / abs(charge)))

        if self.wfs.nspins == 2:
            t()
            t('Total Magnetic Moment: %f' % self.occupations.magmom)
            t('Local Magnetic Moments:')
            for a, mom in enumerate(self.get_magnetic_moments()):
                t(a, mom)
            t()

##         if self.xcfunc.is_gllb():
##             self.xcfunc.xc.print_converged(self)

    def print_iteration(self, iter):
        # Output from each iteration:
        t = self.text

        nvalence = self.wfs.setups.nvalence
        eigerr = self.scf.eigenstates_error / nvalence
        if self.verbose != 0:
            T = time.localtime()
            t()
            t('------------------------------------')
            t('iter: %d %d:%02d:%02d' % (iter, T[3], T[4], T[5]))
            t()
            t('Poisson Solver Converged in %d Iterations' %
              self.hamiltonian.npoisson)
            t('Fermi Level Found  in %d Iterations' % self.occupations.niter)
            t('Error in Wave Functions: %.13f' % eigerr)              
            t()
            self.print_all_information()

        else:
            if iter == 1:
                header = """\
                     log10-error:    Total        Iterations:
           Time      WFS    Density  Energy       Fermi  Poisson"""
                if self.wfs.nspins == 2:
                    header += '  MagMom'
                t(header)

            T = time.localtime()

            if eigerr == 0.0:
                eigerr = ''
            else:
                eigerr = '%-+5.1f' % (log(eigerr) / log(10))

            denserr = self.density.mixer.get_charge_sloshing()
            if denserr is None or denserr == 0 or nvalence == 0:
                denserr = ''
            else:
                denserr = '%+.1f' % (log(denserr / nvalence) / log(10))

            niterocc = self.occupations.niter
            if niterocc == -1:
                niterocc = ''
            else:
                niterocc = '%d' % niterocc

            niterpoisson = '%d' % self.hamiltonian.npoisson

            t("iter: %3d  %02d:%02d:%02d  %-5s  %-5s    %- 12.5f %-5s  %-7s" %
              (iter,
               T[3], T[4], T[5],
               eigerr,
               denserr,
               Hartree * (self.hamiltonian.Etot + 0.5 * self.hamiltonian.S),
               niterocc,
               niterpoisson), end='')

            if self.wfs.nspins == 2:
                t('  %+.4f' % self.occupations.magmom)
            else:
                t()

        self.txt.flush()

    def print_forces(self):
        if self.forces.F_av is None:
            return
        t = self.text
        t()
        t('Forces in eV/Ang:')
        c = Hartree / Bohr
        symbols = self.atoms.get_chemical_symbols()
        for a, symbol in enumerate(symbols):
            t('%3d %-2s %10.5f %10.5f %10.5f' %
              ((a, symbol) + tuple(self.forces.F_av[a] * c)))

    def print_eigenvalues(self):
        """Print eigenvalues and occupation numbers."""
        print >> self.txt, eigenvalue_string(self)

    def plot_atoms(self, atoms):
        cell_cv = atoms.get_cell()
        if (cell_cv - npy.diag(cell_cv.diagonal())).any():
            return
        
        cell_c = npy.diagonal(cell_cv) / Bohr
        pos_ac = atoms.get_positions() / Bohr
        Z_a = atoms.get_atomic_numbers()
        pbc_c = atoms.get_pbc()
        self.text(plot(pos_ac, Z_a, cell_c))

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if not hasattr(self, 'txt') or self.txt is None:
            return
        
        if not dry_run:
            mr = maxrss()
            if mr > 0:
                if mr < 1024.0**3:
                    self.text('Memory usage: %.2f MB' % (mr / 1024.0**2))
                else:
                    self.text('Memory usage: %.2f GB' % (mr / 1024.0**3))

            if hasattr(self, 'timer'):
                self.timer.write(self.txt)

    def warn(self, string=None):
        if not string:
            string = "somethings wrong"
        print >> self.txt, "WARNING >>"
        print >> self.txt, string
        print >> self.txt, "WARNING <<"
                

def eigenvalue_string(paw, comment=None):
    """
    Write eigenvalues and occupation numbers into a string.
    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """

    if not comment:
        comment=' '

    if len(paw.wfs.ibzk_kc) > 1:
        # not implemented yet:
        return ''

    s = ''
    if paw.wfs.nspins == 1:
        s += comment + 'Band   Eigenvalues  Occupancy\n'
        eps_n = paw.get_eigenvalues(kpt=0, spin=0)
        f_n   = paw.get_occupation_numbers(kpt=0, spin=0)
        if paw.wfs.world.rank == 0:
            for n in range(paw.wfs.nbands):
                s += ('%4d   %10.5f  %10.5f\n' % (n, eps_n[n], f_n[n]))
    else:
        s += comment + '                 Up                     Down\n'
        s += comment + 'Band  Eigenvalues  Occupancy  Eigenvalues  Occupancy\n'
        epsa_n = paw.get_eigenvalues(kpt=0, spin=0, broadcast=False)
        epsb_n = paw.get_eigenvalues(kpt=0, spin=1, broadcast=False)
        fa_n   = paw.get_occupation_numbers(kpt=0, spin=0, broadcast=False)
        fb_n   = paw.get_occupation_numbers(kpt=0, spin=1, broadcast=False)
        if paw.wfs.world.rank == 0:
            for n in range(paw.wfs.nbands):
                s += (' %4d  %11.5f  %9.5f  %11.5f  %9.5f\n' %
                      (n,epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
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
