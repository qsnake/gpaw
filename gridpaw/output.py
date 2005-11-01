import Numeric as num
from ASE.ChemicalElements.symbol import symbols

from gridpaw import debug, enumerate, sigusr1


def print_info(paw):
    out = paw.out
    domain = paw.domain
#    print >> out, 'Gamma-point calculation'
    print >> out, 'parallel: parsize = ', domain.parsize_i


def print_converged(paw):
    out = paw.out
    print >> out, 'Converged after %d iterations.' % paw.niter

    print >> out
    if len(paw.nuclei) == 1:
        print >> out, 'energy contributions relative to reference atom:'
    else:
        print >> out, 'energy contributions relative to reference atoms:'

    print >> out, '-------------------------'

    energies = [('kinetic:', paw.Ekin),
                ('potential:', paw.Epot),
                ('XC:', paw.Exc),
                ('entropy (-ST):', -paw.S),
                ('local:', paw.Ebar)]

    for name, e in energies:
        print >> out, '%-14s %+10.5f' % (name, paw.Ha * e)

    print >> out, '-------------------------'
    print >> out, 'free energy:   %+10.5f' % (paw.Ha * paw.Etot)
    print >> out, 'zero Kelvin:   %+10.5f' % \
          (paw.Ha * (paw.Etot + 0.5 * paw.S))
    print >> out
    epsF = paw.wf.occupation.get_fermi_level()
    if epsF is not None:
        print >> out, 'Fermi level:', paw.Ha * epsF

    paw.wf.print_eigenvalues(out, paw.Ha)

    print >> out
    charge = paw.finegd.integrate(paw.rhot_g)
    print >> out, 'total charge: %f electrons' % charge

    dipole = paw.finegd.calculate_dipole_moment(paw.rhot_g)
    print >> out, 'dipole moment: %s' % (dipole * paw.a0)


def plot_atoms(paw):
    domain = paw.domain
    nuclei = paw.nuclei
    out = paw.out
    cell_i = domain.cell_i
    pos_ai = cell_i * [nucleus.spos_i for nucleus in nuclei]
    Z_a = [nucleus.setup.Z for nucleus in nuclei]
    print >> out, plot(pos_ai, Z_a, cell_i)
    print >> out
    print >> out, 'unitcell:'
    print >> out, '         periodic  length  points   spacing'
    print >> out, '  -----------------------------------------'
    for i in range(3):
        print >> out, '  %s-axis   %s   %8.4f   %3d    %8.4f' % \
              ('xyz'[i],
               ['no ', 'yes'][domain.periodic_i[i]],
               paw.a0 * domain.cell_i[i],
               paw.gd.N_i[i],
               paw.a0 * paw.gd.h_i[i])
    print >> out
    

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

