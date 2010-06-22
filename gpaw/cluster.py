"""Extensions to the ase Atoms class

"""
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.data import covalent_radii
from ase.calculators.neighborlist import NeighborList

class Cluster(Atoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, *args, **kwargs):

        self.data = {}

        if len(args) > 0:
            filename = args[0]
            if isinstance(filename, str):
                self.read(filename, kwargs.get('filetype'))
                return
        else:
            Atoms.__init__(self, [])

        if kwargs.get('filename') is not None:
            filename = kwargs.pop('filename')
            Atoms.__init__(self, *args, **kwargs)
            self.read(filename, kwargs.get('filetype'))
        else:
            Atoms.__init__(self, *args, **kwargs)
    
    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.get_positions()
        return np.array([np.minimum.reduce(pos), np.maximum.reduce(pos)])

    def find_connected(self, index, dmax=None, scale=1.5):
        """Find the atoms connected to self[index] and return them.

        If dmax is not None:
        Atoms are defined to be connected if they are nearer than dmax
        to each other.

        If dmax is None:
        Atoms are defined to be connected if they are nearer than the
        sum of their covalent radii * scale to each other.

        """

        # set neighbor lists
        neighborlist = []
        if dmax is None:
            # define neighbors according to covalent radii
            radii = scale * covalent_radii[self.get_atomic_numbers()]
            for atom in self:
                positions = self.positions - atom.position
                distances = np.sqrt(np.sum(positions**2, axis=1))
                radius = scale * covalent_radii[atom.get_atomic_number()]
                neighborlist.append(np.where(distances < radii + radius)[0])
        else:
            # define neighbors according to distance
            nl = NeighborList([0.5 * dmax] * len(self), skin=0)
            nl.update(self)
            for i, atom in enumerate(self):
                neighborlist.append(list(nl.get_neighbors(i)[0]))

        connected = list(neighborlist[index])
        isolated = False
        while not isolated:
            isolated = True
            for i in connected:
                for j in neighborlist[i]:
                    if j in connected:
                        pass
                    else:
                        connected.append(j)
                        isolated = False 
 
        atoms = Cluster()
        for i in connected:
            atoms.append(self[i])

        return atoms

    def minimal_box(self, border=0, h=None):
        """The box needed to fit the structure in.

        The structure is moved to fit into the box [(0,x),(0,y),(0,z)]
        with x,y,z > 0 (fitting the ASE constriction).
        The border argument can be used to add a border of empty space
        around the structure.

        If h is set, the box is extended to ensure that box/h is 
        a multiple of 4. 
        This ensures that GPAW uses the desired h.

        The shift applied to the structure is returned.
         """

        if len(self) == 0:
            return None

        extr = self.extreme_positions()
 
        # add borders
        if type(border)==type([]):
            b = border
        else:
            b = [border, border, border]
        for c in range(3):
            extr[0][c] -= b[c]
            extr[1][c] += b[c] - extr[0][c] # shifted already

        # check for multiple of 4
        if h is not None:
            if not hasattr(h, '__len__'):
                h = np.array([h, h, h])
            for c in range(3):
                # apply the same as in paw.py 
                L = extr[1][c] # shifted already
                N = max(4, int(L / h[c] / 4 + 0.5) * 4)
                # correct L
                dL = N * h[c] - L
                # move accordingly
                extr[1][c] += dL # shifted already
                extr[0][c] -= dL / 2.
            
        # move lower corner to (0, 0, 0)
        shift = tuple(-1. * np.array(extr[0]))
        self.translate(shift)
        self.set_cell(tuple(extr[1]))

        return shift

    def get(self, name):
        """General get"""
        attr = 'get_' + name
        if hasattr(self, attr):
            getattr(self, attr)(data)
        elif self.data.has_key(name):
            return self.data[name]
        else:
            return None

    def set(self, name, data):
        """General set"""
        attr = 'set_' + name
        if hasattr(self, attr):
            getattr(self, attr)(data)
        else:
            self.data[name] = data

    def read(self, filename, format=None):
        """Read the structure from some file. The type can be given
        or it will be guessed from the filename."""

        self.__init__(read(filename, format=format))
        return len(self)

    def write(self, filename=None, format=None, repeat=None):
        """Write the structure to file.

        Parameters
        ----------
        format: string
          can be given or it will be guessed from the filename
        repeat: array, eg.: [1,0,1]
          can be used to repeat the structure
        """

        if filename is None:
            if format is None:
                raise RuntimeError('Please specify either filename or format.')
            else:
                filename = self.get_name() + '.' + format

        out = self
        if repeat is None:
            out = self
        else:
            out = Cluster([])
            cell = self.get_cell().diagonal()
            for i in range(repeat[0] + 1):
                for j in range(repeat[1] + 1):
                    for k in range(repeat[2] + 1):
                        copy = self.copy()
                        copy.translate(np.array([i, j, k]) * cell)
                        out += copy

        write(filename, out, format)

       
