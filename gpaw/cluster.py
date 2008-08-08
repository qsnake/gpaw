import math
import re
import numpy as npy

from ase import Atom, Atoms
from ase.io.cube import write_cube
from ase.io.xyz import read_xyz, write_xyz
from ase.io.pdb import write_pdb
from gpaw.io.cc1 import read_cc1
from ase.io.cube import read_cube
from ase import read as ase_read
from ase import write as ase_write
from gpaw.utilities.vector import Vector3d
#from gpaw.io.xyz import read_xyz

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
        return npy.array([npy.minimum.reduce(pos),npy.maximum.reduce(pos)])

    def find_connected(self, i, dmax):
        """Find the atoms connected to self[i] and return them."""
        
        def add_if_new(atoms, atom):
            new = False
            va = Vector3d(atom.position)
            dmin = 99999999999
            for a in atoms:
                dmin = min(dmin, va.distance(a.position))
            if dmin > 0.1:
                atoms += atom
                return True
            return False

        connected = Cluster(self[i:i + 1])

        isolated = False
        while not isolated:
            new = 0
            for ca in connected:
                # search atoms that are connected to you
                vca = Vector3d(ca.position)
                for oa in self:
                    if vca.distance(oa.position) < dmax:
                        new += int(add_if_new(connected, oa))
            if new == 0:
                isolated = True

        return connected

    def minimal_box(self,border=0):
        """The box needed to fit the structure in.
        The structure is moved to fit into the box [(0,x),(0,y),(0,z)]
        with x,y,z > 0 (fitting the ASE constriction).
        The border argument can be used to add a border of empty space
        around the structure.
        """

        if len(self) == 0:
            return None

        extr = self.extreme_positions()
 
        # add borders
        if type(border)==type([]):
            b=border
        else:
            b=[border,border,border]
        for i in range(3):
            extr[0][i]-=b[i]
            extr[1][i]+=b[i]-extr[0][i] # shifted already
            
        # move lower corner to (0,0,0)
        shift = tuple(-1.*npy.array(extr[0]))
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

    def read(self, filename, filetype=None):
        """Read the structure from some file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'cc1':
            loa = read_cc1(filename)
        elif filetype == 'cube':
            loa = read_cube(filename)
        elif filetype == 'vmol':
            from gpaw.utilities.viewmol import Trajectory
            traj = Trajectory(filename)
            loa = traj[-1]
        elif filetype == 'xyz':
            loa = read_xyz(filename)
        else:
            try:
                loa = ase_read(filename)
            except:
                raise NotImplementedError('unknown file type "'+filetype+'"')
        self.__init__(loa)
                
        return len(self)

    def write(self, filename, filetype=None, repeat=None):
        """Write the structure to file.

        Parameters
        ----------
        filetype: string
          can be given or it will be guessed from the filename
        repeat: array, eg.: [1,0,1]
          can be used to repeat the structure
        """

        out = self
        if repeat is None:
            out = self
        else:
            out = Cluster([])
            cell = self.get_cell().diagonal()
            for i in range(repeat[0]+1):
                for j in range(repeat[1]+1):
                    for k in range(repeat[2]+1):
                        copy = self.copy()
                        copy.translate(npy.array([i,j,k]) * cell)
                        out += copy

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'cube':
            write_cube(filename, out)
        elif filetype == 'pdb':
            write_pdb(filename, out)
        elif filetype == 'xyz':
            write_xyz(filename, out)
        else:
            try:
                ase_write(filename, self)
            except:
                raise NotImplementedError('unknown file type "'+filetype+'"')
                
       
