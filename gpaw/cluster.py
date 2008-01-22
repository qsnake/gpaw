import math
import re
import numpy as npy

from ase import Atom, Atoms
from ase.io.xyz import read_xyz, write_xyz
from ASE.IO.PDB import WritePDB
from gpaw.io.Cube import ReadListOfAtomsFromCube
from gpaw.utilities.vector import Vector3d

class Cluster(Atoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, atoms=None, cell=None,
                 filename=None, filetype=None):
        if atoms is None:
            atoms = []
        Atoms.__init__(self,atoms, cell=cell)

        if filename is not None:
            self.read(filename, filetype)
    
    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.get_positions()
        return npy.array([npy.minimum.reduce(pos),npy.maximum.reduce(pos)])

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
        self.translate(tuple(-1.*npy.array(extr[0])))
        self.set_cell(tuple(extr[1]), fix=True)

        return self.get_cell()

    def read(self,filename,filetype=None):
        """Read the strcuture from some file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'cube':
            loa = ReadListOfAtomsFromCube(filename)
            self.__init__(loa)
        elif filetype == 'xyz':
            loa = read_xyz(filename)
            self.__init__(loa)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
        return len(self)

    def write(self, filename, filetype=None):
        """Write the strcuture to file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'pdb':
            WritePDB(filename,self)
        elif filetype == 'xyz':
            write_xyz(filename, self)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
       
