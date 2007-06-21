import re
import Numeric as num

from ASE import Atom, ListOfAtoms
from ASE.Utilities.GeometricTransforms import Translate
from ASE.IO.xyz import ReadXYZ, WriteXYZ
from ASE.IO.PDB import WritePDB

class Cluster(ListOfAtoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, atoms=None, filename=None, filetype=None):
        if atoms is None:
            ListOfAtoms.__init__(self,[])
        else:
            ListOfAtoms.__init__(self,atoms, periodic=False)

        if filename is not None:
            self.Read(filename,filetype)

    def Center(self):
        """Center the structure to unit cell"""
        extr = self.extreme_positions()
        cntr = 0.5 * (extr[0] + extr[1])
        cell = num.diagonal(self.GetUnitCell())
        Translate(self,tuple(.5*cell-cntr),'cartesian')

    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.GetCartesianPositions()
        return num.array([num.minimum.reduce(pos),num.maximum.reduce(pos)])

    def MinimalBox(self,border=0):
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
        Translate(self,tuple(-1.*num.array(extr[0])),'cartesian')
        self.SetUnitCell(tuple(extr[1]),fix=True)

        return self.GetUnitCell()

    def Read(self,filename,filetype=None):
        """Read the strcuture from some file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'xyz':
            loa = ReadXYZ(filename)
            self.__init__(loa)
        
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
        return len(self)

    def Write(self,filename,filetype=None):
        """Write the strcuture to file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'xyz':
            WriteXYZ(filename,self)
        elif filetype == 'pdb':
            WritePDB(filename,self)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
       
