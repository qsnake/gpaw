import math
import re
import Numeric as num

from ASE import Atom, ListOfAtoms
from ASE.Utilities.GeometricTransforms import Translate as GTTranslate
from ASE.Utilities.GeometricTransforms import RotateAboutAxis
from ASE.IO.xyz import ReadXYZ, WriteXYZ
from ASE.IO.PDB import WritePDB
from gpaw.io.Cube import ReadListOfAtomsFromCube
from gpaw.utilities.vector import Vector3d

class Cluster(ListOfAtoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, atoms=None, cell=None,
                 filename=None, filetype=None,
                 timestep=0.0):
        if atoms is None:
            ListOfAtoms.__init__(self,[],cell=cell)
        else:
            ListOfAtoms.__init__(self,atoms, cell=cell,periodic=False)

        if filename is not None:
            self.read(filename,filetype)

        self.timestep(timestep)
        
    def __add__(self, other):
        assert(type(other) == type(self))
        result = Cluster(self.Copy())
        for a in other:
            result.append(a.Copy())
        return result

    def __ladd__(self, other):
        print "__ladd__"
        return self.add(other)
        
    def __radd__(self, other):
        print "__radd__"
        return self.add(other)
        
    def Center(self):
        """Center the structure to unit cell"""
        extr = self.extreme_positions()
        cntr = 0.5 * (extr[0] + extr[1])
        cell = num.diagonal(self.GetUnitCell())
        GTTranslate(self,tuple(.5*cell-cntr),'cartesian')

    def center_of_mass(self):
        """Return the structures center of mass"""
        cm = num.zeros((3,),num.Float)
        M = 0.
        for atom in self:
            m = atom.GetMass()
            M += m
            cm += m * atom.GetCartesianPosition()
        return cm/M

    def Copy(self):
        return self.copy()

    def copy(self):
        return Cluster(ListOfAtoms.Copy(self))
    
    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.GetCartesianPositions()
        return num.array([num.minimum.reduce(pos),num.maximum.reduce(pos)])

    def MinimalBox(self,border=0):
        return self.minimal_box(border)
    
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
        GTTranslate(self,tuple(-1.*num.array(extr[0])),'cartesian')
        self.SetUnitCell(tuple(extr[1]),fix=True)

        return self.GetUnitCell()

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
            loa = ReadXYZ(filename)
            self.__init__(loa)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
        return len(self)

    def rotate(self, axis, angle=None, unit='rad'):
        """Rotate the structure about the given axis with the given angle.
        Note, that the right hand rule applies: If your right thumb points
        into the direction of the axis, the other fingers show the rotation
        direction."""
        axis=Vector3d(axis)
        if angle is None:
            angle = axis.length()
        axis.length(1.)
        use_RAA=True
        use_RAA=False
        if use_RAA:
            if unit == 'rad':
                angle *= 180. / math.pi
            RotateAboutAxis(self, axis, -angle)
        else:
            for a in self:
                v=Vector3d(a.GetCartesianPosition())
                v.rotate(axis, angle)
                a.SetCartesianPosition(v)

    def timestep(self,timestep=None):
        """Set and/or get the timestep label of this structure"""
        if timestep is not None:
            self.ts = float(timestep)
        return self.ts

    def translate(self,trans_vector):
        """Translate the whole structure"""
        GTTranslate(self,tuple(trans_vector),'cartesian')

    def write(self,filename,filetype=None):
        """Write the strcuture to file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'pdb':
            WritePDB(filename,self)
        elif filetype == 'xyz':
            uc = self.GetUnitCell()
            if uc:
                id=' unit cell'
                for v in uc:
                    id+=' (%g,%g,%g)' % (v[0],v[1],v[2])
            WriteXYZ(filename,self,id=id)
        elif filetype == 'pdb':
            WritePDB(filename,self)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
       
