import numpy as npy

import _gpaw
from ase.atom import Atom
from ase.units import Bohr

class PointCharges(list):
    def __init__(self, file=None):
        list.__init__(self)
        
        if file is not None:
            self.read(file)

    def read(self, file):

        if hasattr(self, 'potential'):
            del(self.potential)
            del(self.gd)

        if isinstance(file, str):
            f = open(file)
        else:
            f = file

        lines = f.readlines()
        L0 = lines[0].split()
        del lines[0]

        for line in lines:
            words = line.split()
            if len(words) > 3:
                q, x, y, z = words[:4]
                self.append(PointCharge(position=(float(x) * Bohr, 
                                                  float(y) * Bohr, 
                                                  float(z) * Bohr),
                                        charge=float(q) ) )
            else:
                break
        print "<PointCharges::read> found %d PC's" % len(self)

    def get_potential(self, gd):
        """Create the Coulomb potential on the grid."""

        if hasattr(self, 'potential') and gd == self.gd:
            # nothing changed
            return self.potential

        potential = gd.empty()

        n = len(self)
        pc_c = npy.empty((n, 3))
        charges = npy.empty((n))
        for a, pc in enumerate(self):
            pc_c[a] = pc.position / Bohr 
            charges[a] = pc.charge

        _gpaw.pc_potential(potential, pc_c, charges, 
                           gd.beg_c, gd.end_c, gd.h_c)

        # save grid descriptor and potential for future use
        self.potential = potential
        self.gd = gd

        return potential

    def translate(self, displacement):
        for pc in self:
            pc.position += displacement

    def write_xyz(self, file='PC.xyz'):
        if isinstance(file, str):
            f = open(file, 'w')
        else:
            f = file
        f.write('%d\nPoint charges\n' % len(self))
        for pc in self:
            (x, y, z) = pc.position
            q = pc.charge
            f.write('PC  %12.6g %12.6g %12.6g  %12.6g\n' % (x, y, z, q))
        f.close()
                    
       

class PointCharge(Atom):
    def __init__(self, position, charge):
        Atom.__init__(self, position=position, charge=charge)

 
