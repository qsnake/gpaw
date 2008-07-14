import os.path

import numpy as npy

import _gpaw
from ase.atom import Atom
from ase.units import Bohr

class PointCharges(list):
    def __init__(self, file=None):
        list.__init__(self)
        
        if file is not None:
            self.read(file)

    def charge(self):
        """Return the summed charge of all point charges."""
        charge = 0
        for pc in self:
            charge += pc.charge
        return charge

    def read(self, file, filetype=None):
        """Read point charges from a file."""

        if hasattr(self, 'potential'):
            del(self.potential)
            del(self.gd)

        if filetype is None and isinstance(file, str):
            # estimate file type from name ending
            filetype = os.path.split(file)[-1].split('.')[-1]
        filetype = filetype.lower()

        if filetype == 'pc_info':
            self.read_PC_info(file)
        elif filetype == 'xyz':
            self.read_xyz(file)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
##        print "<PointCharges::read> found %d PC's" % len(self)


    def read_PC_info(self, file):
        """Read point charges from a PC_info file."""

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

    def read_xyz(self, file):
        """Read point charges from a xyz file."""

        if isinstance(file, str):
            f = open(file)
        else:
            f = file

        lines = f.readlines()
        L0 = lines[0].split()
        del lines[0:2]

        n = int(L0[0])
        for i in range(n):
            words = lines[i].split()
            dummy, x, y, z, q = words[:5]
            self.append(PointCharge(position=(float(x), 
                                              float(y), 
                                              float(z)),
                                    charge=float(q) ) )

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

    def get_ion_energy_and_forces(self, nuclei):
        """Return the ionic energy and force contribution."""
        forces = npy.zeros((len(atoms),3))
        energy = 0
        for a, atom in enumerate(atoms):
            for pc in self:
                dr = atom.position - pc.position
                dist = sqrt(npy.sum(dr*dr))
                e = atom.number * pc.charge / dist
                forces[a] += dr * e
                energy += e
        return energy, forces
        
    def translate(self, displacement):
        for pc in self:
            pc.position += displacement

    def write(self, file='PC.xyz', filetype=None):

        if filetype is None and isinstance(file, str):
            # estimate file type from name ending
            filetype = os.path.split(file)[-1].split('.')[-1]
        filetype = filetype.lower()

        if filetype == 'xyz':
            self.write_xyz(file)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
##        print "<PointCharges::read> found %d PC's" % len(self)

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

 
class ConstantPotential:
    """Constant potential for tests."""
    def __init__(self, constant=1):
        self.constant = constant
    def get_potential(self, gd):
        potential = gd.zeros() + self.constant
        return potential
    def get_ion_energy_and_forces(self, atoms):
        """Return the ionic energy and force contribution."""
        forces = npy.zeros((len(atoms),3))
        energy = 0
        return energy, forces
