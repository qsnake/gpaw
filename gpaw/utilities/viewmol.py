from math import sqrt
import Numeric as num

from ASE import Atom
from ASE.Units import units, Convert

from gpaw.cluster import Cluster
import gpaw.mpi as mpi
MASTER = 0

class ViewmolTrajectory:
    """Write a trajectory for viewmol (http://viewmol.sourceforge.net)

    You can attach the writing to the Calculator:

    from gpaw.utilities.viewmol import ViewmolTrajectory

    c = Calculator()
    H2 = Cluster([Atom('H',(0,0,-.9)),Atom('H',(0,0,.9))])
    H2.SetCalculator(c)

    vmt = ViewmolTrajectory(H2)
    c.attach(vmt.add,100000)
    """
    def __init__(self, list_of_atoms, filename='trajectory.vmol',mode='w'):
        self.Ha = Convert(1, units.GetEnergyUnit(), 'Hartree')
        self.Ang = Convert(1, units.GetLengthUnit(), 'Ang')
        
        self.n = 0
        self.list_of_atoms = list_of_atoms
        if mpi.rank == MASTER:
            self.file = open(filename, mode)
        else:
            self.file = open('/dev/null', mode)
        print >> self.file, ' $coord', 1. / self.Ang
        self.write_atoms()
        print >> self.file, ' $grad'
    
    def __del__(self):
        print >> self.file, ' $end'
        self.file.close()

    def add(self):
        """Write current atomic position to the output file"""
        self.n += 1
        print >> self.file, 'cycle=', self.n,
        print >> self.file, 'SCF energy=', \
              self.list_of_atoms.GetPotentialEnergy() * self.Ha,
        forces = self.list_of_atoms.GetCartesianForces() * (self.Ha / self.Ang)
        max_force = 0
        for f in forces:
            max_force = max(max_force, sqrt(num.sum(f * f)))
        print >> self.file, '|max dE/dxyz|=', max_force
        self.write_atoms()
        for atom, f in zip(self.list_of_atoms, forces):
            print >> self.file, '%10.4g %10.4g %10.4g' % (f[0],f[1],f[2])
        self.file.flush()
       
    def write_atoms(self):
        for atom in self.list_of_atoms:
            c = atom.GetCartesianPosition()
            print  >> self.file, '%10.4f %10.4f %10.4f' % (c[0],c[1],c[2]),\
                      atom.GetChemicalSymbol()

    def read(self, filename='trajectory.vmol', position=0):
        """Read atom configuration of step position"""
        self.file = None
        f = open(filename)
        # find coordinates
        loa = Cluster([])
        coords = False
        for l in f.readlines():
            if coords:
                w = l.split()
                loa.append(Atom(w[3].replace ("\n", "" ),
                                (float(w[0]), float(w[1]), float(w[2]))))
