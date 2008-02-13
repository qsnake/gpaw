from math import sqrt
import numpy as npy

from ase import Angstrom, Atom, Hartree, PickleTrajectory

from gpaw.cluster import Cluster
import gpaw.mpi as mpi
MASTER = 0

class ViewmolTrajectory2(list):
    def __add__(self, other):
        for loa in other:
            self.append(loa)
    
    def __init__(self, filename=None):
        if filename:
            self.read(filename)
    
    def read(self, filename='trajectory.vmol'):
        
        f = open(filename)

        # read the definition first 
        definition=False
        for line in f:
            w = line.split()
            if not definition:
                if w[0] == '$coord':
                    definition=True
                    self.scale = float(w[1])
                    loa = Cluster([])
            else:
                if w[0] == '$grad':
                    # definition ends here
                    self.definition = loa
                    break
                else:
                    # we assume this is a coordinate entry
                    coo = (float(w[0]),  float(w[1]), float(w[2]))
                    loa.append(Atom(w[3], coo))
##        print "<read> loa=", len(loa)

        # get the iterations            
        cycle = False
        for line in f:
            w = line.split()
            if not cycle:
                # search for the cycle keyword
                if w[0] == 'cycle=':
                    cycle=True
                    n_coo=0
                    n_F=0
                    self.append(Cluster([]))
            else:
                if n_coo < len(self.definition):
                    n_coo += 1
                    coo = (float(w[0]),  float(w[1]), float(w[2]))
                    self[-1].append(Atom(w[3], coo))
                elif n_F < len(self.definition):
                    F = (float(w[0]),  float(w[1]), float(w[2]))
                    self[-1][n_F].F = F
                    n_F += 1
                    if n_F == len(self.definition):
                        cycle=False
        
    # ASE interface
    def GetListOfAtoms(self, frame):
#        return self.definition
        return self[frame]

    def write(self, filename='trajectory.vmol', filetype=None):
        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()
        
        if filetype == 'xyz':
             WriteXYZ(filename, trajectory=self)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')

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
    def __init__(self, atoms, filename='trajectory.vmol',mode='w'):
        self.Ha = Hartree
        self.Ang = Angstrom
        
        self.n = 0
        self.atoms = atoms
        if mpi.rank == MASTER:
            self.file = open(filename, mode)
        else:
            self.file = open('/dev/null', mode)
        print >> self.file, ' $coord', 1. / self.Ang
        self.write_atoms(self.atoms)
        print >> self.file, ' $grad'
    
    def __del__(self):
        print >> self.file, ' $end'
        self.file.close()

    def add(self, atoms=None):
        """Write current atomic position to the output file"""
        if atoms is None:
            atoms = self.atoms
        self.n += 1
        print >> self.file, 'cycle=', self.n,
        print >> self.file, 'SCF energy=', \
              atoms.get_potential_energy() * self.Ha,
        forces = atoms.get_forces() * (self.Ha / self.Ang)
        max_force = 0
        for f in forces:
            max_force = max(max_force, sqrt(npy.sum(f * f)))
        print >> self.file, '|max dE/dxyz|=', max_force
        self.write_atoms(atoms)
        for atom, f in zip(self.atoms, forces):
            print >> self.file, '%10.4g %10.4g %10.4g' % (f[0],f[1],f[2])
        self.file.flush()
       
    def write_atoms(self, atoms):
        for atom in atoms:
            c = atom.get_positions()[0]
            print  >> self.file, '%10.4f %10.4f %10.4f' % (c[0],c[1],c[2]),\
                      atom.get_chemical_symbols()[0]

    def read(self, filename='trajectory.vmol', position=0):
        """Read atom configurations of step position"""
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
                
def write_viewmol(pt, filename, mode='w'):
    """Write PickleTrajectory as viewmol file."""
    atoms = pt[0]
    vt = ViewmolTrajectory(atoms, filename, mode)
    for atoms in pt:
        vt.add(atoms)
    del(vt)
    
