from ASE.Units import units, Convert

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
        
        self.n = 0
        self.list_of_atoms = list_of_atoms
        if mpi.rank == MASTER:
            self.file = open(filename, mode)
        else:
            self.file = open('/dev/null', mode)
        print >> self.file, ' $coord', Convert(1, 'Ang', units.GetLengthUnit())
        self.write_atoms()
        print >> self.file, ' $grad'
    
    def __del__(self):
        print >> self.file, ' $end'
        self.file.close()

    def add(self):
        self.n += 1
        print >> self.file, 'cycle=', self.n,
        print >> self.file, 'SCF energy=', \
              self.list_of_atoms.GetPotentialEnergy() * self.Ha,
        print >> self.file, '|dE/dxyz|=', 0
        self.write_atoms()
        for atom in self.list_of_atoms:
            print >> self.file, '0. 0. 0.'
        self.file.flush()
       
    def write_atoms(self):
        for atom in self.list_of_atoms:
            c = atom.GetCartesianPosition()
            print  >> self.file, '%10.4f %10.4f %10.4f' % (c[0],c[1],c[2]),\
                      atom.GetChemicalSymbol()
