import Numeric as num
from ASE.Units import units, Convert
from ASE import ListOfAtoms, Atom 

def ReadListOfAtomsFromCube(file):
    a0_A = Convert(1, 'Bohr', 'Ang')

    f=open(file)
    # first two lines are comments
    f.readline()
    f.readline()
    
    natoms = int(f.readline().split()[0])

    # next three lines define something we are not interested in
    f.readline()
    f.readline()
    f.readline()

    loa = ListOfAtoms([])
    for i in range(natoms):
        Z, dummy, x, y, z = f.readline().split()
        x = float(x) * a0_A
        y = float(y) * a0_A
        z = float(z) * a0_A
        loa.append(Atom(Z=int(Z),position=(x,y,z)))

    return loa

def WriteCubeFloat(atoms,grid,filename):
    """WriteCube(atoms, grid,  filename) -> None

    The atoms and grid information are written in Gaussian Cube format
    """
    import copy

    bohr = Convert(1, 'Bohr', units.GetLengthUnit())

    cubefile = file(filename,'w')

    # header
    cubefile.write('gpaw CUBE FILE\n')

    # loop order
    cubefile.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')

    natoms = len(atoms)
    unitcell = atoms.GetUnitCell()
    n_c = num.array(num.shape(grid))
    corner = num.zeros((3,),num.Float)
    d = unitcell.copy()
    for c in range(3):
        d[c] /= (n_c[c]+1)*bohr
        corner += d[c] 
    # number of atoms and left corner of the grid
    cubefile.write("%5d%12.6f%12.6f%12.6f \n" %
                   (natoms,corner[0],corner[1],corner[2]))

    for c in range(3):
        cubefile.write("%5d%12.6f%12.6f%12.6f \n"%
                       (n_c[c],d[c][0],d[c][1],d[c][2]))

    for atom in atoms:
        atomicnumber = atom.GetAtomicNumber()
        x,y,z = atom.GetCartesianPosition()/bohr
        cubefile.write("%5d%12.6f%12.6f%12.6f%12.6f \n"%(atomicnumber,0.0,x,y,z)) 

    count = 0        
    for ix in range(n_c[0]):
        for iy in range(n_c[1]):
            for iz in range(n_c[2]):
                cubefile.write("%e " %(grid[ix][iy][iz]))
                count += 1

                if ((count % 6) == 5):
                    cubefile.write("\n")
                    count = 0

    cubefile.close()

def WriteCube(atoms,grid,filename,real=False):
    import string
    if grid.typecode()=='D':

        if real:
           print 'Converting complex array into real'
           s = num.argsort(abs(grid.flat))
           maxval = grid.flat[s[-1]]
           phase = maxval/abs(maxval)
           grid1 = grid/phase
           WriteCubeFloat(atoms,grid1.real,filename)
        else:
            basename = string.split(filename,'.cube')
            filename_phase = basename[0] + '_phase.cube'
            print 'Using absolute value of complex array then writing Cube file.'
            print 'Writing phase information to file: ',filename_phase
            phase_grid = num.log(copy.copy(grid)+0.0000001).imag
            WriteCubeFloat(atoms,phase_grid,filename_phase)
            WriteCube(atoms,abs(grid),filename)
            
    else:
        WriteCubeFloat(atoms,grid,filename)
