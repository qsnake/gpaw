"""IO routines for gOpenMol binary plt format"""

from struct import calcsize,pack,unpack
import Numeric as num
from gpaw.utilities import check_unit_cell

def read_plt(filename):
    """Read plt files
    returns the cell(3x3 matrix), the grid and the origin
    """
    f = open(filename)
    fmt='ii'
    # check endian encoding
    byteswap = False
    three, tos = unpack(fmt,f.read(calcsize(fmt)))
##    print "<read_plt> three=",three
    if three != 3: byteswap = True

    fmt='iii'
    if byteswap: fmt='>iii'
    nx, ny, nz = unpack(fmt,f.read(calcsize(fmt)))
##    print "<read_plt> nx,ny,nz=",nx,ny,nz
    

    fmt='ff'
    if byteswap: fmt='>ff'
    z0, ze = unpack(fmt,f.read(calcsize(fmt)))
    y0, ye = unpack(fmt,f.read(calcsize(fmt)))
    x0, xe = unpack(fmt,f.read(calcsize(fmt)))
    dz = (ze-z0)/(nz-1)
    dy = (ye-y0)/(ny-1)
    dx = (xe-x0)/(nx-1)
    cell = num.zeros((3,3),num.Float)
    cell[0,0] = nx*dx 
    cell[1,1] = ny*dy 
    cell[2,2] = nz*dz 
    
    fmt='f'
    if byteswap: fmt='>f'
    size = nx*ny*nz * calcsize(fmt)
##    print "<read_plt> nx,ny,nz=",nx,ny,nz
##    print "<read_plt> size=",size
    arr = num.fromstring(f.read(size),num.Float32)
    if byteswap: arr = arr.byteswapped()
    f.close()

    return cell, num.transpose(num.resize(arr,(nx,ny,nz))), (x0,y0,z0)

def write_plt(cell, grid, filename,
              origin=(0.0,0.0,0.0), # ASE uses (0,0,0) as origin
              type=4):
    """Input:
    cell = unit cell object as given from ListOfAtoms.GetUnitCell()
    grid = the grid to write
    type = Type of surface (integer)
    
    cell is assume to be in Angstroms and the grid in atomc units (Bohr)"""
    
    # Check that the cell is orthorhombic
    check_unit_cell(cell)
    xe, ye, ze = num.diagonal(cell)

    # Check, that the grid is 3D
    if len(grid.shape) != 3:
        raise RuntimeError("grid must be 3D")
    nx, ny, nz = grid.shape
    dx, dy, dz = [ xe/nx, ye/ny, ze/nz ]

    f = open(filename, 'w')
    f.write(pack('ii', 3, type))
    f.write(pack('iii', nx, ny, nz))

    x0, y0, z0 = origin
    xe = x0 +(nx-1)*dx
    ye = y0 +(ny-1)*dy
    ze = z0 +(nz-1)*dz
    f.write(pack('ff', z0, ze ))
    f.write(pack('ff', y0, ye ))
    f.write(pack('ff', x0, xe ))

    # we need a float array
    if grid.typecode() == 'f':
        fgrid = num.transpose(grid)
    else:
        fgrid = num.array(num.transpose(grid).tolist(),'f')
#    num.asarray does not work here !
#    fgrid = num.asarray(num.transpose(grid), num.Float32)
    f.write(fgrid.tostring())

    f.close()

