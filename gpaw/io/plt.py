"""IO routines for gOpenMol binary plt format"""

from struct import pack
import Numeric as num
from gpaw.utilities import check_unit_cell

def write_plt(cell, grid, filename, type=4):
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

    # ASE uses (0,0,0) as origin ????
    x0 = y0 = z0 = 0.
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

