"""IO routines for gOpenMol binary plt format"""

from struct import pack
import Numeric as num

def write_plt(gd, grid, filename, type=4):
    """write: I: gd = grid descriptor
                 grid = the grid to write
                 type = Type of surface (integer)
    input is assumed to be in atomc units (Bohr)
    """

    if tuple(gd.N_c) != grid.shape:
        raise RuntimeError("grid descriptor does not correspond to the grid")
    
    scale = 0.52917725 # Bohr to Angstroem
    
    f = open(filename, 'w')
    f.write(pack('ii', 3, type))
    nx, ny, nz = gd.N_c
    f.write(pack('iii', nx, ny, nz))

    # ASE uses (0,0,0) as origin ????
    x0 = y0 = z0 = 0.
    dx, dy, dz = scale*gd.h_c
    xe = x0 +(nx-1)*dx
    ye = y0 +(ny-1)*dy
    ze = z0 +(nz-1)*dz
    f.write(pack('ff', z0, ze ))
    f.write(pack('ff', y0, ye ))
    f.write(pack('ff', x0, xe ))

    # we need a float array
    fgrid = num.asarray(num.transpose(grid), num.Float32)
    f.write(fgrid.tostring())

    f.close()

