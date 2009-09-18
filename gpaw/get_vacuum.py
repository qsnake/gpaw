from ase import *
from gpaw import *
from ase.units import Bohr, Hartree
        

def coordinates(gd,a0,v=None):
    """Constructs and returns matrices containing cartesian coordinates,
       and the square of the distance from vector v
       If v=None the corner (0,0,0) of the box is taken.
       The origin is placed in the center of the box described by the given
       grid-descriptor 'gd'.
    """    
    I  = indices(gd.n_c)
    dr = reshape(gd.h_c, (3, 1, 1, 1))
    if v is None:
        #r0 = reshape(gd.h_c * gd.beg_c - .5 * gd.cell_c, (3, 1, 1, 1))
        r0 = reshape(gd.h_c * gd.beg_c , (3, 1, 1, 1))
    else:
        r0 = reshape(gd.h_c * gd.beg_c - v/a0, (3, 1, 1, 1))
    r0 = ones(I.shape) * r0
    xyz = r0 + I * dr
    xyz = xyz*a0
    r2 = sum(xyz**2, axis=0)


def get_vacuum(calc,s=0,delta=0.5):
    """ returns value of effective ks potential on the boundaries """
    #get effective potential
    v = calc.get_effective_potential(spin=s, pad=False)
    #get coordinates
    gd = calc.gd
    xyz, r2 = coordinates(gd,Bohr)
    #get slices:planes xy,xz,zy
    atoms = calc.get_atoms()
    cell = atoms.get_cell()
    for i in range(3):
        plane = v[where(xyz[i,:,:,:]<=delta)]
    #get averages and variances
        nlen = len(plane)
    #print
        print nlen
        print 'mean:',plane.mean(),'st. dev.:',sqrt(plane.var())
    for i in range(3):
        plane = v[where(xyz[i,:,:,:]>=(cell[i,i]-delta))]
        #get averages and variances
        nlen = len(plane)
        #print
        print nlen
        print 'mean:',plane.mean(),'st. dev.:',sqrt(plane.var())
    return


calc = Calculator('out.gpw')
get_vacuum(calc)
