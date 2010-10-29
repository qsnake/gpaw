import numpy as np
from math import sqrt, pi

def get_primitive_cell(a):
    """From the unit cell, calculate primitive cell and volume. """

    vol = np.abs(np.dot(a[0],np.cross(a[1],a[2])))
    BZvol = (2. * pi)**3 / vol

    b = np.linalg.inv(a.T)

    b *= 2 * pi

    assert np.abs((np.dot(b.T, a) - 2.*pi*np.eye(3)).sum()) < 1e-10

    return b, vol, BZvol


def set_Gvectors(acell, bcell, nG, Ecut):
    """Calculate the number of planewaves with a certain cutoff, their reduced coordinates and index."""

    # Refer to R.Martin P85
    Gmax = np.zeros(3, dtype=int)
    for i in range(3):
        a = acell[i]
        Gcut = sqrt(2*Ecut[i])
        Gmax[i] = sqrt(a[0]**2 + a[1]**2 + a[2]**2) * Gcut/ (2*pi)
     
    Nmax = 2 * Gmax + 1
    
    m = {}
    for dim in range(3):
        m[dim] = np.zeros(Nmax[dim],dtype=int)
        for i in range(Nmax[dim]):
            m[dim][i] = i
            if m[dim][i] > np.int(Gmax[dim]):
                m[dim][i] = i- Nmax[dim]       

    G = np.zeros((Nmax[0]*Nmax[1]*Nmax[2],3),dtype=int)
    n = 0
    for i in range(Nmax[0]):
        for j in range(Nmax[1]):
            for k in range(Nmax[2]):
                tmp = np.array([m[0][i], m[1][j], m[2][k]])
                tmpG = np.dot(tmp, bcell)
                Gmod = sqrt(tmpG[0]**2 + tmpG[1]**2 + tmpG[2]**2)
                if Gmod < Gcut:
                    G[n] = tmp
                    n += 1
    npw = n
    Gvec = G[:n]

    Gindex = {}
    id = np.zeros(3, dtype=int)

    for iG in range(npw):
        G = Gvec[iG]
        for dim in range(3):
            if G[dim] >= 0:
                id[dim] = G[dim]
            else:
                id[dim] = nG[dim] - np.abs(G[dim])
        Gindex[iG] = np.array(id)
    
    return npw, Gvec, Gindex
