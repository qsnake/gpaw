import numpy as npy
from gpaw.gauss import Gauss, Lorentz

def fold(arrays,width,xmin=None,xmax=None,folding='Gauss'):
    """fold arrays with Gaussians/Lorentzians

    The array must have the following structure:
    array[0]...array[n], where
    array[i][0]=list of x-values, 
    array[i][1]=list of f(x)-values

    returns an array of the following structure:
    res[0]....res[n+1], where
    res[0]=list of sampled x-values,
    res[i+1]=list of folded f(x) values corresponding to array[i]
    """

    xmi=None
    xma=None
    dx = width/4. # sample 4 points in the width
    
    # initialise the folding function
    if folding == 'Gauss':
        func=Gauss(width)
    elif folding == 'Lorentz':
        func=Lorentz(width)
    else:
        raise RuntimeError('unknown folding "'+folding+'"')

    # check shapes and get xmin and xmax
    if xmin: xmi=xmin
    if xmax: xma=xmax
    for a in arrays:
        assert(a.shape[0],2)
        if xmi is None: xmi = a[0][0]
        if xma is None: xma = a[0][0]
        if xmin is None: xmi = min(xmi,min(a[0]))
        if xmax is None: xma = max(xma,max(a[0]))
    # make sure, that the lowest and highest peaks are in
    if xmin is None: xmi-=5*dx
    if xmax is None: xma+=5*dx

    n = int((xma-xmi)/dx)

    res = npy.zeros((a.shape[0]+1,n+1))

    x = xmi
    for i in range(n+1):
        j=0
        res[j][i]=x
        for a in arrays:
            j+=1
            for xx,yy in zip(a[0],a[1]):
                res[j][i] += yy*func.Get(xx-x)
        x += dx

    return res
