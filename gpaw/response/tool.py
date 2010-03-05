import numpy as np
from scipy.optimize import leastsq

def find_peaks(x,y,threshold = None):
    """ Find peaks for a certain curve.

    Usage:
    threshold = (xmin, xmax, ymin, ymax)

    """

    assert type(x) == np.ndarray and type(y) == np.ndarray
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape[0] == y.shape[0]

    if threshold is None:
        threshold = (x.min(), x.max(), y.min(), y.max())
        
    if type(threshold) is not tuple:
        threshold = (threshold, )

    if len(threshold) == 1:
        threshold += (x.max(), y.min(), y.max())
    elif len(threshold) == 2:
        threshold += (y.min(), y.max())
    elif len(threshold) == 3:
        threshold += (y.max(),)
    else:
        pass

    xmin = threshold[0]
    xmax = threshold[1]
    ymin = threshold[2]
    ymax = threshold[3]

    peak = {}
    npeak = 0
    for i in range(1, x.shape[0]-1):
        if (y[i] >= ymin and y[i] <= ymax and
            x[i] >= xmin and x[i] <= xmax ):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peak[npeak] = np.array([x[i], y[i]])
                npeak += 1

    peakarray = np.zeros([npeak, 2])
    for i in range(npeak):
        peakarray[i] = peak[i]
        
    return peakarray

def lorz_fit(x,y, npeak=1, initpara = None):
    """ Fit curve using Lorentzian function

    Note: currently only valid for one and two lorentizian

    The lorentzian function is defined as::

                      A w
        lorz = --------------------- + y0
                (x-x0)**2 + w**2

    where A is the peak amplitude, w is the width, (x0,y0) the peak position

    Parameters:

    x, y: ndarray
        Input data for analyze
    p: ndarray
        Parameters for curving fitting function. [A, x0, y0, w]
    p0: ndarray
        Parameters for initial guessing. similar to p
    
    """

    
    def residual(p, x, y):
        
        err = y - lorz(x, p, npeak)
        return err

    def lorz(x, p, npeak):

        if npeak == 1:        
            return p[0] * p[3] / ( (x-p[1])**2 + p[3]**2 ) + p[2]
        if npeak == 2:
            return ( p[0] * p[3] / ( (x-p[1])**2 + p[3]**2 ) + p[2]
                   + p[4] * p[7] / ( (x-p[5])**2 + p[7]**2 ) + p[6] )
        else:
            raise ValueError('Larger than 2 peaks not supported yet!')

    if initpara is None:
        if npeak == 1:
            initpara[i] = np.array([1., 0., 0., 0.1])
        if npeak == 2:
            initpara[i] = np.array([1., 0., 0., 0.1,
                                    3., 0., 0., 0.1])
    p0 = initpara
    
    result = leastsq(residual, p0, args=(x, y), maxfev=2000)

    yfit = lorz(x, result[0],npeak)

    return yfit, result[0]
    
