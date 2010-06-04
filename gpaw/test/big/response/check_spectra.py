import numpy as np

def check_EELS(head):
# check whether EELS spectra is unchanged

    d = np.loadtxt(head + '_q_list')
    q = d[:]
    ndata = q.shape[0]
    xpeak = np.zeros(ndata)
    ypeak = np.zeros(ndata)
    
    for i in range(ndata):
        filename = head + '_EELS_' + str(i+1) 
    
        d = np.loadtxt(filename)
        x = d[:, 0]
        y = d[:, 2]

        dw = x[1] - x[0]
        Nw = np.int(10 / dw)
        for ii in range(1, Nw):
            if y[ii] > y[ii-1] and y[ii] > y[ii+1]:
                xpeak[i] = ii * dw
                ypeak[i] = y[ii]
                
    return xpeak, ypeak

                
xpeak, ypeak = check_EELS('graphite')

xcheck = np.array([6.8, 7.1, 7.5, 8.1, 8.9, 9.7, 6.3])
ycheck = np.array([0.95075697, 0.96481923, 0.9835551, 0.89167785,
                   0.71880955, 0.51752835, 0.30116302])

for i in range(xpeak.shape[0]):
    if np.abs(xpeak[i] - xcheck[i]) > 1e-2:
        print i, xpeak[i], xcheck[i]
        raise ValueError('Plasmon peak not correct ! ')
    if np.abs(ypeak[i] - ycheck[i]) > 1e-5:
        print i, ypeak[i], ycheck[i]
        raise ValueError('Please check spectrum strength ! ')
    
