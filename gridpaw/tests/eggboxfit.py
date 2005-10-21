import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
import pylab as py


def eggboxfit(filename):
    
    """eggboxfit function
    Load the pickle file made by the eggbox function. Calculate the delta
    energy (the difference between max and min of the total energy) and the max
    force for each h, which include the whole array of the total energies at
    varying positions. Then plots those values as function of the grid space h.
    """

    a = open(filename)
    object = pickle.load(a)

    h = object['array of grid space']
    s = object['Atomic symbol']
    e = object['Total energy of atom']
    xc = object['exchange-correlation']
    g = object['array of grid points']

    f_E = open('Delta_Energy.dat', 'w')
    f_f = open('Max_Forces.dat', 'w')

    fmax = []
    delta_e = []

    for i in range( len(h) ):
        
        tot = max(e[i]) - min(e[i])
        delta_e.append(tot)
    
        f = (e[i,1:] - e[i,:-1]) / (h[i] / 101)
        fm = max( abs(f) )
        fmax.append(fm)

    print >> f_f, h, fmax
    print >> f_E, h, delta_e

    # The figure are plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}

    py.subplot(211)
    py.plot(h, delta_e, 'ro')
    py.title("Delta function of %s in eggbox with %s" %(s, xc), font)
    py.ylabel('Max energy difference (eV)', font)

    delta_min = py.amin(delta_e)
    delta_max = py.amax(delta_e)
    delta_wid = abs(delta_max - delta_min)
    h_min = py.amin(h)
    h_max = py.amax(h)
    h_wid = abs(h_max - h_min)
    py.axis([h_min-0.05*h_wid, h_max+0.05*h_wid,
             delta_min-0.05*delta_wid, delta_max+0.05*delta_wid])
    
    py.subplot(212)
    py.plot(h, fmax, 'g^')
    py.xlabel('Grid space (Aangstroem)', font)
    py.ylabel('Max force', font)

    f_min = py.amin(fmax)
    f_max = py.amax(fmax)
    f_wid = abs(f_max - f_min)
    py.axis([h_min-0.05*h_wid, h_max+0.05*h_wid,
             f_min-0.05*f_wid, f_max+0.05*f_wid])
    
    py.savefig('%s-%s-eggbox-test.png' %(s, xc))
    py.show()
    
    dir = os.getcwd()
    dir += '/%s-%s-eggbox-test.png' %(s, xc)

    text = """\
Eggbox test
=============

The maximum energy difference and force relative to its position in the grid
space are plotted against the gridspace for %s. The %s exchange correlation
functional has been used. The figure is shown below:

.. image:: %s

""" % (s, xc, dir)

    return text


if __name__ == '__main__':
    
    filename = sys.argv[1]
 
    eggboxfit(filename)
