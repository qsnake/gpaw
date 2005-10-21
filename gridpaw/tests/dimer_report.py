from dimerfit import dimerfit
import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
import pylab as py
from elements import elements


def dimer_report(filename):
    
    """Dimer_report function
    Load the pickle file made by the dimer function, where the results of the
    heavy calculation are saved. Make use of the dimerfit function to determine
    the values of the bond length, the atomization energy and the vibrational
    frequency. In the end, plot them against the variation of the grid space h.
    """

    f = open(filename)
    object = pickle.load(f)

    bond = object['Varying bond-length']
    energy = object['bonding energy']
    z = object['number of calculations']
    grid = object['array of grid space']
    s = object['Atomic symbol']
    xc = object['exchange-correlation']

    f_hw = open('vibrational_frequency.dat', 'w')
    f_b = open('bond_lengths.dat', 'w')
    f_Ea = open('atomization_energy.dat', 'w')

    m = elements[s]

    Grid_space = []
    Energy_bond = []
    Bond_length = []
    Frequency = []
 
    for i in range(len(grid)):
        b = bond[i]
        e = energy[i]
        h = grid[i]
        Grid_space.append(h)
                
        fhw, fb, fEa = dimerfit(b, e, s, p=z)

        print >> f_hw, h, fhw
        print >> f_b, h, fb
        print >> f_Ea, h, fEa
        
        Energy_bond.append(fEa)
        Bond_length.append(fb) 
        Frequency.append(fhw)

        i += 1

    # The figure are plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}

    py.subplot(311)
    py.plot(Grid_space, Energy_bond, 'go')
    py.title("%s dimer with %s \n" %(s, xc), font, fontsize = 14, color = 'r')
    py.ylabel('Energy (eV)', font)

    t1 = py.arange(0.0, 5.0, 0.1)

    Energy_min = py.amin(Energy_bond)
    Energy_max = py.amax(Energy_bond)
    Energy_wid = abs(Energy_max - Energy_min)
    Grid_min = py.amin(Grid_space)
    Grid_max = py.amax(Grid_space)
    Grid_wid = abs(Grid_max - Grid_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Energy_min-0.05*Energy_wid, Energy_max+0.05*Energy_wid])

    py.subplot(312)
    py.plot(Grid_space, Bond_length, 'r^')
    py.ylabel('Bond length (Aa)', font)
    py.axhline(y = m[2], color = 'b')

    Bond_min = py.amin(Bond_length)
    Bond_max = py.amax(Bond_length)
    Bond_wid = abs(Bond_max - Bond_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Bond_min-0.05*Bond_wid, Bond_max+0.05*Bond_wid])

    py.subplot(313)
    py.plot(Grid_space, Frequency, 'ms')
    py.xlabel('Grid space (Aangstroem)', font)
    py.ylabel('hw (meV)', font)

    Freq_min = py.amin(Frequency)
    Freq_max = py.amax(Frequency)
    Freq_wid = abs(Freq_max - Freq_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Freq_min-0.05*Freq_wid, Freq_max+0.05*Freq_wid])

    py.savefig('%s-Dimer-%s-Report.png' %(s, xc))
    py.show()
    
    dir = os.getcwd()
    dir += '/%s-Dimer-%s-Report.png' %(s, xc)

    text = """\
Dimer test
=============

The bond length between the two identical atoms are varied in order to
determine the values of the bond length, the atomization energy and the
vibrational frequency. These are plotted against the gridspace for %s. The %s
exchange correlation functional has been used. The figure is shown below:

.. image:: %s

""" % (s, xc, dir)

    return text
    

if __name__ == '__main__':
    
    filename = sys.argv[1]
 
    dimer_report(filename)

