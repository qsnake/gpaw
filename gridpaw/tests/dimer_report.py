from dimerfit import dimerfit
import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
from pylab import *
from elements import elements


def dimer_report(filename):
    
    """Dimer report.
    
    Load the pickle file made by the dimer function, where the results of the
    heavy calculation are saved. Make use of the dimerfit function to determine
    the values of the bond length, the atomization energy and the vibrational
    frequency. In the end, plot them against the variation of the grid
    spacings, h.
    """
    
    f = open(filename)
    object = pickle.load(f)

    bond = object['Bond lengths']
    energy = object['Atomization energy']
    grid = object['Grid spacings']
    s = object['Atomic symbol']
    xc = object['Exchange-correlation functional']
    L = object['Size of unit cell']

    m = elements[s]
    grid_space = []
    energy_bond = []
    bond_length = []
    frequency = []
 
    for i in range(len(grid)):
        
        b = bond[i]
        e = energy[i]
        h = grid[i]
        grid_space.append(h)
                
        fhw, fb, fEa = dimerfit(b, e, s)
       
        energy_bond.append(fEa)
        bond_length.append(fb) 
        frequency.append(fhw)

        i += 1

    # The figure are plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}
    
    figure(figsize = (10, 10))
    subplot(311)
    plot(grid_space, energy_bond, 'go')
    title("%s dimer with %s \n" %(s, xc), font, fontsize = 14, color = 'r')
    ylabel(r'$E_a[eV]$', font, fontsize=13)

    t1 = arange(0.0, 5.0, 0.1)

    energy_min = amin(energy_bond)
    energy_max = amax(energy_bond)
    energy_wid = abs(energy_max - energy_min)
    grid_min = amin(grid_space)
    grid_max = amax(grid_space)
    grid_wid = abs(grid_max - grid_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             energy_min-0.05*energy_wid, energy_max+0.05*energy_wid])

    subplot(312)
    plot(grid_space, bond_length, 'r^')
    ylabel('b[Ang]', font)
    axhline(y = m[2], color = 'b')
    text(grid_space[len(grid)/2], m[2], 'exp', font, color='k')

    bond_min = amin(bond_length)
    bond_max = amax(bond_length)
    bond_wid = abs(bond_max - bond_min)
    bond_limit_min = bond_min - 0.05 * bond_wid
    bond_limit_max = bond_max + 0.05 * bond_wid

    if bond_limit_max < m[2]:    
        bond_limit_max = m[2] + 0.1 * bond_wid
    elif bond_limit_min > m[2]:
        bond_limit_min = m[2] - 0.1 * bond_wid
        
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             bond_limit_min, bond_limit_max])

    subplot(313)
    plot(grid_space, frequency, 'ms')
    xlabel('h[Ang]', font)
    ylabel(r'$h\omega[meV]$', font, fontsize=13)

    freq_min = amin(frequency)
    freq_max = amax(frequency)
    freq_wid = abs(freq_max - freq_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             freq_min-0.05*freq_wid, freq_max+0.05*freq_wid])

    savefig('%s-Dimer-%s-Report.png' %(s, xc))
    show()
    
    dir = os.getcwd()
    dir += '/%s-Dimer-%s-Report.png' %(s, xc)

    report = """\
Dimer test
=============

The bond length between the two identical atoms are varied in order to
determine the values of the bond length, the atomization energy and the
vibrational frequency. These are plotted against the gridspace for %s. The %s
exchange correlation functional has been used. The figure is shown below:

.. figure:: %s

""" % (s, xc, dir)

    return report
