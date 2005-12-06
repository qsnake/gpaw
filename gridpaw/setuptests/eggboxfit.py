import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
from pylab import *


def eggboxfit(filename):
    
    """Eggbox report.
    
    Load the pickle file made by the eggbox function. Calculate the delta
    energy (the difference between max and min of the total energy) and the max
    force for each grid spacing, h, at varying positions. Then plots those
    values as function of the grid spacings, h.
    """

    file = open(filename)
    object = pickle.load(file)

    h = object['Grid spacings']
    s = object['Atomic symbol']
    e = object['Total energies']
    xc = object['Exchange-correlation functional']

    fmax = []
    delta_e = []
    position = []

    for i in range(len(h)):
        
        tot = max(e[i]) - min(e[i])
        delta_e.append(tot)
    
        f = (e[i,1:] - e[i,:-1]) / (h[i] / 101)
        fm = max(abs(f))
        fmax.append(fm)
        
    energy = e[len(h)/2,:]
    grid = h[len(h)/2]

    for j in range(101):
        k = j / 100.0
        position.append(k)

    # The figure are plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}
    
    figure(figsize = (10, 10))
    subplot(311)
    plot(h, delta_e, 'ro')
    title("%s eggbox-test with %s" %(s, xc), font)
    ylabel(r'$\Delta E_{max}[meV]$', font)

    delta_min = amin(delta_e)
    delta_max = amax(delta_e)
    delta_wid = abs(delta_max - delta_min)
    h_min = amin(h)
    h_max = amax(h)
    h_wid = abs(h_max - h_min)
    axis([h_min-0.05*h_wid, h_max+0.05*h_wid,
             delta_min-0.05*delta_wid, delta_max+0.05*delta_wid])
    
    subplot(312)
    plot(h, fmax, 'g^')
    xlabel('h[Ang]', font)
    ylabel(r'$F_{max}[eV/Ang]$', font)
    axhline(y = 0.05, color = 'b')
    text(h[1], 0.05, 'max force', font, color='k')

    f_min = amin(fmax)
    f_max = amax(fmax)
    f_wid = abs(f_max - f_min)
    f_limit_min = f_min - 0.1 * f_wid
    f_limit_max = f_max + 0.1 * f_wid

    if f_limit_max < 0.05:    
        f_limit_max = 0.05 + 0.2 * f_wid
    elif f_limit_min > 0.05:
        f_limit_min = 0.05 - 0.2 * f_wid
    
    axis([h_min-0.05*h_wid, h_max+0.05*h_wid,
             f_limit_min, f_limit_max])

    subplot(313)
    plot(position, energy, 'ms')
    xlabel('/h[Ang]', font)
    ylabel('E[eV]', font)

    energy_min = amin(energy)
    energy_max = amax(energy)
    energy_wid = abs(energy_max - energy_min)
    position_min = amin(position)
    position_max = amax(position)
    position_wid = abs(position_max - position_min)
    axis([position_min-0.05*position_wid, position_max+0.05*position_wid,
             energy_min-0.005, energy_max+0.005])

    savefig('%s-%s-eggbox-test.png' %(s, xc))
    show()
    
    dir = os.getcwd()
    dir += '/%s-%s-eggbox-test.png' %(s, xc)

    report = """\
Eggbox test
=============

The maximum energy difference and force relative to its position in the grid
space are plotted against the gridspace for %s. The %s exchange correlation
functional has been used. The figure is shown below:

.. figure:: %s

""" % (s, xc, dir)

    return report
