from bulkfit import bulkfit
import sys
import os
import pickle
import matplotlib
matplotlib.use('TkAgg')
from pylab import *


def bulk_report(filename):
    
    """bulk_report function.
    
    Load the pickle file made from the bulk function, where the results of the
    heavy calculation are saved. Make use of the bulkfit function to determine
    the values of the min lattice constant and the related cohesive energy. In
    the end plot those values for the variation of the grid spacings, h.
    """

    f = open(filename)
    object = pickle.load(f)
           
    lat = object['Lattice constants']
    s = object['Atomic symbol']
    xc = object['Exchange-correlation functional']
    coh = object['Cohesive energies']
    grid = object['Grid spacings']
    crys = object['Crystal type']
    kpoint = object['Number of k-points']
 
    grid_space = []
    coh_energy = []
    lat_cons = []
    bulk_mod = []

    for i in range(len(grid)):

        h = grid[i]
        c = coh[i]
        a = lat[i]
     
        fa, fEc, fB = bulkfit(a, c, crys)

        lat_cons.append(fa)
        coh_energy.append(fEc)
        bulk_mod.append(fB)
        grid_space.append(h)
        
        i += 1

    # The figure is plotted:
    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}

    figure(figsize = (10, 10))
    subplot(311)
    plot(grid_space, coh_energy, 'ro')
    title("%s in %s structure with %s" %(s, crys, xc), font)
    ylabel(r'$E_c[eV]$', font, fontsize=12)
   
    coh_min = amin(coh_energy)
    coh_max = amax(coh_energy)
    coh_wid = abs(coh_max - coh_min)
    grid_min = amin(grid_space)
    grid_max = amax(grid_space)
    grid_wid = abs(grid_max - grid_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             coh_min-0.05*coh_wid, coh_max+0.05*coh_wid])
  
    subplot(312)
    plot(grid_space, lat_cons, 'g^')
    ylabel('a[Ang]', font)
   
    lat_min = amin(lat_cons)
    lat_max = amax(lat_cons)
    lat_wid = abs(lat_max - lat_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             lat_min-0.05*lat_wid, lat_max+0.05*lat_wid])
    
    subplot(313)
    plot(grid_space, bulk_mod, 'ms')
    xlabel('h[Ang]', font)
    ylabel('B[GPa]', font)
   
    bulk_min = amin(bulk_mod)
    bulk_max = amax(bulk_mod)
    bulk_wid = abs(bulk_max - bulk_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             bulk_min-0.05*bulk_wid, bulk_max+0.05*bulk_wid])
  
    savefig('%s-Bulk-%s-%s-Report.png' %(s, crys, xc))
    show()
    
    dir = os.getcwd()
    dir += '/%s-Bulk-%s-%s-Report.png' %(s, crys, xc)

    text = """\
Bulk test
============

The lattice constant and the cohesive energy are plotted against the
gridspace for %s. The %s exchange correlation functional has been
used.  The bulk is a %s crystal structure. The figure is shown below:

.. figure:: %s

""" % (s, xc, crys, dir)

    return text
