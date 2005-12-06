from hcpfit import hcpfit
import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
from pylab import *


def hcp_report(filename):
    
    """hcp_report function
    Load the pickle file made from the hcp function, where the results of the
    heavy calculation are saved. Make use of the hcpfit function to determine
    the values of the minimum lattice constant (a and c/a), the related
    cohesive energy and Bulk Modulus at each grid point set. In the end plot
    those values for the variation of the mean value of the grid spacing h.
    """

    f = open(filename)
    object = pickle.load(f)
           
    lattice = object['Lattice constants']
    covera = object['Covera c/a']
    s = object['Atomic symbol']
    xc = object['Exchange-correlation functional']
    cohesive = object['Cohesive energies']
    grid_space = object['Grid spacing meanvalues']
    kpoint = object['Number of k-points']
   
    coh_energy = []
    lat_cons = []
    covera_cons = []
    bulk_mod = []

    for i in range(len(grid_space)):

        coh = cohesive[i]
 
        fa, fcov, fB, fEc = hcpfit(lattice, covera, coh)

        coh_energy.append(fEc)
        lat_cons.append(fa)
        covera_cons.append(fcov)
        bulk_mod.append(fB)

    # The figure are plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}
    
    figure(figsize = (10, 10))
    subplot(411)
    plot(grid_space, coh_energy, 'ro')
    title("%s hcp with %s" %(s, xc), font)
    ylabel(r'$E_c[eV]$', font, fontsize=13)
   
    coh_min = amin(coh_energy)
    coh_max = amax(coh_energy)
    coh_wid = abs(coh_max - coh_min)
    grid_min = amin(grid_space)
    grid_max = amax(grid_space)
    grid_wid = abs(grid_max - grid_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             coh_min-0.05*coh_wid, coh_max+0.05*coh_wid])
  
    subplot(412)
    plot(grid_space, lat_cons, 'g^')
    ylabel('a[Ang]', font)
   
    lat_min = amin(lat_cons)
    lat_max = amax(lat_cons)
    lat_wid = abs(lat_max - lat_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             lat_min-0.05*lat_wid, lat_max+0.05*lat_wid])

    subplot(413)
    plot(grid_space, covera_cons, 'g^')
    ylabel('c/a', font)
   
    covera_min = amin(covera_cons)
    covera_max = amax(covera_cons)
    covera_wid = abs(covera_max - covera_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             covera_min-0.05*covera_wid, covera_max+0.05*covera_wid])

    subplot(414)
    plot(grid_space, bulk_mod, 'ms')
    xlabel('h[Ang]', font)
    ylabel('B[GPa]', font)
   
    bulk_min = amin(bulk_mod)
    bulk_max = amax(bulk_mod)
    bulk_wid = abs(bulk_max - bulk_min)
    axis([grid_min-0.05*grid_wid, grid_max+0.05*grid_wid,
             bulk_min-0.05*bulk_wid, bulk_max+0.05*bulk_wid])
  
    savefig('%s-hcp-%s-Report.png' %(s, xc))
    show()
    
    dir = os.getcwd()
    dir += '/%s-hcp-%s-Report.png' %(s, xc)

    text = """\
HCP test
============

The lattice constant a, the covera as c/a, the cohesive energy and the bulk
modulus are plotted against the mean value of the gridspace for %s. The %s
exchange correlation functional has been used. It is a hcp crystal structure.
The figure is shown below:

.. figure:: %s

""" % (s, xc, dir)

    return text
