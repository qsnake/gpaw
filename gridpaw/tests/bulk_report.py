from bulkfit import bulkfit
import sys
import os
import Numeric as num
import pickle
import matplotlib
matplotlib.use('TkAgg')
import pylab as py


def bulk_report(filename):
    
    """bulk_report function
    Load the pickle file made from the bulk function, where the results of the
    heavy calculation are saved. Make use of the bulk fit function to determine
    the values of the min lattice constant and the related cohesive energy. In
    the end plot those values for the variation of the grid spacing h.
    """

    f = open(filename)
    object = pickle.load(f)
           
    lat = object['Lattice constants']
    s = object['Atomic symbol']
    xc = object['exchange-correlation']
    coh = object['Cohesive energies']
    grid = object['Grid space']
    n = object['number of calc']
    crys = object['crystal type']
    kpoint = object['kpoint set']
   
    f_a = open('lattice_constants.dat', 'w')
    f_b = open('bulk_modulus.dat', 'w')
    f_c = open('cohesive_energies.dat', 'w')
   
    Grid_space = []
    Coh_energy = []
    Lat_cons = []
    Bulk_mod = []

    for i in range(len(grid)):

        h = grid[i]
        c = coh[i]
        a = lat[i]

        Grid_space.append(h)
                
        fa, fEc, fB = bulkfit(a, c, n, crys)

        print >> f_a, h, fa
        print >> f_b, h, fB
        print >> f_c, h, fEc

        Coh_energy.append(fEc)
        Lat_cons.append(fa)
        Bulk_mod.append(fB)
 
        i += 1

    # The figure is plotted

    font = {'fontname'   : 'Courier',
            'color'      : 'b',
            'fontweight' : 'bold',
            'fontsize'   : 11}

    py.subplot(311)
    py.plot(Grid_space, Coh_energy, 'ro')
    py.title("%s bulk as %s structure with %s" %(s, crys, xc), font)
    py.ylabel('Energy (eV)', font)
   
    Coh_min = py.amin(Coh_energy)
    Coh_max = py.amax(Coh_energy)
    Coh_wid = abs(Coh_max - Coh_min)
    Grid_min = py.amin(Grid_space)
    Grid_max = py.amax(Grid_space)
    Grid_wid = abs(Grid_max - Grid_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Coh_min-0.05*Coh_wid, Coh_max+0.05*Coh_wid])
  
    py.subplot(312)
    py.plot(Grid_space, Lat_cons, 'g^')
    py.ylabel('Lattice constant', font)
   
    Lat_min = py.amin(Lat_cons)
    Lat_max = py.amax(Lat_cons)
    Lat_wid = abs(Lat_max - Lat_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Lat_min-0.05*Lat_wid, Lat_max+0.05*Lat_wid])
    
    py.subplot(313)
    py.plot(Grid_space, Bulk_mod, 'ms')
    py.xlabel('Grid space (Aangstroem)', font)
    py.ylabel('Bulk mod (eV/Aa)', font)
   
    Bulk_min = py.amin(Bulk_mod)
    Bulk_max = py.amax(Bulk_mod)
    Bulk_wid = abs(Bulk_max - Bulk_min)
    py.axis([Grid_min-0.05*Grid_wid, Grid_max+0.05*Grid_wid,
             Bulk_min-0.05*Bulk_wid, Bulk_max+0.05*Bulk_wid])
  
    py.savefig('%s-Bulk-%s-%s-Report.png' %(s, crys, xc))
    py.show()
    
    dir = os.getcwd()
    dir += '/%s-Bulk-%s-%s-Report.png' %(s, crys, xc)

    text = """\
Bulk test
============

The lattice constant and the cohesive energy are plotted against the
gridspace for %s. The %s exchange correlation functional has been
used.  The bulk is a %s crystal structure. The figure is shown below:

.. image:: %s

""" % (s, xc, crys, dir)

    return text
    

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
 
    bulk_report(filename)



