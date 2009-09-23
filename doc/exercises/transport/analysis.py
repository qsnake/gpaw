from gpaw.transport.analysor import Transport_Plotter
from pylab import *

plotter = Transport_Plotter('bias', 'Pt_H2_nsc.dat')
plotter.set_ele_steps(None, 0)
plotter.plot_ele_step_info('ham', [0], 0,0)

