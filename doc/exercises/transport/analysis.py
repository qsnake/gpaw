from gpaw.transport.analysor import Transport_Plotter
from pylab import *

plotter = Transport_Plotter()
plot(plotter.bias_steps[0].tc[0,0,0])
show()


