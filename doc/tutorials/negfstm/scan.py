from ase import *
from gpaw import *
import pickle
from gpaw.transport.jstm import STM

# tip data
tip = GPAW('tip', txt=None)
h1, s1 = pickle.load(open('tip_hs.pckl')) # h and s matrix
h10, s10 = pickle.load(open('tip_p_hs.pckl')) #principal layer h and s matrix

# surface data
srf = GPAW('srf', txt=None)
h2, s2 = pickle.load(open('srf_hs.pckl')) 
h20, s20 = pickle.load(open('srf_p_hs.pckl'))

# set up the stm-calculator
stm = STM(tip, srf,
          hs1=(h1[0], s1[0]),    # tip 
          hs10=(h10[0], s10[0]), # tip principal layer
          hs2=(h2[0], s2[0]),    # surface 
          hs20=(h20[0], s20[0]), # surface principal layer
          bias = 0.05,           # bias
          de = 0.05/7.,          # spacing of the energy grid at which the
                                 # Green's functions should be evaluated
          logfile='scan.log')    

stm.set(dmin=6) # set tip to surface distance
stm.initialize() 
stm.scan()

stm.linescan([[0, 0], [27, 27]]) # linescan along the [111] direction

# plot the current map and the linescan
stm.plot(label='I[nA]')



