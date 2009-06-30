from math import pi
from itertools import izip
from gpaw.utilities import hartree
from gpaw.atom.all_electron import AllElectron
import numpy as npy

class AllElectronPotential:
   def __init__(self, paw):
      self.paw = paw
      
   def write_spherical_ks_potentials(self, txt):
      f = open(txt,'w')
      for a in self.paw.density.D_asp:
         r_g, vKS_g = self.get_spherical_ks_potential(a)
         setup = self.paw.density.setups[a]
         # Calculate also atomic LDA for reference
         g = AllElectron(setup.symbol, xcname='LDA',nofiles=True, txt=None)
         g.run()
         g.vr += setup.Z
         g.vr /= g.r
         for r, vKS,vr in zip(r_g,vKS_g, g.vr):
            print >> f, r, vKS,vr, vKS-vr
      f.close()
      
   def get_spherical_ks_potential(self,a):
   
      # Get D_sp for atom a
      D_sp = self.paw.density.D_asp[a]
      setup = self.paw.density.setups[a]
      # Get xccorr for atom a
      xccorr = setup.xc_correction

      vxc_sg = npy.zeros((len(D_sp), xccorr.ng))
      radvxc_sg = npy.zeros((len(D_sp), xccorr.ng))
      radn_sg =  npy.zeros((len(D_sp), xccorr.ng))
      vHr = npy.zeros((xccorr.ng,))
      Etot = 0
      integrator = xccorr.get_integrator(None)
      for n_sg, i_slice in izip(xccorr.expand_density(D_sp),
                                integrator):
         E = xccorr.get_energy_and_potential_slice(n_sg, vxc_sg)
         w, Y_L = i_slice
         radn_sg += Y_L[0] * 4*pi*w * Y_L[0] * n_sg
         radvxc_sg += Y_L[0] * 4*pi*w * Y_L[0] * vxc_sg

      print radvxc_sg
      print n_sg
      hartree(0, radn_sg[0] * xccorr.rgd.r_g * xccorr.rgd.dr_g, setup.beta, setup.ng, vHr)
      #vHr -= setup.Z
      vHr[1:] /= xccorr.rgd.r_g[1:]
      vHr[0] = vHr[1]
      vKS_g = vHr + radvxc_sg[0]
      print "The multipole corrections to spherical potential (a constant function) is missing!!"
      return (xccorr.rgd.r_g, vKS_g)
