from math import pi,sqrt
from itertools import izip
from gpaw.utilities import hartree
from gpaw.utilities.blas import gemmdot
from gpaw.atom.all_electron import AllElectron
from gpaw import extra_parameters
from gpaw.sphere import weights, points
import numpy as npy


def get_scaled_positions(atoms, positions):
   """COPY PASTE FROM ASE! Get positions relative to unit cell.
   
   Atoms outside the unit cell will be wrapped into the cell in
   those directions with periodic boundary conditions so that the
   scaled coordinates are beween zero and one."""
   
   scaled = npy.linalg.solve(atoms._cell.T, positions.T).T
   for i in range(3):
      if atoms._pbc[i]:
         scaled[i] %= 1.0
   return scaled

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
         g.vr[1:] /= g.r[1:]
         g.vr[0] = g.vr[1]
         for r, vKS,vr in zip(r_g,vKS_g, g.vr):
            print >> f, r, vKS,vr, (vKS-vr)

      f.close()

   def grid_to_radial(self, a, gd, f_g):
      bohr_to_ang = 1/1.88971616463

      # Coordinates of an atom
      atom_c = self.paw.atoms.get_positions()[a]

      
      # Get xccorr for atom a
      setup = self.paw.density.setups[a]
      xccorr = setup.xc_correction
      
      radf_g = npy.zeros(xccorr.ng)
      for w,p in zip(weights, points):
         # Very inefficient loop
         for i, r in enumerate(xccorr.rgd.r_g):
            # Obtain the position of this integration quadrature point in specified grid
            pos_c = atom_c + (r * bohr_to_ang) * p
            # And in scaled coordinates 
            scaled_c = get_scaled_positions(self.paw.atoms, pos_c)
            # Use scaled coordinates to interpolate (trilinear interpolation) correct value
            radf_g[i] += w * gd.interpolate_grid_point(scaled_c, f_g)
      return radf_g
      
   def get_spherical_ks_potential(self,a):
      # If the calculation is just loaded, density needs to be interpolated
      if self.paw.density.nt_sg is None:
         print "Interpolating density"
         self.paw.density.interpolate()
         
      # Get xccorr for atom a
      setup = self.paw.density.setups[a]
      xccorr = setup.xc_correction

      # Get D_sp for atom a
      D_sp = self.paw.density.D_asp[a]

      # density a function of L and partial wave radial pair density coefficient
      D_sLq = gemmdot(D_sp, xccorr.B_Lqp, trans='t')

      # The 'spherical' spherical harmonic
      Y0 = 1.0/sqrt(4*pi)

      # Generate cartesian fine grid xc-potential
      print "Generate cartesian fine grid xc-potential"
      gd = self.paw.finegd
      vxct_g = gd.zeros()
      self.paw.hamiltonian.xc.get_energy_and_potential(self.paw.density.nt_sg[0], vxct_g)

      # ---------------------------------------------
      # The Electrostatic potential                  
      # ---------------------------------------------
      # V_ES(r) = Vt_ES(r) - Vt^a(r) + V^a(r), where
      # Vt^a = P[ nt^a(r) + \sum Q_L g_L(r) ]       
      # V^a = P[ -Z\delta(r) + n^a(r) ]             
      # P = Poisson solution
      # ---------------------------------------------

      print "Evaluating ES Potential..."
      # Make sure that the calculation has ES potential
      # TODO
      if self.paw.hamiltonian.vHt_g == None:
         raise "Converge tha Hartree potential first."
      
      # Interpolate the smooth electrostatic potential from fine grid to radial grid
      radHt_g = self.grid_to_radial(a, gd, self.paw.hamiltonian.vHt_g)

      print "D_sp", D_sp

      # Calculate the difference in density and pseudo density
      dn_g = Y0 * (xccorr.expand_density(D_sLq, xccorr.n_qg, xccorr.nc_g, xccorr.ncorehole_g).n_sLg[0][0]
                   - xccorr.expand_density(D_sLq, xccorr.nt_qg, xccorr.nct_g).n_sLg[0][0])
      
      # Calculate the Hartree potential for this
      vHr = npy.zeros((xccorr.ng,))
      hartree(0, dn_g * xccorr.rgd.r_g * xccorr.rgd.dr_g, setup.beta, setup.ng, vHr)

      # Add the core potential contribution
      vHr -= setup.Z
      vHr[1:] /= xccorr.rgd.r_g[1:]
      vHr[0] = vHr[1]

      # Calculate the compensation charge contribution
      comp = self.paw.density.Q_aL[a][0] * setup.wg_lg[0] / Y0
      comp[1:] /=  xccorr.rgd.dv_g[1:]
      comp[0] = comp[1]
      vHr -= comp
      
      radHt_g += vHr
      
      # --------------------------------------------
      # The XC potential                           
      # --------------------------------------------
      # V_xc = Vt_xc(r) - Vt_xc^a(r) + V_xc^a(r)   
      # --------------------------------------------

      print "Evaluating xc potential"
      # Interpolate the smooth xc potential  from fine grid to radial grid
      radvxct_g = self.grid_to_radial(a, gd, vxct_g)

      # Arrays for evaluating radial xc potential slice
      e_g = npy.zeros((xccorr.ng,))
      vxc_sg = npy.zeros((len(D_sp), xccorr.ng))

      # Create pseudo/ae density iterators for integration
      n_iter = xccorr.expand_density(D_sLq, xccorr.n_qg, xccorr.nc_g, xccorr.ncorehole_g)
      nt_iter = xccorr.expand_density(D_sLq, xccorr.nt_qg, xccorr.nct_g)
      
      # Take the spherical average of smooth and ae radial xc potentials
      for n_sg, nt_sg, integrator in izip(n_iter,
                                          nt_iter,
                                          xccorr.get_integrator(None)):
         # Add the ae xc potential
         xccorr.calculate_potential_slice(e_g, n_sg, vxc_sg)
         radvxct_g += integrator.weight * vxc_sg[0]
         # Substract the pseudo xc potential
         xccorr.calculate_potential_slice(e_g, nt_sg, vxc_sg)
         radvxct_g -= integrator.weight * vxc_sg[0]

      radvks_g = radvxct_g + radHt_g
      return (xccorr.rgd.r_g, radvks_g)

if not extra_parameters.get('usenewxc'):
    raise "New XC-corrections required. Add --gpaw usenewxc=1 to command line and try again."

