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
         g = AllElectron(setup.symbol, xcname='LDA',nofiles=True, scalarrel=True, txt=None)
         g.run()
         print >>f, r_g[0], vKS_g[0], g.vr[0], 0.0
         for r, vKS,vr in zip(r_g[1:],vKS_g[1:], g.vr[1:]):
            print >> f, r, vKS,vr, (vKS-vr)/r

      f.close()

   def grid_to_radial(self, a, gd, f_g):
      bohr_to_ang = 1/1.88971616463

      # Coordinates of an atom
      atom_c = self.paw.atoms.get_positions()[a]

      
      # Get xccorr for atom a
      setup = self.paw.density.setups[a]
      xccorr = setup.xc_correction
      
      radf_g = npy.zeros(xccorr.ng)
      target_g = npy.zeros(xccorr.ng)
      
      for w,p in zip(weights, points):
         scaled_nc = []
         # Very inefficient loop
         for i, r in enumerate(xccorr.rgd.r_g):
            # Obtain the position of this integration quadrature point in specified grid
            pos_c = atom_c + (r * bohr_to_ang) * p
            # And in scaled coordinates 
            scaled_c = get_scaled_positions(self.paw.atoms, pos_c)
            scaled_nc.append(scaled_c)

         scaled_nc = npy.array(scaled_nc)

         gd.interpolate_grid_points(scaled_nc, f_g, target_g, use_mlsqr=True)
         radf_g += w * target_g

      return radf_g
      
   def get_spherical_ks_potential(self,a):
      #self.paw.restore_state()

      print "XC:", self.paw.hamiltonian.xc.xcfunc.xcname
      if not self.paw.hamiltonian.xc.xcfunc.xcname == 'LDA':
         raise NotImplementedError

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
      radHt_g *= xccorr.rgd.r_g
      
      print "D_sp", D_sp

      # Calculate the difference in density and pseudo density
      dn_g = Y0 * (xccorr.expand_density(D_sLq, xccorr.n_qg, xccorr.nc_g, xccorr.ncorehole_g).n_sLg[0][0]
                   - xccorr.expand_density(D_sLq, xccorr.nt_qg, xccorr.nct_g).n_sLg[0][0])

      # Add the compensation charge contribution
      dn_g -= Y0 * self.paw.density.Q_aL[a][0] * setup.g_lg[0]
      
      # Calculate the Hartree potential for this
      vHr = npy.zeros((xccorr.ng,))
      hartree(0, dn_g * xccorr.rgd.r_g * xccorr.rgd.dr_g, setup.beta, setup.ng, vHr)

      # Add the core potential contribution
      vHr -= setup.Z

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

      radvks_g = radvxct_g*xccorr.rgd.r_g + radHt_g
      return (xccorr.rgd.r_g, radvks_g)

class CoreEigenvalues(AllElectronPotential):
   
   def get_core_eigenvalues(self, a, scalarrel=True):
      """Return the core eigenvalues by solving the radial schrodinger equation.

      Using AllElectron potential class, the spherically averaged Kohn--Sham potential
      is obtained around the spesified atom. The eigenstates for this potential are solved,
      the the resulting core states returned. Still experimental.
      
      """
      
      r, v_g = self.get_spherical_ks_potential(a)

      # Get xccorr for atom a
      setup = self.paw.density.setups[a]
      xccorr = setup.xc_correction
      symbol = setup.symbol

      # Create AllElectron object for eigensolver
      atom = AllElectron(symbol, txt=None, scalarrel=scalarrel)
      # Calculate initial guess
      atom.run()

      # Set the potential
      atom.vr[:len(v_g)] = v_g
      # After the setups cutoff, arbitrary barrier is used
      atom.vr[len(v_g):] = 10.0

      # Solve the eigenstates
      atom.solve()

      # The display format is just copy paste from AllElectron class
      # TODO: Make it a method in AllElectron class, thus it can be called directly
      def t(a):
         print a

      t('Calculated core eigenvalues of atom '+str(a)+':'+symbol)
      t('state      eigenvalue         ekin         rmax')
      t('-----------------------------------------------')
      for m, l, f, e, u in zip(atom.n_j, atom.l_j, atom.f_j, atom.e_j, atom.u_j):
         # Find kinetic energy:
         k = e - npy.sum((npy.where(abs(u) < 1e-160, 0, u)**2 * #XXXNumeric!
                            atom.vr * atom.dr)[1:] / atom.r[1:])

         # Find outermost maximum:
         g = atom.N - 4
         while u[g - 1] >= u[g]:
            g -= 1
         x = atom.r[g - 1:g + 2]
         y = u[g - 1:g + 2]
         A = npy.transpose(npy.array([x**i for i in range(3)]))
         c, b, a = npy.linalg.solve(A, y)
         assert a < 0.0
         rmax = -0.5 * b / a

         s = 'spdf'[l]
         t('%d%s^%-4.1f: %12.6f %12.6f %12.3f' % (m, s, f, e, k, rmax))
      t('-----------------------------------------------')
      t('(units: Bohr and Hartree)')
      return atom.e_j
      
if not extra_parameters.get('usenewxc'):
    raise "New XC-corrections required. Add --gpaw usenewxc=1 to command line and try again."

