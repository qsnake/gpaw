"""This module defines different external potentials to be used in 
time-independent and time-dependent calculations."""

class ExternalPotential:
    """ External potential

    """

    def add_linear_field(self, pt_nuclei, a_nG, b_nG, strength, kpt):
        """Adds (does NOT apply) linear field 
        f(x,y,z) = str_x * x + str_y * y + str_z * z to wavefunctions.

        Parameters
        ----------
        pt_nuclei: List of ?LocalizedFunctions?
            Projectors (paw.pt_nuclei)
        a_nG:
            the wavefunctions
        b_nG:
            the result
        strength: float[3]
            strength of the linear field
        kpt: KPoint
            K-point
        """

        # apply local part of x to smooth wavefunctions psit_n
        for i in range(kpt.gd.n_c[0]):
            x = (i + kpt.gd.beg_c[0]) * kpt.gd.h_c[0]
            b_nG[:,i,:,:] += (strength[0] * x) * a_nG[:,i,:,:]

        # FIXME: combine y and z to one vectorized operation,
        # i.e., make yz-array and take its product with a_nG

        # apply local part of y to smooth wavefunctions psit_n
        for i in range(kpt.gd.n_c[1]):
            y = (i + kpt.gd.beg_c[1]) * kpt.gd.h_c[1]
            b_nG[:,:,i,:] += (strength[1] * y) * a_nG[:,:,i,:]

        # apply local part of z to smooth wavefunctions psit_n
        for i in range(kpt.gd.n_c[2]):
            z = (i + kpt.gd.beg_c[2]) * kpt.gd.h_c[2]
            b_nG[:,:,:,i] += (strength[2] * z) * a_nG[:,:,:,i]


        # apply the non-local part for each nucleus
        for nucleus in pt_nuclei:
            if nucleus.in_this_domain:
                # position
                x_c = nucleus.spos_c[0] * kpt.gd.domain.cell_c[0]
                y_c = nucleus.spos_c[1] * kpt.gd.domain.cell_c[1]
                z_c = nucleus.spos_c[2] * kpt.gd.domain.cell_c[2]
                
                # apply linear x operator
                nucleus.apply_linear_field( a_nG, b_nG, kpt.k,
                                            strength[0] * x_c 
                                            + strength[1] * y_c
                                            + strength[2] * z_c, strength )

            # if not in this domain
            else:
                nucleus.apply_linear_field(a_nG, b_nG, kpt.k, None, None)



    # BAD, VERY VERY SLOW, DO NOT USE IN REAL CALCULATIONS!!!
    def apply_scalar_potential(self, pt_nuclei, a_nG, b_nG, func, kpt):
        """Apply scalar function f(x,y,z) to wavefunctions. BAD

        NOTE: BAD, VERY VERY SLOW, DO NOT USE IN REAL CALCULATIONS!!!
        The function is approximated by a low-order polynomial near nuclei.

        Currently supports only quadratic (actually, only linear as
        nucleus.apply_polynomial support only linear)::
        
          p(x,y,z) = a + b_x x + b_y y + b_z z 
                       + c_x^2 x^2 + c_xy x y
                       + c_y^2 y^2 + c_yz y z
                       + c_z^2 z^2 + c_zx z x 


        The polynomial is constructed by making a least-squares fit to
        points (0,0,0), 3/8 (r_cut,0,0), sqrt(3)/4 (r_cut,r_cut,r_cut), and 
        to points symmetric in cubic symmetry. (Points are given relative to 
        the nucleus).
        """

        # apply local part to smooth wavefunctions psit_n
        for i in range(kpt.gd.n_c[0]):
            x = (i + kpt.gd.beg_c[0]) * kpt.gd.h_c[0]
            for j in range(kpt.gd.n_c[1]):
                y = (j + kpt.gd.beg_c[1]) * kpt.gd.h_c[1]
                for k in range(kpt.gd.n_c[2]):
                    z = (k + kpt.gd.beg_c[2]) * kpt.gd.h_c[2]
                    b_nG[:,i,j,k] = func.value(x,y,z) * a_nG[:,i,j,k]

        # apply the non-local part for each nucleus
        for nucleus in pt_nuclei:
            if nucleus.in_this_domain:
                # position
                x_c = nucleus.spos_c[0] * kpt.gd.domain.cell_c[0]
                y_c = nucleus.spos_c[1] * kpt.gd.domain.cell_c[1]
                z_c = nucleus.spos_c[2] * kpt.gd.domain.cell_c[2]
                # Delta r = max(r_cut) / 2
                # factor sqrt(1/3) because (dr,dr,dr)^2 = Delta r
                rcut = max(nucleus.setup.rcut_j)
                a = rcut * 3.0 / 8.0
                b = 2.0 * a / npy.sqrt(3.0)
                
                # evaluate function at (0,0,0), 3/8 (r_cut,0,0),
                # sqrt(3)/4 (r_cut,r_cut,rcut), and at symmetric points 
                # in cubic symmetry
                #
                # coordinates
                coords = [ [x_c,y_c,z_c], \
                           [x_c+a, y_c,   z_c], \
                           [x_c-a, y_c,   z_c], \
                           [x_c,   y_c+a, z_c], \
                           [x_c,   y_c-a, z_c], \
                           [x_c,   y_c,   z_c+a], \
                           [x_c,   y_c,   z_c-a], \
                           [x_c+b, y_c+b, z_c+b], \
                           [x_c+b, y_c+b, z_c-b], \
                           [x_c+b, y_c-b, z_c+b], \
                           [x_c+b, y_c-b, z_c-b], \
                           [x_c-b, y_c+b, z_c+b], \
                           [x_c-b, y_c+b, z_c-b], \
                           [x_c-b, y_c-b, z_c+b], \
                           [x_c-b, y_c-b, z_c-b] ]
                # values
                values = npy.zeros(len(coords))
                for i in range(len(coords)):
                    values[i] = func.value( coords[i][0],
                                            coords[i][1],
                                            coords[i][2] )
                
                # fit polynomial
                # !!! FIX ME !!! order should be changed to 2 as soon as
                # nucleus.apply_polynomial supports it
                nuc_poly = Polynomial(values, coords, order=1)
                #print nuc_poly.c
                
                # apply polynomial operator
                nucleus.apply_polynomial(a_nG, b_nG, self.k, nuc_poly)
                
            # if not in this domain
            else:
                nucleus.apply_polynomial(a_nG, b_nG, self.k, None)


