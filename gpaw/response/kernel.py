from gpaw.utilities.blas import gemmdot
import numpy as np

def calculate_Kxc(xc, gd, nt_sG, npw, Gvec_Gc, nG, vol, bcell_cv):
    """LDA kernel"""

    # get fxc(tilden(r))
    assert np.abs(nt_sG[0].shape - nG).sum() == 0
    
    fxc_sg = np.zeros_like(nt_sG)
    xc.calculate_fxc(gd, nt_sG, fxc_sg)
    fxc_g = fxc_sg[0]

    # FFT fxc(r)
    nG0 = nG[0] * nG[1] * nG[2]
    tmp_g = np.fft.fftn(fxc_g) * vol / nG0

    r_vg = gd.get_grid_point_coordinates()
    
    Kxc_GG = np.zeros((npw, npw), dtype=complex)
    for iG in range(npw):
        for jG in range(npw):
            dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
            if (nG / 2 - np.abs(dG_c) > 0).all():
                index = (dG_c + nG) % nG
                Kxc_GG[iG, jG] = tmp_g[index[0], index[1], index[2]]
            else: # not in the fft index
                dG_v = np.dot(dG_c, bcell_cv)
                dGr_g = gemmdot(dG_v, r_vg, beta=0.0) 
                Kxc_GG[iG, jG] = gd.integrate(np.exp(-1j*dGr_g)*fxc_g)

    return Kxc_GG / vol
                
