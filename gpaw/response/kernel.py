"""Defines kernel. Not in use now. """
def fxc(self, n):
    
    name = self.xc
    nspins = self.nspin

    libxc = XCFunctional(name, nspins)
   
    N = n.shape
    n = np.ravel(n)
    fxc = np.zeros_like(n)

    libxc.calculate_fxc_spinpaired(n, fxc)
    return np.reshape(fxc, N)


def calculate_Kxc(self, gd, nt_G):
    # Currently without PAW correction
    
    Kxc_GG = np.zeros((self.npw, self.npw), dtype = complex)
    Gvec = self.Gvec

    fxc_G = self.fxc(nt_G)

    for iG in range(self.npw):
        for jG in range(self.npw):
            dG = np.array([np.inner(Gvec[iG] - Gvec[jG],
                          self.bcell[:,i]) for i in range(3)])
            dGr = np.inner(dG, self.r)
            Kxc_GG[iG, jG] = gd.integrate(np.exp(-1j * dGr) * fxc_G)
            
    return Kxc_GG / self.vol
                
